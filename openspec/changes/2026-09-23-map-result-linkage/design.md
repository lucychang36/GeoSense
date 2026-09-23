# design.md — 2026-09-23-map-result-linkage

## D1 overlay 通用协议（一次到位，渲染器按需）

SSE `overlay` 事件载荷：

```json
{
  "url": "/api/overlays/overlay_report_xxx.geojson",
  "fit_bounds": [113.50, 34.73, 113.70, 34.88],
  "legend": [
    {"key": "gain", "color": "#E6323C", "label": "新增建筑用地（疑似）"},
    {"key": "loss", "color": "#3C78EB", "label": "消失建筑用地（疑似）"}
  ],
  "render_hint": "fill"
}
```

- **语义完备原则**（W4 意图-渲染分层的空间版）：颜色/标签/限定词全部由后端产出，前端是通用渲染器，零领域硬编码。
- `render_hint` 枚举 `fill | circle | choropleth`，**本次只实现 fill**；circle 已由 result 事件承担；choropleth 留位不实现。
- `legend[].key` 即 GeoJSON feature properties 的 kind 字段——图例与数据同源。

## D2 图斑过滤进 THEMES（per-theme，禁全局常量）

`THEMES[key].min_patch_px`：builtup 默认 5（≥1968 m²，语义「一栋中型建筑起步」）；water 注册为 1（小水塘 1-2px 即真实水域）——阈值是领域知识，进专题注册表（纯数据），与指数/阈值/语义色并列。

管线实现（spike 实测过的路径）：

```python
# 1) 最小图斑过滤：scipy.ndimage.label + bincount（0.01s 级）
lab, _ = label(mask); sizes = np.bincount(lab.ravel()); sizes[0] = 0
filtered = (sizes >= min_patch_px)[lab]
# 2) 多边形化：rasterio.features.shapes（transform 直接映射 4326，无需重投影）
# 3) 简化：shapely simplify(5e-5, preserve_topology=True)（selftest 断言面积保持 ≥98%）
# 4) 属性注入：{"kind", "area_m2": px_count * px_m2, "theme", "period"}
```

**面积用像元计数 × px_m2**（不用几何面积）——与报告 `_pixel_area_km2` 口径一致，popup 数字与报告可对账（含中纬度 cos 缩放，无需重算）。

## D3 双口径（本次最重要的分析决策）

- **主数字 = 滤波后口径**：`index_change()` 内在差分后先做 min_patch_px 过滤，再统计 gain/loss/net km²——analysis、报告、地图三者同源（同一 filtered mask）。
- **原始口径保留披露**：metrics 增加 `raw_gain_km2 / raw_loss_km2 / raw_px / patch_px / patch_keep_ratio`；报告指标表新增「原始总量（含 <Npx 碎斑）」行。
- 语义升级：`min_patch_px` 过滤本身是去噪步骤，报告方法说明补充「已过滤 <N 像元（≥X m²）碎斑以抑制伪变化」。
- 数字闸门：change_ratio/档位词按滤波后主数字投影（AREA_BANDS 复用）；新增「碎斑占比」定性词（>80% 高 / >50% 中 / 其余低）进 qualitative，LLM 叙事可见——碎斑占比本身是质量信号。

## D4 落盘 + 路由（不走 SSE 嵌入）

- GeoJSON 落盘 `data/output/overlays/overlay_{report_id}.geojson`（实测 ~1 MB）——SSE 单消息嵌 1 MB 不合适，且落盘后自动获得缓存/重复下载能力。
- `GET /api/overlays/{name}`：复用 `download_report` 模式——`Path(name).name` 剥目录 + 后缀白名单 `{.geojson}` + `is_file()` 检查 + GZipMiddleware 自动压缩（minimum_size=1024 已覆盖）。

## D5 接线链（全部复用现成投影点）

```
cartography_node → state.overlay_path（新增 NotRequired 字段，state 不放大 GeoJSON）
  ↓ report_tool 投影（langchain_tools.py:187 同款）：返回 dict 加 overlay_url/bbox/legend
  ↓ main.py ToolMessage 分支：检测 overlay_url → _sse("overlay", {...})（与 report 事件并列）
  ↓ 前端 handleEvent 加 overlay 分支：fetch → addSource + fill/line 图层 + popup + 图例 + fitBounds
```

- supervisor 专题汇总同步加一行「地图叠加：已上图（N 个图斑）」。
- overlay 管线在 cartography 节点内执行（classify 结果已在手，零重复计算）；报告生成（report_worker）不受影响。

## D6 前端底图区域联动（scene 事件）

- overlay 事件同时携带 `basemap: {tile_path, bounds}`（后端从 REGION_PRESETS / manifest 反查分析区域的瓦片 path 与 bbox）。
- 前端换源：`map.removeSource('cog') → addSource('cog', {tiles:[新URL], bounds}) → addLayer 重建`（raster source 不可原地改 URL）+ `fitBounds`。
- 无匹配 region 时不下发 basemap（前端保持现状深圳湾）——向后兼容。

## D7 生命周期管理（W4 ids 模式复用）

- Vue data 增 `overlayLayerIds/overlaySourceIds`；`status` 事件（新一轮提问）清除上一轮 overlay 图层/源/图例——与 district 清层同款时序（先清后等新事件）。
- 图例 DOM：地图左下角浮层，由 legend 数组渲染，清层时同步移除。

## D8 旧伏笔回收

| 先例 | 本案回收点 |
|---|---|
| W2 SEMANTIC 语义表 | legend 颜色/标签进 THEMES（纯数据） |
| W4 图层 ids 生命周期 | overlay 图层管理同款 |
| 第11月 report 事件 + /api/reports 路由 | overlay 事件 + /api/overlays 同构 |
| 第9月 safe_cog_path 白名单 | overlays 路由防穿越同款 |
| W2 explore spike 实测定参数 | min_patch_px/simplify 容差全部来自实测非拍脑袋 |
