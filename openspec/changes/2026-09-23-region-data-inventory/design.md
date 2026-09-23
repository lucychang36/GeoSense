# 设计 — 2026-09-23-region-data-inventory

## D1 region 开放语义：注册表美化 + 空间求交判定（开放-封闭）

用户核心约束：**region 支持全国甚至全世界**。拆成两个正交问题分别解决：

- **判定问题（region 是什么、有没有数据）→ 空间求交，天然开放**。任意地名/任意 bbox 与 manifest 影像 bbox 做 shapely `intersects/contains`，全国全球都不需要登记。这与 W2 两个变更同构：THEMES 注册表让新专题加 dict 条目、overlay 协议让新场景走同一条 SSE 通道，本提案让新区域**加数据即生效、零代码**。
- **显示问题（region 的中文名/描述）→ REGION_META 注册表，封闭部分最小化**。只有人类可读文案需要登记：

```python
REGION_META = {
    "szbay": {"label": "深圳湾", "desc": "深圳湾河口-深圳河流域"},
    "zhengzhou_hightech": {"label": "郑州高新区", "desc": "郑州高新技术产业开发区（bbox 近似）"},
}
# 未注册 region 自动生成：label = region 原串，desc = "Sentinel-2 影像（{date_min}~{date_max}）"
```

- `manifest.json` 的 `region` 字段语义升级：自由字符串（影像所属区域的事实声明），不再受 REGION_PRESETS 枚举约束；REGION_PRESETS 保留于 `satellite_download.py`（下载侧的便捷入口），**查询侧不依赖它**。

## D2 manifest 唯一真相，STAC 为导出视图

**决策：inventory 查询读 `manifest.json`，不读 STAC。**

- 依据：本次金水区案例的根因之一就是 STAC 与 manifest 漂移（郑州两景只在 manifest）。STAC 重建依赖手动跑脚本，任何「读 STAC」的查询路径都会重蹈 stale 覆辙。
- manifest 写入侧（`satellite_download.py`）每次下载即更新，天然与磁盘同源。
- STAC 保留：互操作标准视图（对外交换/求职演示价值），T1 重建使其不过时；写清 docstring「export view, source of truth = manifest.json」。
- 反向保护：inventory 输出含 `manifest_mtime`，便于诊断「目录里有文件但 manifest 没登记」的脏状态。

## D3 区域解析三级回退（地名 → bbox）

`resolve_region_bbox(region: str) -> ResolveResult`，逐级回退：

1. **admin_boundary 精确匹配**：`WHERE name = :region`（全国省市县入库后覆盖绝大多数中文地名，含「金水区」「郑州市」「河南省」）。
2. **admin_boundary 模糊匹配**：`name LIKE %:region%`。**命中 >1 条必须返回候选列表**（朝阳区歧义红旗 R1），由 LLM/用户消歧，工具签名含 `candidates[]` 字段。
3. **内置 GEO_FALLBACK 词表**：少量高频非行政区名（深圳湾、雄安新区等 admin_boundary 没有的自定义区域），硬编码 bbox dict——与 REGION_PRESETS 分离（下载侧 vs 查询侧）。

全未命中：返回 `resolved=False` + 诚实指引「无法解析地名，请提供 bbox=[w,s,e,n]」——**全球范围的支持方式是 bbox 直给**，不为境外地名做 geocode 服务（scope 红线）。

## D4 inventory 查询与输出协议

```python
def data_inventory_tool(region: str = "", bbox: str = "") -> str:
    # bbox 优先于 region；返回 JSON 字符串（与既有工具风格一致）
```

输出结构（LLM 友好的紧凑 JSON）：

```json
{
  "query": "郑州金水区",
  "resolved": true,
  "query_bbox": [113.66, 34.76, 113.85, 34.90],
  "scenes": [
    {"id": "S2A_49SGU_20230626_0_L2A", "region": "zhengzhou_hightech",
     "date": "2023-06-26", "cloud": 5.2, "bbox": [113.5, 34.73, 113.7, 34.88],
     "intersection": "partial", "bands": ["B2","B3","B4","B8","B11","B12"]}
  ],
  "coverage": "none",
  "recommendation": "查询区域无影像覆盖。最接近的可用区域：zhengzhou_hightech（郑州高新区，2023-06-26/2025-06-27 两期，bbox 113.5-113.7E 与查询区西缘相交 0.04°）",
  "candidates": []
}
```

- `coverage`: `full` / `partial` / `none`（按查询 bbox 被影像 bbox 覆盖面积比 ≥0.95 / >0 / =0 分档，阈值常量声明）。
  > **apply 修订（2026-09-23 实测驱动）**：原拟对 partial 加面积下限（<0.2 判 none），实测发现深圳湾自定义框（0.336）与金水区相交（0.337）比例几乎相同——**纯阈值无法区分「数据自属区域 bbox 近似」与「非自属区域窄条相交」**。改为 `region_known` 身份判定：查询区域映射到 REGION_META 标签或 ALIAS_REGION 别名 → 数据自属区域，partial 不拒绝（bbox 近似是已知红线）；非自属区域 partial → recommendation 警告「仅 X% 覆盖，不足以支撑全区口径」+ 最近邻替代。金水区实测 34% 覆盖 → 警告 + 推荐 zhengzhou_hightech。
- `recommendation` 生成规则：无覆盖时取**同省/最近邻**已登记 region（centroid 欧氏距离即可，不引 shapely 巨计算）；日期年份不在两期范围内的场景由 LLM 依据 scenes 日期自行说明（2015 无解是数据事实，工具只给日期清单）。
- 求交实现：shapely `box` × 逐 scene `box.intersects` —— scenes 数量 ~10，纯几何零 IO。

## D5 state 不放大：inventory 结果不入 state 大对象

inventory 输出只作为工具返回值进 LLM 上下文（一次对话 <2 KB），**不新增 state 字段**——与 `_diff_masks` 抽离（D5 上案）同理，state 只进结果性轻量引用。data_node prompt 注入「inventory 是只读探查，不要把 JSON 原文贴进回复，转译为自然语言 + 表格」。

## D6 admin_boundary 全国化：下载器 + 导入 + 缓存

`scripts/fetch_admin_boundaries.py`：

- 数据源：阿里云 DataV GeoAtlas（`https://geo.datav.aliyun.com/areas_v3/bound/{adcode}.json`）——省级 `100000_full`、地市级逐省 `_full`。
- 落盘：`data/admin/ china_provinces.geojson` + `china_cities.geojson`（缓存，支持 `--from-cache` 离线重放，红旗 R3）。
- 导入：`seed_postgis.py` 扩展 `--admin-geojson` 参数，`INSERT ... ON CONFLICT (name, level) DO UPDATE`；level 字段区分 `district`（深圳 10 区，保留）/ `city` / `province`。
- 查询侧按 level 降序精确匹配（区县优先于省市，避免「朝阳区」先撞上同名街道级条目）。
- 世界级：`level` 枚举预留 `country`，导入函数对任意 GeoJSON 通用（机制开放），本次不提供世界数据（non-goal）。

## D7 agent 接入：prompt 纪律 + 路由改造

- `langchain_tools.py` 新增 `data_inventory_tool` 并注册进 data_node 工具集。
- `prompts.py` / data_node SYSTEM 注入纪律：「涉及具体地理区域的问句，若不确定数据可用性，**先调 data_inventory_tool**，基于返回决定：有数据→继续分析流程；无数据→按 recommendation 转译诚实拒绝 + 主动推荐替代区域」。
- `REGION_KEYWORDS` 路由（W2 加的郑州关键词表）**降级为回退**：inventory 求交优先——否则每来一个新城市就要加关键词，违背开放语义。data_node 直接以 inventory 返回的 `region` 字段喂给 analysis 节点参数。
- 拒绝话术模板（planner/analysis 共享）：「{query} 无数据 → 推荐替代：{region_label} {dates} → 询问是否切换」，复用本次金水区实测的良好行为并补上「郑州高新区有数据」的事实层。

## D8 测试策略

- `scripts/data_inventory.py` 自带 selftest（`python data_inventory.py --selftest`）：合成 bbox 场景覆盖 full/partial/none/歧义候选/未解析五分支 + 断言 `bool()` 包裹（项目铁律）。
- 真实数据校验：金水区查询 → recommendation 含 `zhengzhou_hightech`；杭州查询 → `resolved=False`。
- e2e 新场景 `--jinshui`：问句「对比郑州金水区 2015 和 2025 的建筑用地变化」→ 断言回复含「郑州高新区/高新区」与「2023」「2025」、**不含幻觉面积数字**（正则 `\d+\.\d+\s*km` 不出现于推荐句）；深圳湾回归场景并行全绿。
- STAC 重建校验：items 数、日期、六波段与 manifest 逐字段 diff 为零。
