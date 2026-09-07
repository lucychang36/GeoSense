# Design：text_to_map（第10月 W4）

## D1 架构：意图-渲染分层（explore 方案 B，用户已确认）

```
NL「给深圳湾做一张NDVI图，植被茂密用绿色」
  → planner（graph 已有 LLM 调用，零新增）识别制图意图
  → text_to_map_tool（langchain_tools 注册，模式同 temporal_change_tool）
      └ scripts/text_to_map.py
          extract_intent(query)   ← DeepSeek temp=0，只产出受限枚举 IR
          validate_ir(ir)         ← 三道闸，确定性
          assemble_style(ir)      ← 纯函数，枚举→合法 Style Spec v8
  → SSE 新事件 style（全量 JSON）→ 前端注入渲染
```

分工铁律：**LLM 只负责"语义→枚举"，语法（枚举→合法 Style Spec）100% 由代码保证。**
LLM 永远不直接生成 hex 色、表达式、字段名——它只在白名单内选择。

## D2 CartographyIR schema

```python
@dataclass
class CartographyIR:
    theme: str              # "poi" | "ndvi" | "cog"
    visible_classes: list[str]   # ⊆ {"subway","park","school"}（poi 主题）
    label_classes: list[str]     # ⊆ visible_classes（走 W3 避让）
    color_intent: str | None     # 语义意图 → 必须命中 SEMANTIC/PALETTES 已注册条目
    cog: str | None              # ⊆ data/cogs/manifest.json 文件名
    raster_opacity: float        # 0.0~1.0（cog 主题）
    region: str                  # "深圳湾" | "深圳全市" → bbox/center 枚举
    rationale: str               # LLM 一句话决策理由
```

- 枚举与 W3 对齐：`fclass` 用英文（subway/park/school），data/osm/poi_*.json + W3 `load_osm_layer` 直接复用。
- `color_intent` 不接受自由文本颜色：LLM 输出如 `"水"`/`"植被"`/`"高风险"`，映射查 W2 `SEMANTIC` 表；未命中 → 校验拒绝 → 重试 → 回退机械 palette。这是"约束生成"的教学核心：**LLM 的输出空间是有限集合，不是字符串**。

## D3 校验三道闸 + 失败路径

1. **parse 闸**：LLM 输出必须是合法 JSON（DeepSeek `response_format={"type":"json_object"}`）
2. **枚举闸**：theme/region ∈ 白名单；classes 子集关系；opacity ∈ [0,1]
3. **引用闸**：color_intent 命中 SEMANTIC；cog ∈ data/cogs/manifest.json

失败路径：单次重试（把校验错误信息拼进重试 prompt）→ 仍失败 → 回退默认 IR（poi 全类 + 默认语义色 + rationale="默认方案"）→ 工具返回中标注 `fallback: true`（诚实暴露降级，不静默）。

## D4 数据源（explore 红旗④ 的收敛结论）

| 主题 | 数据 | 来源 | 依赖 |
|------|------|------|------|
| poi | subway 388 / park 936 / school 1096（具名点） | `data/osm/poi_*.json`，复用 W3 `load_osm_layer` | 零外部依赖（引擎独立可跑） |
| ndvi | 栅格 → ~40×40 网格多边形 + ndvi 属性 | `data/cogs/szbay_real_mosaic.tif`，`raster_to_grid` 辅助函数（复用第5月 data_pipeline 的 COG 读取通路；apply 时核实实际库，rasterio/rioxarray 以现依赖为准） | 本地文件 |
| cog | 瓦片服务 | `http://127.0.0.1:8001/tiles/{z}/{x}/{y}.png?path=data/cogs/<ir.cog>`（前端 COG_TILE_URL 同源） | :8001 服务在线 |

蓝藻 mask 精确轮廓化 = 非目标（proposal 已声明）。

## D5 三主题 ↔ 骨架回收映射（本设计的伏笔回收表）

| theme | 消费的 W2/W3 骨架 | 生成的层 |
|-------|------------------|---------|
| poi | W2 `to_mapbox_style` 的 **match 表达式**（键名 circle-color 复用同一表达式对象——演示"表达式与图层类型解耦"）+ W3 `to_mapbox_labels` **symbol 层** | circle（分类着色）+ symbol（标注避让） |
| ndvi | W2 `to_mapbox_style` 的 **interpolate 表达式**（连续断点） | fill（网格多边形连续着色） |
| cog | 无表达式——raster-opacity 纯 paint 属性 | raster（卫星影像叠加） |

W2 的 interpolate 分支在 explore 初版一度无处安放（项目缺连续值多边形数据），`raster_to_grid` 网格化补上这块：NDVI 网格专题图是经典制图案例，且中心点采样的近似性诚实入红旗。

## D6 SSE 接入：新事件类型 `style`

`backend/api/main.py` `_stream` 的 ToolMessage 分支现状：summary[:200] + 条件附 geojson。追加（~8 行）：

```python
data = json.loads(_text_of(m.content))
if isinstance(data, dict) and "map_style" in data:
    yield _sse("style", data["map_style"])   # 全量，不经 [:200]
```

- **为何新事件而非复用 result**：result 的 payload 协议是"轻量摘要 + 可选 geojson"；style 是全量 sources+layers（几十 KB），语义与体积都不同。与 district 事件同理——按渲染语义分事件类型，是本项目 SSE 协议的既有演进模式。
- **为何 summary 字段前置**（W2 教训应用）：工具 JSON 首字段是 `summary`，前端 step 卡片与 result[:200] 截断都先看到人话摘要；`map_style` 放后。
- 工具返回结构：`{"summary": str, "rationale": str, "fallback": bool, "map_style": {"layer_ids": [...], "sources": {...}, "layers": [...]}}`

## D7 前端渲染（~50 行，克制注入）

`handleEvent` 加分支：

```js
} else if (ev.event === 'style') {
  this.applyStyle(ev.data);
}
```

`applyStyle(payload)`：
1. 清理：`this.styleLayerIds` 逐个 removeLayer/removeSource（下一轮 style 到来时先清旧——图层生命周期由前端记录 layer_ids 管理）
2. 注入：payload.sources 逐个 addSource → payload.layers 逐个 addLayer，id 全部加 `t2m-` 前缀防与既有图层冲突
3. fitBounds 到 region 枚举对应 bbox
4. answer 流照常显示 rationale（用户能看到 LLM 的决策理由）

欢迎语追加示例问法：「给深圳湾做一张 NDVI 专题图，植被茂密用绿色」。

## D8 验证矩阵

| 层 | 手段 | 覆盖 |
|----|------|------|
| 组装器（纯函数） | selftest 断言 | IR 校验拒绝/三主题骨架/表达式结构——LLM 不在场全跑 |
| extract_intent（LLM） | CLI demo 3 条真实 NL | temp=0 真实调用 + 落盘 JSON |
| SSE 通道 | **起服务 curl 实测**（单测直调函数不暴露路由/SSE 问题——第3月 mount catch-all 教训） | style 事件可达、JSON 可 parse |
| 前端渲染 | 浏览器实测（用户验收） | 图层出现、既有图层不破坏、二次提问清旧层 |
| 回归 | multi_agent demo 重跑 | 既有 9 工具零影响 |

## 红旗继承（proposal 预登记 5 条，不重复）
