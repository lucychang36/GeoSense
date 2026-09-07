# Proposal：第10月 W4 Text-to-Map（text_to_map）

> 状态：**applied（2026-09-07，验证证据见 tasks.md；验收 5 条全过：selftest 13/13、CLI demo 三主题互异零回退、起服务 curl style 事件 + 浏览器实测双主题上图清旧层、multi_agent 回归 67.64% 不变、W2/W3 引擎零改动）**
> 日期：2026-09-07
> 决策：方案 B（意图-渲染分层：LLM 只做 NL→受限枚举 IR，Style JSON 由确定性代码组装）+ **SSE 事件驱动统一入口**（用户指定 demo 直接渲染在 frontend/index.html，经质询修订：废弃独立控件方案，走 /api/chat 的 style 事件）

## Why

- 学习计划第10月 W4：`text_to_map.py`——自然语言→Mapbox 样式 JSON。这是阶段 4（自动制图）的收官环节，也是前三个 W2/W3 零 LLM 节点后的第一个 LLM 主场。
- W2 留了 fill-color/interpolate 骨架、W3 留了 symbol-sort-key 标注骨架（代码注释明确"W4 直接组合"）——本节点回收两处伏笔：**LLM 负责语义映射（NL→枚举），确定性代码负责语法（枚举→合法 Style Spec）**。
- 业界 Text-to-X 的标准模式是约束生成而非自由生成：Mapbox Style Spec 是强 schema 语言，让 LLM 直出完整 JSON 必然幻觉非法字段。分层后每个字段可白名单校验，幻觉面积最小。

## 需求（What）

### 1. 引擎 `scripts/text_to_map.py`（独立可跑，核心交付物）

- **CartographyIR**（dataclass，全部受限枚举字段）：
  - `theme`: `"poi" | "ndvi" | "cog"`（三主题各回收一块骨架，见 design D5）
  - `visible_classes`: poi 主题显示哪些类（subway/park/school 子集）
  - `label_classes`: 标注哪些类（⊆ visible_classes，走 W3 避让）
  - `color_intent`: 语义覆盖意图 → 只能引用 W2 SEMANTIC/PALETTES 已注册条目
  - `cog`: 瓦片化 COG 文件名（白名单 = data/cogs/manifest.json）
  - `raster_opacity`: 0.0~1.0（cog 主题）
  - `region`: `"深圳湾" | "深圳全市"`（bbox 枚举）
  - `rationale`: LLM 的一句话决策理由（answer 流展示）
- **extract_intent(query) -> CartographyIR**：DeepSeek temp=0 + JSON 输出 + 三道校验闸（parse → 枚举白名单 → 引用存在性），失败重试 1 次 → 回退默认方案（poi 全类默认色）
- **assemble_style(ir) -> dict**：纯确定性组装完整 Style Spec v8 JSON（sources + layers）：
  - poi → circle 层（复用 W2 to_mapbox_style 的 match 表达式对象，键名换 circle-color）+ W3 to_mapbox_labels symbol 层
  - ndvi → 栅格降采样网格多边形（辅助函数 `raster_to_grid`）+ W2 interpolate fill 层
  - cog → raster source 指向 :8001 瓦片服务 + raster-opacity
- **CLI demo**：3 条自然语言（每主题 1 条）→ 3 份完整 Style JSON 落盘 `data/output/text_to_map/` + 断言三份结构互异
- **--selftest**：组装器纯确定性全测（IR 校验拒非法枚举 / 表达式结构断言 / 三主题骨架断言 / LLM 不在场可全跑）

### 2. Agent 工具接入（模式同 temporal_change_tool）

- `backend/agent/langchain_tools.py`：注册 `text_to_map_tool`（SPATIAL_TOOLS 9→10），返回 JSON **summary 字段前置**（防 result 事件 [:200] 截断），异常→`[工具错误]`+可枚举 theme 提示（不抛异常防炸 ReAct）
- `graph.py` SPATIAL_SYSTEM_PROMPT 增加一条使用规则（第 10 条工具的触发说明）

### 3. SSE 协议扩展 + 前端渲染（用户指定 demo 通道）

- `backend/api/main.py` `_stream`：ToolMessage 处理分支——工具 JSON 含 `map_style` 时 yield **新事件类型 `style`**（全量 `{rationale, sources, layers, layer_ids}`，不经过 result 的 [:200] 摘要路径）
- `frontend/index.html` `handleEvent` 加 style 分支：先移除上一批 layer_ids → addSource/addLayer 注入 → fitBounds；欢迎语增加示例问法（~50 行，克制）

## 验收标准

1. `--selftest` 断言全过：IR 校验器拒绝非法枚举/越界 opacity/未注册语义引用；三主题 assemble 输出结构断言（poi 含 match circle-color + symbol 层；ndvi 含 interpolate fill 层；cog 含 raster-opacity）；LLM 不在场 selftest 可独立跑
2. CLI demo：3 条 NL → 3 份 Style JSON 落盘且主题互异（结构断言），extract_intent 走真实 DeepSeek 调用（temp=0）
3. **起服务实测 SSE**（单测直调不暴露路由/SSE 问题——第3月 mount catch-all 教训）：curl /api/chat 发制图请求，收到 `style` 事件且 JSON 可 parse、含 sources+layers；前端浏览器实测渲染出专题图层且不破坏既有 COG/POI 图层
4. multi_agent 回归：现有工具行为不变（temporal_change_tool 等 9 个工具零改动，仅追加）
5. W2/W3 引擎零改动（仅 import 复用）

## 非目标

- 不做蓝藻 mask → 风险区多边形化（raster_to_grid 已覆盖"栅格→面"教学点，精确轮廓化超范围）
- 不做完整底图 Style 文档替换（setStyle 会清掉已验证图层；诚实定位为 Text-to-Overlay，完整 basemap 组合超教学重心）
- 不做标注引线/旋转标签（W3 已声明局限，继承）
- 不做多轮对话式改图（"把公园改成紫色"这种增量编辑是 W5+ 话题，v1 每次全量重生成）
- 不改 frontend 既有图层与交互逻辑（只追加 handleEvent 分支 + 欢迎语）

## 涉及文件

| 文件 | 动作 |
|------|------|
| `scripts/text_to_map.py` | 新增（~350 行） |
| `backend/agent/langchain_tools.py` | 追加 text_to_map_tool（~40 行） |
| `backend/agent/graph.py` | SPATIAL_SYSTEM_PROMPT +1 条规则 |
| `backend/api/main.py` | _stream 增加 style 事件分支（~10 行） |
| `frontend/index.html` | handleEvent style 分支 + 欢迎语（~50 行） |
| `README.md` | 阶段4 进度 W4 + 关键数据块 + 项目结构 + 快速开始 + 里程碑（阶段4收官） |
| `openspec/changes/2026-09-07-text-to-map/` | proposal / design / tasks |

## 诚实红旗（预登记）

1. LLM 对训练外语义表述可能选错枚举（"生态好不好"→哪个 theme），temp=0 + prompt 举例缓解不根除；回退默认方案保证可用性
2. planner 工具路由依赖 LLM 识别制图意图——temporal_change_tool 已验证该模式，但误触发/漏触发概率非零（验收 3 的 curl 用例覆盖正例）
3. style 事件全量 JSON 体积几十 KB 级（W3 labels 53 Feature inline geojson），SSE 可承受但比摘要事件重；后续数据量大时需改"事件带 id + 前端拉取"
4. `raster_to_grid` 网格多边形化是**中心点采样近似**，格网分辨率（~40×40）是教学取舍，边缘锯齿不可避免
5. 真实渲染验证依赖网络 + MAPBOX_TOKEN 配额 + :8001 瓦片服务在线（cog 主题），CI 式回归不能依赖浏览器实测——selftest 只测 JSON 结构
