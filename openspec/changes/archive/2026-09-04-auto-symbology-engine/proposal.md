# Proposal：第10月 W2 自动符号化引擎（auto_symbology）

> 状态：**archived（2026-09-04，验证证据见 tasks.md；实现与验收全过：selftest 8/8、三联 demo、multi_agent 回归 67.64% 不变）**
> 日期：2026-09-04
> 决策：路线 A（规则引擎 + demo + 接 1 个消费方）+ 输出双格式（matplotlib cmap + Mapbox Style JSON 骨架）

## Why

W1 复盘红旗③点名的"制图靠人肉 matplotlib"：项目内配色硬编码分散 4+ 处
（cyanobacteria_monitor 6 处 RGBA / cartography_node 变化纯红 / change_detection cmap="gray" /
unet render_mask 采样默认 CMAP），无"数据类型→配色方案"的决策层，颜色是拍脑袋选的。
学习计划第10月 W2 = `auto_symbology.py`：根据数据类型自动配色（ColorBrewer 规则）。
LLM 选色是 W4（Text-to-Map）的范畴，W2 引擎为纯规则（可复现、可单测）。

## 知识核心（本节点的可学知识）

数据类型 → palette 决策规则：

| 数据类型 | 判据 | ColorBrewer 类 | 默认方案 |
|----------|------|---------------|---------|
| 定性分类 | unique ≤ 10 | Qualitative | Set2/Dark2 + 语义约定覆盖 |
| 单极连续 | 连续、无有意义中点 | Sequential | YlGnBu（NDVI）/ OrRd（风险） |
| 双极连续 | 连续、0/中点有意义（变化差值、z） | Diverging | RdBu（新增红/消退蓝） |
| 二值事件 | 0/1 mask | Binary | 高对比单色 + 透明度叠加 |

语义约定层（领域知识注入，优先于机械 palette）：
水=蓝域 / 植被=绿域 / 城市=灰 / 新增=红 / 消退=蓝 / 持续=黄；
palette 优先选色盲安全子集（YlGnBu / OrRd / RdBu / cividis 等）。

## 需求（Must）

1. `scripts/auto_symbology.py`（~200 行，纯函数零 LLM）：
   - `profile_data(values, labels=None) -> DataProfile`：探查 unique 数 / 值域 / 是否含负 / 类别名
   - `choose_symbology(profile) -> SymbologyPlan`：决策 palette 名 + hex 列表 + colorblind_safe 标记 + 一句话决策理由
   - `to_matplotlib(plan, n) -> Colormap`：ListedColormap / LinearSegmentedColormap
   - `to_mapbox_style(plan, field, breaks) -> dict`：fill-color interpolate 表达式骨架（W4 消费）
   - `SEMANTIC` 语义约定表：分类数据先查语义，命中覆盖机械 palette
2. demo（`main`）：真实数据三联对比图 `data/output/symbology_demo.png`：
   NDVI（sequential）/ U-Net 分类 szbay_real_20250727（qualitative+语义）/ 两期变化 mask（diverging）
   —— 一图证明"数据变→方案变"。
3. 消费接入：`backend/agent/multi_agent/agents.py` 的 `cartography_node`
   变化区颜色改为引擎推导（diverging profile），语义不变（高异常=红）。

## 非目标（范围克制）

- 不迁移 `cyanobacteria_monitor.py` / `render_mask`（W4 已验证交付物，统一渲染归 W3 `auto_cartography.py`）
- 不调 LLM 选色（ColorBrewer 是有限规则表；LLM 进场是 W4）
- Mapbox Style 只到 fill-color interpolate 骨架，完整 Style Spec（source/layers）是 W4
- 不做标注避让 / 地图综合（W3 范围）

## 涉及文件

- 新增：`scripts/auto_symbology.py`
- 修改：`backend/agent/multi_agent/agents.py`（cartography_node 约十几行）
- 修改：`README.md`（进度表 W2 + 关键数据 + 项目结构 + 快速开始 + 里程碑）

## 验收标准

1. 决策表单测全覆盖：categorical / sequential / diverging / binary 四分支 + 语义 override 优先级
   （如"水"命中语义表 → 蓝域而非 Set2 机械色）
2. demo 三联图落盘且三段 palette 名互不相同（证明决策分支生效）
3. `to_mapbox_style` 输出含 `interpolate` 表达式与合法 hex 断点
4. multi_agent demo 重跑：change_ratio **67.64% 不变**（引擎只改色不改算法），PNG 正常落盘
