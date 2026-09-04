# Design：第10月 W2 自动符号化引擎（auto_symbology）

> 状态：design 定稿，待 apply
> 前置：proposal.md（路线 A + 双格式已收敛）
> 参照：计划伪代码 L979 `generate_style(data_type, analysis_result)` 四步中的 2-3 步（选配色 + 选符号化），LLM 步是 W4

## Context

W1 复盘红旗③"制图靠人肉"的具体病灶：4+ 处硬编码（cyanobacteria 6 处 RGBA /
cartography_node 变化纯红 / change_detection `cmap="gray"` / render_mask 默认 CMAP 采样）。
W2 建一个**纯规则引擎**（零 LLM、可复现、可自测）作为统一决策层；
W4 Text-to-Map 时 LLM 只负责"理解自然语言意图"，选色仍复用本引擎的规则表。

## D1 范围决策

- 只接 1 个消费方：`multi_agent/agents.py` 的 `cartography_node`（binary 分支验证）。
- cyanobacteria_monitor / render_mask **不迁移**：W4 已验证交付物，回归成本 > 收益；
  统一渲染归 W3 `auto_cartography.py`（完整制图流水线的题中之义）。
- 不新增第三方依赖：ColorBrewer palette 以**硬编码 hex 字典**内置（~8 个方案），
  不装 colorbrewer / palettable 包（项目依赖纪律）。

## D2 数据模型

```python
@dataclass
class DataProfile:
    kind: str            # "categorical" | "sequential" | "diverging" | "binary"
    n_classes: int       # categorical: unique 数；其它为 0
    vmin: float          # 连续: 值域下限
    vmax: float          # 连续: 值域上限
    labels: list[str]    # categorical: 类别名（可能为空 → 机械 palette）

@dataclass
class SymbologyPlan:
    kind: str
    palette_name: str        # "YlGnBu" | "RdBu" | "Set2" | "semantic" | ...
    colors: list[str]        # hex 串（#RRGGBB），categorical=逐类，连续=断点色
    colorblind_safe: bool
    reason: str              # 一句话决策理由（日志/LLM 消费用）
    semantic_hits: dict      # {"水": "#3787C0", ...} 语义覆盖记录（审计用）
```

## D3 决策规则与优先级（引擎灵魂）

`profile_data(values, labels=None, force_kind=None)` 判定顺序：

1. `force_kind` 显式指定 → 直接采用（逃生舱，测试与特殊场景用）；
2. dtype 为 bool → **binary**；
3. 整数型且 unique ≤ 10 → **categorical**（labels 为空则记 `[]`）；
4. float 且 `vmin < 0 < vmax`（值域跨 0）→ **diverging**；
5. 其余 float → **sequential**。

`choose_symbology(profile)` 分派：

| kind | 规则 | 默认 palette |
|------|------|-------------|
| binary | 高对比双档：背景透明 + 事件色（语义"变化"=红） | 语义红 `#E24B4A` |
| categorical | **先查 SEMANTIC 语义表**（逐类命中则覆盖）；未命中类从机械 qualitative 补位 | Set2（CB-safe 系） |
| diverging | 值域跨 0 → 双端色 + 中点 `#F7F7F7` | RdBu |
| sequential | 按数据语义选族（labels 含"植被/NDVI"→ Greens；风险/密度 → OrRd；缺省） | YlGnBu |

- SEMANTIC 初始表（8 条）：水=蓝 `#3787C0` / 植被=绿 `#4C9F38` / 城市=灰 `#8C8C8C` /
  建成区=灰（同城市）/ 新增=红 `#E24B4A` / 消退=蓝 `#3787C0` / 持续=黄 `#E6B93C` / 变化=红。
- colorblind_safe：内置白名单 {YlGnBu, OrRd, RdBu, BrBG, Greens, cividis} → True；
  Set2 → False（诚实标注，注释说明 ColorBrewer 官方定性 CB-safe 方案有限）。

## D4 双格式输出

```python
to_matplotlib(plan, n=None) -> matplotlib.colors.Colormap
# categorical/binary → ListedColormap(colors)；连续 → LinearSegmentedColormap.from_list

to_mapbox_style(plan, field: str, breaks: list[float] | None = None) -> dict
# categorical → {"fill-color": ["match", ["get", field], v0, c0, ..., fallback]}
# 连续       → {"fill-color": ["interpolate", ["linear"], ["get", field], b0, c0, ...]}
# breaks 缺省由 vmin/vmax 均分 5 档生成。骨架仅 fill-color（source/layers 归 W4）。
```

## D5 demo 设计（三联对比图，真实数据）

`data/output/symbology_demo.png`，1×3 + 每联下色卡行（palette 名 + reason，展示"引擎自己解释决策"）：

| 联 | 数据 | 预期判定 | 预期方案 |
|----|------|---------|---------|
| 1 | NDVI（szbay_real_20250727，(B8−B4)/(B8+B4)） | sequential（float 不跨 0） | YlGnBu |
| 2 | U-Net 分类（unet_finetuned.pt + infer_full 同景） | categorical（0/1/2）+ 语义命中 | 水=蓝/城=灰/植=绿 |
| 3 | NDVI 差值（2025−2023，无云交集） | diverging（值域跨 0） | RdBu |

- 相对 proposal 的细化：proposal 写第三联为"变化 mask（diverging）"，design 落地为
  **NDVI 差值**——布尔变化 mask 只有 0/1，走 binary；NDVI 差值才是天然的 diverging
  （变绿正/变褐负）。cartography_node 接入即 binary 分支的验证，与三联互补，
  决策表 4 分支全部被 demo+接入覆盖。
- U-Net 推理 ~4s（MPS），可接受；无云交集规则复用 W8 约定（valid & ~cloud 交集）。

## D6 消费接入（cartography_node）

```python
from scripts.auto_symbology import profile_data, choose_symbology
plan = choose_symbology(profile_data(change.astype(np.uint8), labels=["不变", "变化"]))
bg[change.astype(bool)] = matplotlib.colors.to_rgb(plan.colors[-1])   # "变化"→语义红
```

- 视觉结果与 W1 相同（变化=红），但**颜色来源从硬编码变成"引擎按数据性质+语义表推导"**，
  reason 进入 step_log 可审计。
- 布尔 mask 走 `astype(np.uint8)` 进 profile（bool→binary 同样可行，取 uint8+labels
  是为了让"变化"语义命中）。

## D7 测试策略（项目无 pytest，沿用脚本自测模式）

`scripts/auto_symbology.py --selftest`：断言全覆盖（独立于 demo，秒级、零数据依赖）：

1. bool → binary；2. 0/1/2 + labels[水,城,植] → categorical 且 colors[0] 为语义蓝域；
3. 3 类无标签 → Set2 机械色；4. 值域 [−1,1] → diverging RdBu；
5. 值域 [0,0.8] → sequential YlGnBu；6. force_kind="diverging" 覆盖 sequential；
7. to_mapbox_style：categorical 输出 match 表达式 / 连续输出 interpolate 表达式，
   结构合法（list 首元素、hex 以 # 开头、断点数 = 颜色数）；
8. colorblind_safe 标记与白名单一致。

集成验证：demo 三联图落盘（三段 palette 名互异）+ multi_agent demo 重跑
**change_ratio 67.64% 不变**（引擎只改色不改算法）。

## 风险与红旗（沿 proposal，design 补充）

- 语义表 8 条是初始集，遇"耕地/湿地"回落机械色（W3 扩表点）；
- RdBu 中点 `#F7F7F7` 在真彩叠底上可能近白难辨——demo 中 diverging 直接 imshow（非叠底）规避；
- to_mapbox_style 的 match 分支值类型是原始类别值（int），W4 接入前端时需与 layer data 对齐。
