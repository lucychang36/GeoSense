# Tasks：第10月 W2 自动符号化引擎（auto_symbology）

> 前置：proposal.md（路线 A + 双格式）+ design.md（D1-D7）
> 状态：**全部完成（2026-09-04，证据见下）**

## 实现

- [x] **T1 引擎核心**（`scripts/auto_symbology.py`，~330 行纯函数零 LLM）
  DataProfile/SymbologyPlan dataclass（实现补全：SymbologyPlan 增加 class_values +
  vmin/vmax 字段——D4 的 match 表达式与 breaks 缺省需要，design D2 漏列）；
  PALETTES 8 个 ColorBrewer hex 字典 + CB_SAFE 白名单 + SEMANTIC 语义表；
  profile_data（force_kind → bool→binary → int unique≤10 → 跨0→diverging → sequential）
  + choose_symbology（binary 语义红透明底 / categorical 先语义后 Set2 补位 /
  diverging RdBu / sequential 按语义选族）

- [x] **T2 双格式输出**
  to_matplotlib（ListedColormap / LinearSegmentedColormap）；
  to_mapbox_style（categorical→match、连续→interpolate、hex8→rgba() 转换）

- [x] **T3 自测** `--selftest` **8/8 PASS**（exit 0）
  bool→binary / 语义命中（水→B 通道最大）/ 无标签→Set2 / [-1,1]→RdBu /
  [0,0.8]→YlGnBu / force_kind 覆盖 / match+interpolate 结构合法 / CB-safe 标注一致

## 验证

- [x] **T4 demo 三联图** `data/output/symbology_demo.png`（569KB）
  联1 NDVI → **sequential YlGnBu**（force_kind，见下「教学发现」）/
  联2 U-Net 分类 → **semantic**（水=#3787C0 城市=#8C8C8C 植被=#4C9F38 全命中）/
  联3 NDVI 差值 → **diverging RdBu**（[-0.944, 0.941] 跨 0）
  —— 三段 palette 名互异 ✅（验收达成）
  **教学发现**：NDVI 单景自动判定得 diverging（水面 NDVI 负值、值域跨 0）——
  数据性质 ≠ 制图语义（负值来自「水的物理性质」非「反向植被活性」），
  联1 用 force_kind="sequential" 注入领域知识（demo 注释诚实记录）。
  另：labels 必须用语义表全称（"城"/"植" 单字不命中 → demo 对齐为 城市/植被）。

- [x] **T5 cartography_node 接入 + 回归**
  agents.py：变化色改 `choose_symbology(profile_data(change.astype(np.uint8),
  labels=["不变","变化"]))` → palette "Set2+semantic"（不变→Set2 补位、变化→语义红
  #E24B4A），reason 进 step_log；multi_agent demo 重跑 **change_ratio=67.64% 不变**
  （valid 334,583 px 一致）+ PNG 560KB 落盘 ✅

- [x] **T6 收尾**
  README（阶段4 标题 W2 完成 / W2 行 / W2 关键数据块 / 项目结构 / 快速开始 6.9 /
  技术选型「自动符号化」行 / 里程碑）；memory 追加；git 提交。

## apply 过程中的修复记录

1. selftest 断言 7 切片错位：match 表达式 [2::2] 取到的是类别值（int）而非颜色
   → 颜色在 [3::2]（值/色交替结构：0=match, 1=get, 2=v0, 3=c0, ...）
2. SymbologyPlan 缺 vmin/vmax（design D4 要求 breaks 缺省按值域均分，D2 数据模型
   漏列）→ 四处 return 补字段
3. title 手写 U+2212（−）字体缺字形 → 改 ASCII 连字符 + rcParams unicode_minus=False

## 完成后

- [x] proposal.md 状态更新为 applied（本文件即证据）
- [x] 向用户汇报：验收证据 + 实测关键数值 + 红旗
