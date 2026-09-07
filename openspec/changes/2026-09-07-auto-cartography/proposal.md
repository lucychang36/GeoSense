# Proposal：第10月 W3 自动标注 + 地图综合（auto_cartography）

> 状态：**applied（2026-09-07，验证证据见 tasks.md；验收 4 条全过：selftest 10/10、demo 硬断言标签两两不重叠、ST_Simplify 对照含保拓扑语义发现、W2 引擎零改动）**
> 日期：2026-09-07
> 决策：路线 A（引擎 + 真实数据 demo，不改现有消费方）+ 标注输出双格式（matplotlib + Mapbox symbol layer 骨架）

## Why

- 学习计划第10月 W3：`auto_cartography.py` 完整制图流水线——① 自动标注（冲突检测与避让，MapboxGL 标注策略）② 地图综合（缩放级别自适应简化，simplify-geojson / PostGIS ST_Simplify）。
- W2 交付了符号化引擎（`auto_symbology.py`），但一张"成品图"还缺两件事：**要素该标谁、标在哪**（标注避让）与**缩放时要素该简化到什么程度**（综合）。W3 补齐这两环，流水线 = W2 符号化 + W3 标注/综合。
- 手写一遍标注避让与 Douglas-Peucker，才能理解 MapboxGL symbol 引擎与 ST_Simplify 在生产中"自动做到"的事。

## 需求（What）

新增 `scripts/auto_cartography.py`（纯函数、零 LLM、零新依赖——shapely 已有）：

1. **load_osm_layer**：Overpass JSON（node/way/relation）→ 统一 Feature（point/polygon + name + class）；读 `data/osm/poi_{subway,park,school}.json` 真实数据。
2. **generalize(features, zoom)**：Douglas-Peucker（shapely simplify），tolerance 按 zoom 换算（像素容差 × 米/像素）；min-area 过滤（低 zoom 小多边形隐藏）。
3. **place_labels(features, zoom)**：8 方位候选锚点（右上优先）+ 标签包围盒碰撞检测 + 优先级贪心（subway > park > school，同级按要素面积/名称）；返回 labels + dropped。
4. **render_map**：复用 W2 `choose_symbology` 给点层配色 + 标注绘制 → PNG。
5. **to_mapbox_labels**：symbol layer 骨架（text-field、icon/text-offset、symbol-sort-key 按优先级）——W4 Text-to-Map 直接组合 W2+W3 两份 JSON。
6. **main demo**：两档 zoom 三联图（z10 全市概览 388 地铁标注 + 丢弃率 / z12 中心城区避让效果 / 公园多边形综合 z10 vs z12 vs z14 对比）+ `--selftest`。

## 验收标准

1. `--selftest` 断言全过：碰撞检测正确性（两重叠点只放一个标签）；DP 简化顶点数单调不增、面积守恒在容差内；8 方位候选次序正确。
2. 真实数据 demo 落盘 `data/output/auto_cartography_demo.png`：三联 palette/zoom 互异，标注无重叠（脚本内断言输出标签两两包围盒不相交），报告丢弃率。
3. 综合对照实验：自写 DP vs PostGIS `ST_Simplify` 同一多边形，顶点数与形状偏差在报告内给出。
4. W2 引擎零改动（仅 import 复用）；`multi_agent` / `cyanobacteria` 均不动。

## 非目标

- 不做线状要素弯曲简化（项目暂无河流/道路矢量层）。
- 不做标注引线（leader line）与曲线标注——超教学范围，诚实标注为局限。
- 不迁移 cyanobacteria / multi_agent 渲染（保持已验证交付物稳定）。
- LLM 零参与——Text-to-Map 是 W4 范围。

## 涉及文件

| 文件 | 动作 |
|------|------|
| `scripts/auto_cartography.py` | 新增（~300 行） |
| `README.md` | 阶段4 进度 W3 + 关键数据块 + 项目结构 + 快速开始 + 里程碑 |
| `openspec/changes/2026-09-07-auto-cartography/` | proposal / design / tasks |

## 诚实红旗（预登记）

1. 标注避让是**贪心近似**，非全局最优（NP-hard）——密集区丢弃率会偏高，丢弃是特性不是 bug。
2. 包围盒碰撞按轴对齐矩形近似，未考虑标签旋转。
3. tolerance 按 Web Mercator 米/像素换算，高纬度变形未修正（深圳纬度影响 ~5%，可接受）。
4. OSM way/relation 坐标直接取 center 或 geometry，多边形合法性未校验（自相交多边形 simplify 行为未定义）。
