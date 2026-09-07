# Tasks：auto_cartography

## T1 引擎核心 + selftest
- [x] `scripts/auto_cartography.py`：墨卡托工具 / Feature / load_osm_layer / load_districts / generalize / place_labels / to_mapbox_labels
- [x] `--selftest` 10 断言全过（design 9 项清单中碰撞项拆为 2 条）

## T2 真实数据 demo
- [x] main()：z10 全市 / z12 中心城区 / 区界综合 三联图 → data/output/auto_cartography_demo.png（179,807 B）
- [x] 脚本内硬断言：已放置标签两两包围盒不重叠（联1/联2 均通过）；控制台报告放置/丢弃数
- [x] W2 choose_symbology 配色复用（语义回落 Set2 行为记录）

## T3 ST_Simplify 对照
- [x] PostGIS 可用（geosense-postgis 容器拉起）：龙岗区边界 492 顶点三档容差对照

## T4 收尾
- [x] README：阶段4 进度 W3 + 关键数据块 + 项目结构 + 快速开始 6.10 + 技术选型 + 里程碑
- [x] tasks.md 勾选回填证据 + proposal 状态 applied
- [x] memory 追加 + git 提交（代码 + README + openspec，无 data/.workbuddy 混入）

## 完成后（证据回填）

- **selftest**：10/10 PASS（墨卡托往返 / m_per_px 减半 / NE 首候选 / 共点 9 要素溢出丢弃+标签两两不重叠 / 远离两点 / DP 单调 / 面积守恒 ratio=1.0000 / min-area / mapbox 骨架）
- **demo**：z10 全市地铁 388 → 放置 109 / 丢弃 279（72%）；z12 中心窗口 → 放置 53 / 丢弃率 87%；Mapbox symbol 骨架 53 Feature
- **ST_Simplify 对照**（龙岗区 492 顶点）：
  - z12/z14：自写（preserve_topology=True）vs `ST_SimplifyPreserveTopology` **逐点一致 Hausdorff=0.000 m**
  - z10（tol=141m）：`ST_Simplify`（默认**无拓扑保持**）249 顶点 vs 保拓扑 250 顶点，Hausdorff 65.253 m —— 同一算法家族、不同拓扑语义的实证发现
- **偏差记录**（对 proposal）：① 综合实验数据源从"OSM 公园多边形"切换为"PostGIS admin_boundary 区界"——发现 data/osm/ 当年 `out center` 下载无 way geometry（诚实记录于脚本注释与 README）② selftest 9→10 条（碰撞项拆分）
- **commit hash**：见 git log（本轮提交）
