# Design：auto_cartography（自动标注 + 地图综合）

> 依据 proposal.md（路线 A + 双格式）。全部纯函数、零 LLM、零新依赖（shapely/psycopg 已有）。

## 数据模型

```python
@dataclass
class Feature:
    name: str            # OSM tags.name（无名的进不了标注流程，但可参与渲染）
    fclass: str          # subway / park / school（决定优先级与配色层）
    geom: BaseGeometry   # shapely：Point / Polygon / LineString（Web Mercator 米制）
    priority: int        # class 优先级：subway=0 < park=1 < school=2（数字小者优先）
```

- **坐标系决策（D1）**：内部全程 **Web Mercator 米制**（lonlat→mercator 墨卡托正变换），碰撞、简化、面积全在米制做——把"米/像素"变成一次乘法。matplotlib 绘图也用 Mercator 坐标 + `aspect='equal'`（墨卡托保形，等比显示即正确形状）；只有标注文本不需要反变换。
- **zoom→分辨率（D2）**：`m_per_px = 156543.03392 * cos(lat_ref) / 2**zoom`，lat_ref 取数据纬度中心（深圳 ~22.65°）；DP tolerance = `px_tolerance(1.0px) * m_per_px`；min-area = `(min_px(3px) * m_per_px)**2`。

## 算法决策

- **D3 load_osm_layer**：Overpass JSON 三类 element——`node`→Point(lat/lon)；`way.geometry` 闭合→Polygon、不闭合→LineString；无 geometry 的 way/relation→`center`→Point。无 name 的要素保留（渲染/避让用），无名不标注。
- **D4 place_labels（标注避让）**：
  - 候选锚点 8 方位，**右上优先**顺序固定：`NE, E, SE, N, NW, W, SW, S`（MapboxGL text-anchor 惯例，自测断言首候选 NE）。
  - 标签包围盒：宽 = `len(name) * font_px * mpp`（CJK 全宽近似），高 = `font_px * 1.2 * mpp`；anchor 决定盒相对锚点的偏移方向。
  - **碰撞两类**：① 已放置标签盒 ② 所有要素点位小盒（半径 2px）——标签不许压点。
  - 优先级贪心：`sort by (priority, -area, name)` 逐要素尝试 8 候选，全撞则**丢弃**（dropped 记录原因，宁缺不叠）。
- **D5 generalize**：Polygon/LineString 走 `geom.simplify(tolerance, preserve_topology=True)`（Douglas-Peucker）；Point 不动；min-area 过滤只作用于 Polygon。**面积守恒断言**：tolerance 1px 下面积比 ≥ 0.95。
- **D6 双格式输出**：matplotlib（PNG demo）+ `to_mapbox_labels(placed)` → symbol layer 骨架：`text-field: ["get","name"]`、`text-anchor`、`text-offset`、`symbol-sort-key: priority`（引擎端做最终避让，我们的排序就是它的输入质量）。
- **D7 ST_Simplify 对照（验收③）**：取 demo 用例同一公园多边形，插入 PostGIS 临时表 → `ST_Simplify(geom, tolerance_m)` → 取回；对比指标 = 顶点数、**Hausdorff 距离**（shapely `hausdorff_distance`，两条简化结果的形状偏差）。psycopg3 连接 `geosense/geosense@127.0.0.1:5432/geosense`（docker-compose 定义）；daemon 不可用则诚实降级为"仅本地 DP 结果"，不造假对照。

## demo 设计

`main()` 出三联图 `data/output/auto_cartography_demo.png`：

| 联 | 范围/zoom | 内容 | 验证点 |
|----|----------|------|--------|
| 1 | z10 全市 | subway 388 全点渲染 + 标注（大面积丢弃）| 丢弃率报告——"城市概览不标全部站点"的直观课 |
| 2 | z12 中心城区（福田-罗湖窗口 ~12km）| subway+park+school 三层标注 | 避让效果可见：placed 两两不重叠（**脚本内硬断言**）|
| 3 | 中心窗口内最大公园 | 原始多边形 vs z10/z12/z14 简化叠加 + 顶点数标注 | 顶点数单调减、形状渐次趋简 |

配色复用 W2：`profile_data(class 值, labels=["地铁","公园","学校"])` → **语义表无这三类 → Set2 机械回落**——正好实证 W2 红旗②"未收录类别回落机械色"的行为（诚实记录，不改 W2 语义表，属验收④零改动）。

## selftest 断言（≥8 条）

1. 墨卡托往返：lonlat→merc→lonlat 与原值误差 < 1e-6（度）。
2. m_per_px：z0≈156543·cos(lat)，z 每 +1 减半。
3. 候选顺序：首候选 NE（dx>0 且 dy>0）。
4. 两重叠点只放 1 个标签（碰撞正确性）。
5. 相距远的两点放 2 个（无误杀）。
6. DP 顶点数单调不增（同一线，tolerance 递增）。
7. DP 面积守恒：1px tolerance 下面积比 ≥ 0.95。
8. min-area 过滤：低 zoom 小多边形被隐藏、大多边形保留。
9. to_mapbox_labels：含 text-field 与 symbol-sort-key，长度 = placed 数。

## 验收映射

| proposal 验收 | 落点 |
|--------------|------|
| ① selftest 全过 | selftest() 9 断言 |
| ② demo 标注两两不重叠 + 丢弃率 | main() 内 assert + 标题/控制台报告 |
| ③ ST_Simplify 对照 | compare_with_postgis()，输出顶点数 + Hausdorff |
| ④ W2 零改动 | 仅 import；`git diff scripts/auto_symbology.py` 为空 |
