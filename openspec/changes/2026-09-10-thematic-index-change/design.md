# design：thematic-index-change（光谱指数专题框架）

## D1 THEMES 专题注册表（纯数据，核心设计）

`backend/model_service/themes.py` 新文件，字典结构，**零逻辑**：

```python
THEMES = {
    "builtup": {
        "label": "建筑用地",
        "index": {"formula": "ndbi", "b_pos": 5, "b_neg": 4},   # (SWIR-NIR)/(SWIR+NIR)，波段按 6 波段 COG 索引
        "rule": {"op": "gt", "threshold": 0.0, "assist": {"formula": "ndvi", "op": "lt_index"}},  # NDBI>0 且 NDVI<NDBI
        "color": "#5F5E5A",   # 城市灰（对齐 W2 SEMANTIC）
    },
    "green":  {...},   # 绿化用地：NDVI>threshold，排除 builtup/water（先注册不实测）
    "water":  {...},   # 水域：NDWI>threshold，单指数（先注册不实测）
}
```

- 设计原则回收：W2 `SEMANTIC` 语义色表（领域知识=纯数据）、第9月 `task_executors` 注入点（扩展点加条目，核心零改动）
- `assist` 条目实现联合判定去噪（builtup 需 NDVI<NDBI 排除农田/裸土；green 需排除 builtup/water）
- 阈值初始值为文献先验，**T3 真实影像标定后回填并在注释中留标定数字**

## D2 index_change 泛化引擎（唯一实现）

`backend/model_service/change.py` 新函数 `index_change(cog_a, cog_b, theme_key)`：

1. 读两期 6 波段 COG → 按 THEMES[theme] 算指数（归一化到 [-1,1]，除零保护）
2. 联合判定 → 两期专题掩膜 → 差分：新增（b 有 a 无）/ 消失（a 有 b 无）/ 保持
3. 像元面积 = 10m 网格实算（rasterio transform）→ 输出 `{method: "index_change", theme, theme_a_pct, theme_b_pct, gain_pct, loss_pct, gain_km2, loss_km2, net_km2, valid_pct}`——字段命名与 spectral_diff_change 输出同构，cartography/report 零适配成本
4. selftest（零网络零数据）：合成 6 波段小数组，人工埋一块"新增建筑"矩形（期 A 低 NDBI、期 B 高 NDBI 低 NDVI），断言 gain 像元数与 GT 一致

## D3 数据获取泛化（satellite_download.py）

- `BBOX` 常量 → `--region` 参数（内置 REGION_PRESETS = {"szbay": ..., "zhengzhou_hightech": (113.50, 34.73, 113.70, 34.88)}），日期对 `--start/--end`，默认值保持深圳湾（向后兼容）
- `REAL_BANDS` 扩展为 6 项：`{"blue":"B2","green":"B3","red":"B4","nir":"B8","swir16":"B11","swir22":"B12"}`（earth-search element84 L2A 资产名）；SWIR 20m 由 COGReader feature(width=W,height=H) 统一重采样到 10m 网格——**波段顺序成为仓库约定 v2：B2,B3,B4,B8,B11,B12**
- 合成模式同步生成 6 波段（SWIR 模拟：城市高/植被低），保证 selftest 与离线教学用同一格式；真彩色预览 bands=(3,2,1) 索引不变
- manifest 条目加 `bands` 数量字段；cog_reader 按实际波段数自适应
- 同 tile 双时相配对 + 覆盖度评分逻辑不变（现成）；郑州约在 MGRS 49P 系列 tile

## D4 阈值标定（先探查后定值，不拍脑袋）

郑州期 B 影像落库后：NDBI/NDVI/NDWI 各出直方图 + 分位数表（脚本内置 `--probe` 模式打印）→ 结合文献先验定阈值 → 回填 THEMES 注释（标定日期、影像 ID、分位数字）。红旗①的工程解。

## D5 报告专题参数化（report_engine）

- `collect(state)`：region 从 `state` 取（analysis_result.region 或 plan），缺省回落"深圳湾"；`theme_label` 从 analysis_result.theme 取，缺省"水域"——**向后兼容旧链路**（旧 state 无这些字段时输出与现在逐字节一致，selftest 断言）
- 标题模板：`{region}{theme_label}变化分析报告`；新增数字闸门字段：`gain_km2 / loss_km2 / net_km2`（theme 面积档位：<0.5 轻微 / <2 中等 / ≥2 显著，代码化映射同 D1 先例——LLM prompt 只见档位词）
- 模板（report.md.j2 / html.j2）只加变量不加章节逻辑，双模板一致性 selftest 同步扩展

## D6 multi_agent 集成（最小改动面）

- `agents.py` analysis worker：plan.task_type 新增 `"index_change"` 分支（读 plan.theme，调 `index_change`），**不动 spectral_diff 原路径**
- `agents.py` planner 提示词加 1 条拆解规则：「建筑用地/绿化/水域等专题变化 → task_type=index_change + theme 枚举」（LLM 只选枚举，同 W4 意图分层）
- `graph.py` data_node：按 query 中的区域关键词匹配文件名前缀（深圳湾→szbay、郑州→zhengzhou），解决"多区域共存时选错数据"
- **不加新 LangChain 工具**：temporal_change_tool 图入口已通，planner 在图内路由即可（防止 SPATIAL_TOOLS 无谓膨胀）

## D7 验证矩阵

1. **themes selftest**（零网络）：注册表结构合法性（波段索引在界、exclude 引用存在、阈值 ∈[-1,1]）
2. **index_change selftest**（零网络零数据）：合成 GT 断言（D2.4）
3. **report selftest**：region/theme 注入 + 无 theme 字段回落逐字节一致 + 面积档位闸门（prompt 泄漏集为空）
4. **真实数据实测**：郑州两期下载（同月、云量 <10%、覆盖度 ≥50%）→ D4 标定 → index_change → 人工检查专题掩膜叠加真彩色（新增区应落在建成区/工业园区）
5. **e2e 扩展**：一句 NL「对比郑州高新区 2023 和 2025 的建筑用地变化并生成报告」→ planner 路由 index_change → report 事件 → md 含 net_km2 数字 → 下载 200
6. **回归**：旧 4 波段深圳湾链路 e2e 7/7 + 全部既有 selftest 不破坏
7. verify.py：direct（1-3 断言）+ `--integration`（复用 e2e_test.py 模式）

## D8 回收的旧伏笔

- W2「数据性质 ≠ 制图语义」→ 本次同款：「光谱指数命中 ≠ 土地利用真值」（NDBI 高也可能是裸土，联合判定只是缓解）——报告措辞用"疑似建筑用地"
- W4 意图-渲染分层 → LLM 只选 theme 枚举不碰计算（输出空间约束的同款应用）
- 第11月数字闸门 → 面积字段同构接入（输入空间约束）
- 第9月 `safe_cog_path` 教训 → REGION_PRESETS 白名单思路一致（枚举优于自由字符串）
