# tasks：thematic-index-change

## T1 数据获取泛化（design D3）
- [x] satellite_download.py：`--region/--start/--end` 参数化 + REGION_PRESETS（szbay/zhengzhou_hightech），默认值向后兼容
- [x] REAL_BANDS 扩 SWIR（B11/B12，6 波段约定 v2）；合成模式同步 6 波段；manifest 加 bands 字段；cog_reader 波段自适应
- [x] 实测：郑州高新区两期（同月、云量 <10%、覆盖度 ≥50%）下载入库；配不齐时如实记录数据限制

## T2 专题框架（design D1/D2/D4）
- [x] `backend/model_service/themes.py`：THEMES 注册表（builtup/green/water，green/water 只注册）
- [x] `index_change(cog_a, cog_b, theme)`：指数 → 联合判定 → 两期差分 → km² 面积（输出与 spectral_diff_change 同构）
- [x] selftest：THEMES 结构断言 + 合成 GT 断言（零网络）
- [x] D4 阈值标定：`--probe` 直方图/分位数探查 → 标定数字回填 THEMES 注释

## T3 报告与集成（design D5/D6）
- [x] report_engine collect()：region/theme 参数化 + 无字段回落逐字节兼容 + 面积档位闸门（gain/loss/net km2）；模板加变量；selftest 扩展
- [x] analysis worker 加 index_change 分支；planner 提示词加拆解规则；data_node 区域关键词选文件前缀
- [x] multi_agent 回归：深圳湾旧 query 输出与现版本一致

## T4 端到端验证（design D7）
- [x] e2e 扩展：「郑州高新区建筑用地变化」NL → index_change → report 事件 → md 含 net_km2 → 下载 200
- [x] 真实报告人工检查：新增区叠真彩色合理性（建成区/工业园区）
- [x] 回归：e2e 7/7（旧链路）+ 既有 selftest 全过

## T5 收尾
- [x] README：第12月 W2 实战检验条目、专题框架说明、仓库波段约定 v2
- [x] tasks.md 证据回填 + proposal 状态流转；memory；git 提交

## 完成后（证据回填）

- **T1 数据获取**：`--region/--start/--end/--tile auto` 落地（REGION_PRESETS szbay/zhengzhou_hightech，默认 szbay 向后兼容）；REAL_BANDS 六波段 v2；合成模式同步 SWIR 模拟；cog_reader `_pick_bands` 本就按 count 自适应（count>=3→RGB），无需改动；manifest entries 自带 bands 字段。郑州实测：**49SGU** 双景入库（2023-06-26 云 5.2% / 2025-06-27 云 4.4%，覆盖 100%，各 9.2MB）——红旗②解除
- **T2 专题框架**：`scripts/thematic_change.py`（THEMES 三专题 + FORMULAS 三指数 + classify 联合/排除 + index_change 同构输出 + --probe 标定）；selftest **21/21**（合成 GT 精确 2400 px；water 稳定 0/0；未知专题显式报错；面积档位）；阈值标定：NDBI p50=-0.007（中位数贴 0）→ `NDBI>0 且 NDVI<NDBI` 联合判定 → 两期建成区占比 11.4%/11.1%（半城乡混合实况），标定数字回填 THEMES 注释
- **T3 报告与集成**：collect() region/theme 参数化（旧 state 回落标题**逐字节一致**，selftest 断言）；面积档位 AREA_BANDS + prompt 只进档位词（泄漏集为空）；双模板加面积行 + 方法说明条件分支；agents.py：planner 枚举 + 规则回退关键词版、data_node 区域关键词路由（3 用例实测）、analysis/cartography/supervisor 各加 index_change 分支（spectral 原路径零改动）；**踩坑**：模板 Edit 静默失败复现（面积行丢失）→ heredoc patch + 回读 assert 是唯一可靠路径
- **T4 端到端**：e2e 参数化（默认深圳湾回归 / `--zhengzhou` 专题）；**郑州 10/10**（title=郑州高新区建筑用地变化分析报告、7.4% 数字闸门、面积行、疑似 建筑用地、../ 负例）+ **深圳湾 7/7 回归**；5 个 selftest 全绿（21+17+14+8+10）；目视检查：中部偏东红色发展走廊可信，蓝色消失沿道路线状=阴影/BRDF 伪变化（红旗②如实入 README）
- **踩坑/偏差**：① CLI 交付自检 `.2f` 与 metrics `round` 口径差（0.074→"7.4%" vs "7.40%"，历史值 11.82 恰好两位掩盖）→ e2e/CLI 同口径修复 ② INDEX IndexError：rasterio 波段从 1 起（selftest 抓住）③ README 结构区 satellite_download 重复行自查删除
- **commit hash**：见 git log（本轮提交）
