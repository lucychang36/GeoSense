# tasks.md — 2026-09-23-map-result-linkage

## T1 overlay 管线（thematic_change.py 扩展）

- [x] `min_size_filter(mask, min_px)`：scipy.ndimage.label + bincount（spike 已验证路径）
- [x] `THEMES[key]["min_patch_px"]`：builtup=5 / green=5 / water=1（注册表纯数据）
- [x] `build_overlay(cog_a, cog_b, theme)`：过滤 → features.shapes → simplify(5e-5) → 属性注入（kind/area_m2/theme/period，面积=px×px_m2）
- [x] index_change() 接入双口径：filtered mask 为主数字来源，metrics 增 raw_* / patch_keep_ratio / min_patch_px 字段
- [x] selftest 扩展：合成 GT 断言图斑数量 / 面积属性 / 阈值过滤语义（≥min_patch_px 才保留）/ simplify 面积保持 ≥98%；旧断言按双口径更新
- [x] 回归：thematic selftest 全绿

## T2 报告双口径（report_engine + 模板）

- [x] collect() metrics 增滤波口径字段；qualitative 增碎斑占比档位（>80% 高 / >50% 中 / 其余低）
- [x] build_narrative_prompt：碎斑占比档位词进 LLM 输入（仍纯定性，闸门不破）
- [x] md/html 模板：指标表加「原始总量（含 <Npx 碎斑）」行；方法说明加过滤声明
- [x] selftest：专题用例断言双口径字段 + 原始口径不泄漏进 prompt；全量回归

## T3 服务化接线（后端）

- [x] cartography_node：index_change 分支内跑 build_overlay → state.overlay_path（+图斑数进 step_log）
- [x] overlays 落盘 data/output/overlays/；multi_agent state 加 overlay_path/overlay_meta
- [x] report_tool 投影加 overlay_url/bbox/legend/render_hint/basemap（region→tile path 反查）
- [x] main.py：GET /api/overlays/{name}（剥目录+后缀白名单+is_file）；ToolMessage 分支发 overlay 事件
- [x] supervisor 专题汇总加「地图叠加」行

## T4 前端（frontend/index.html）

- [x] handleEvent overlay 分支：fetch geojson → removeSource/addSource 重建（W4 ids 模式）→ fill（match 上色）+ line + click popup（方向/面积）+ fitBounds
- [x] 图例 DOM（左下角浮层，legend 数组渲染）；status 事件清层时同步移除
- [x] basemap 动态换源：scene 载荷 → cog source 重建 + fitBounds；无 basemap 保持现状

## T5 端到端 + 收尾

- [x] e2e 郑州场景加断言：overlay 事件在场 / geojson 下载 200 / **图斑面积总和 == 报告 gain/loss km²（±0.01）** / basemap path 含 zhengzhou；深圳湾回归 7/7
- [x] 全 selftest 回归（thematic / report_engine / text_to_map / symbology / cartography）
- [x] 浏览器人工验证：郑州问句 → 底图切换 + 叠加 + popup + 图例
- [x] README：第12月 W2/W3 条目 + 关键数据（碎斑发现、双口径、协议设计）
- [x] tasks.md 证据回填 + proposal 状态流转 + memory + git 提交

## 完成后（证据回填）

- [x] spike 实测数字沉淀：23,608 区块 / 0.37s 多边形化 / 93% 碎斑面积 / 5px→38.5% 保持 / 1.2 MB→gzip ~300 KB
- [x] T1：thematic selftest 29 断言全绿（双口径 GT：主口径 2400 大块保留 + 4px 碎斑滤除 + overlay 面积对账 ±0.1% + water min_patch_px=1）
- [x] T2：report_engine selftest 22 断言全绿（noise 档位 + 原始口径行渲染 + raw 数字不进 prompt）
  - 排查插曲：selftest themed_state 字段 patch 静默失配（源文件同行为 "valid_pct..., gain_px..."，old 凭记忆拆行）→ heredoc 单独 patch + 独立 assert 解决；教训：多 replace 共用一个 assert 会掩盖个别失配
- [x] T3：agents/state/langchain_tools/main 四文件接线（独立 assert 逐文件验证）
- [x] T4：前端 overlay 渲染器 + 图例 DOM + status 清层 + basemap 重建
- [x] T5：e2e 郑州 17/17（overlay 面积总和==报告 3.91/4.57 km²、1903 图斑、basemap zhengzhou、双负例）+ 深圳湾回归 7/7（水域走 index_change(water) min_patch_px=1 现场验证）
- [x] 环境坑（本变更最大排查）：**残留旧服务占 8010 → e2e Popen 的 uvicorn 静默 bind 失败（stderr=DEVNULL），断言全打到旧服务**——三次 10/11 假 FAIL、lsof 实锤 PID 39447（用户 13:18 实测残留）、kill 后 17/17。教训：e2e 起服前应探测端口占用并显式报错
