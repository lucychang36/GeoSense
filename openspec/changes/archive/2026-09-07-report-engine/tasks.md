# Tasks：report-engine

## T1 引擎核心（scripts/report_engine.py + templates/）
- [ ] `collect(state)`：analysis_result/tool 结果/map_path → 定性投影（档位映射代码化）+ 数字上下文（模板直用）
- [ ] `render_charts()`：Matplotlib 统计图 PNG ×2（中文字体 PingFang→Noto 回退）+ Mapbox Static 底图（失败跳过）
- [ ] `llm_narrative(qualitative)`：DeepSeek temp=0，只收定性输入；失败→返回 None 且标记（不静默）
- [ ] `render_report(context)`：Jinja2 双模板（report.md.j2 / report.html.j2）同 context 渲染 → `data/output/reports/` 落盘
- [ ] `--selftest`：合成 state + LLM stub 全流程；断言含「md 中数字与 state 一致」「叙事缺席时报告仍完整」「双模板章节标题集合一致」「数字闸门（narrative prompt 不含原始数值）」

## T2 CLI demo
- [ ] 跑 multi_agent 真实管道 → state → 引擎 → 落盘 md+html+PNG，打印交付物清单
- [ ] 断言：md 含 change_ratio 真实数字；文件非空；清单与落盘一致

## T3 双链路接入
- [ ] multi_agent：state.py 加 report_path/report_title/narrative_skipped；agents.py 加 report_worker；graph.py 插入 cartography→report→supervisor；analysis_error 短路；multi_agent_demo 回归 4 worker 既有断言不破坏
- [ ] report_tool（langchain_tools.py）：薄封装调图，JSON 返回数字前置；SPATIAL_TOOLS 10→11；graph.py 系统提示 5→6 条（报告路由规则）
- [ ] main.py：`GET /api/reports/{name}`（safe_cog_path 同款净化 + 后缀白名单 + is_file + FileResponse）；SSE `report` 事件（URL 型载荷，不走 [:200] 路径）
- [ ] frontend：handleEvent report 分支 + 交付物卡片（标题 + 下载链接，~30 行）

## T4 e2e_test.py
- [ ] 起服（8010）→ POST /api/chat 自然语言 → 断言 tool 含 report_tool / report 事件 JSON 合法 / 下载 URL 全 200 / md 落盘含真实数字
- [ ] multi_agent demo + W4 text_to_map 浏览器路径回归不破坏

## T5 收尾
- [ ] README：第11月 W1-W4 进度行、项目结构（report_engine/templates/e2e）、快速开始、里程碑「终极交互全链路打通」
- [ ] tasks.md 证据回填 + proposal 状态流转；memory 追加；git 提交（代码 + README + openspec）

## 完成后（证据回填）

## 完成后（证据回填）

- **selftest**：10/10 PASS（零网络零 LLM，stub 驱动）——数字闸门 prompt 泄漏集为空 / md+html 含真实数字 11.82% / 双模板章节标题集合一致 / LLM 宕机降级（narrative_skipped + ⚠ 头部提示 + 数据部分完整）
- **CLI demo**：真实 multi_agent 管道 change_ratio=0.6764（与第10月 baseline 一致）→ md+html+stats.png+basemap.png 落盘，md 含 67.64% 自检过；DeepSeek 叙事质量良好（归因潮位/季节/人类活动 + 建议，无数值编造）
- **e2e_test.py 7/7 PASS**：起服 8010 → chat 一句 NL → planner 自主路由 temporal_change_tool + report_tool（11s）→ report 事件（title + change_ratio=0.6764 + md_url）→ GET 200 → md 含 67.64%（数字闸门端到端）→ ../ 穿越负例拒（400/404）→ done 收尾
- **回归**：multi_agent demo 67.64% 不变（4→6 步含 report）；text_to_map selftest 14/14；report_engine selftest 复跑全过
- **踩坑/偏差**：① ① 首版时相图用绝对路径（html 相对链接断）→ 拷进报告目录 ② Edit 工具静默失败坑再次大规模复现（multi_agent 三件 + main.py 首轮全没落盘 / 部分延迟可见）→ 全程 heredoc patch + 事后 assert 验证 ③ supervisor 汇总文案补 report 行（design D8 红旗⑤提前处理）④ W4 上轮 README 的 W4 行 ⏭ 残留（Edit 陷阱遗留）本轮回溯修正
- **commit hash**：见 git log（本轮提交）

## verify 阶段（2026-09-08）

- 新增 `verify.py`（模式同前两次：direct 直调 + `--integration`；后者复用 scripts/e2e_test.py 自起 8010，不重复造轮子）
- **direct 27/27 PASS**（~10s，stub 驱动零真实 LLM）：
  - 1a 引擎 selftest 复跑 rc=0
  - 1b QUAL_BANDS 边界三连（0.01→轻微 / 0.05→中等 / 0.50→显著）
  - 1c 数字闸门：叙事 prompt 对 11.82 / 0.1182 / 226325 / 334583 零泄漏（时间/地名为设计允许）；stub 实收 prompt 同样零泄漏
  - 1d generate_report stub 全流程：md 含真实数字（模板注入）+ 叙事进入
  - 1e 宕机降级：narrative_skipped=True + ⚠ 提示 + 数字完整
  - 2 接线 8 条：工具 10→11 / prompt 报告规则 / report_worker 短路 / 图线性插入 / state 三字段 / SSE+路由 / 前端卡片
  - 3 路由净化直调 4 条：../ 穿越 400 / 白名单外后缀 400 / 不存在 404 / 真实文件 200（免起服直调 download_report，比 curl 更快且可进 CI）
  - 4 产物 4 条：67.64% baseline 数字 / 相对路径 / 数字来源声明
- **--integration 29/29 PASS**：e2e_test.py rc=0（18s）+ 7/7 汇总行
- verify 自身两处修正（实现无恙）：① `ok = ... in {...} and qual.get("coverage")` 返回字符串 '较高'（truthy）致 sum() 崩——W4 verify 同款 and 链隐性返回值坑第二次出现，bool() 包裹 ② verify 产出的测试报告主动 unlink 清理（不加重 selftest 污染问题；复盘发现的 P3 缺陷仍挂账待修）
- proposal 状态维持 applied，待用户确认后归档
