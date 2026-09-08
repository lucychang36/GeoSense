# Design：report-engine

## D1 数字闸门（本变更的核心设计，W4-D1「语义-语法分层」同款思想）

**LLM 在报告链路里拿不到任何原始数字。**

```
state.analysis_result {change_ratio: 0.1187, n_change: 8421, ...}
        │
        ▼  collect() 做定性化投影（代码，非 LLM）
qualitative = {direction: "增加", magnitude: "显著"(档位映射), region: "深圳湾", period: "2023→2025"}
        │
        ▼  llm_narrative(qualitative) → DeepSeek temp=0
"监测期内深圳湾水域面积呈显著增加趋势，主要分布在……"（解读叙事，无数值）
        │
        ▼  Jinja2 模板：{{ "%.2f"|format(r.change_ratio*100) }}% 等直接从 state 注入
最终报告 = 代码注入的数字 + LLM 的定性解读
```

- 档位映射规则在代码里（如 change_ratio∈[0,0.02)轻微 / [0.02,0.08)中等 / ≥0.08显著），LLM 输入只有档位词
- 叙事失败（LLM 超时/异常）→ 跳过该段，报告头部标记 `> ⚠ 自动叙事生成失败，以下仅含数据部分`——**降级可见，不静默**
- 红线：不许把 analysis_result 原始 dict 塞进 prompt「让 LLM 自己看着写」

## D2 双模板同 context（零新依赖）

- `templates/report.md.j2` + `templates/report.html.j2`（放 `scripts/report_templates/`），同一份 context 各渲染一次
- md 是第一交付物（git 友好、LLM 生态通用）；html 内嵌相对路径图表，浏览器直接打开即预览
- **不引 python-markdown 库**——md→html 转换不做，两个模板各自直出（结构对齐由 context 保证，代价是模板写两份，收益是零依赖且排版自由）
- 模板章节：标题/概述 → 核心指标表 → 统计图 → 叙事解读（可缺席）→ 方法说明（数据源、模型、阈值）→ 附录：交付物清单

## D3 图表与底图

- **统计图**：Matplotlib PNG（变化面积柱状/趋势线各一张）；中文字体复用 `unet_segmentation.py:255-266` 的 PingFang→Noto 回退模式（老坑已趟）
- **专题底图**：Mapbox Static Image API `https://api.mapbox.com/styles/v1/mapbox/dark-v11/static/{bbox}/{w}x{h}@2x?access_token=…`（token 从 .env，同前端视觉）；失败→跳过该图不致命
- **变化掩膜图**：直接引用 cartography worker 已产出的 `map_path` PNG（不重画）
- selftest 注入假 URL/合成数组，**不打网络**

## D4 图结构：report worker 与两条消费链路

- **multi_agent 图**：`cartography → report → supervisor` 线性插入（`backend/agent/multi_agent/graph.py`）。state 新增 `report_path / report_title / narrative_skipped`；`analysis_error` 存在时 report worker **短路**（step_log 记原因，不生成空报告）——回收第9月 `safe_cog_path`「缺字段不得溜进队列」的边界教训
- **chat 链路**：`report_tool(query)` 薄封装 = 调 multi_agent 图跑完整管道 → 返回 `{report_path, html_path, title, change_ratio}`（JSON，数字前置——W2 截断教训；绝不回文件内容）
- `backend/agent/langchain_tools.py` SPATIAL_TOOLS 10→11；graph.py 系统提示加报告路由规则（5→6 条）

## D5 下载路由（前端交付物卡片的前提）

- StaticFiles 只挂 `frontend/`，`data/output/` **不可下载** → 新增 `GET /api/reports/{name}`
- 路径净化同 `safe_cog_path` 模式：剥目录只留文件名 + 后缀白名单（.md/.html/.png）+ `is_file()` 显式检查 + FileResponse
- SSE 新事件 `report`：ToolMessage 分支识别 report_tool → `yield _sse("report", {title, md_url, html_url, chart_urls})`——URL 而非路径（前端可直接 `<a href>`）；不经过 result 的 [:200] 摘要路径
- 前端 handleEvent 加分支：聊天流插入交付物卡片（标题 + md/html/图下载链接），~30 行

## D6 e2e_test.py（W4，验收⑤）

- 自检 uvicorn 子进程起服（端口 8010 避让）→ POST /api/chat「对比深圳湾 2023 和 2025 的水域变化，并生成分析报告」→ 解析 SSE 流
- 断言：`tool` 事件含 report_tool → `report` 事件 JSON 合法 → 三个 URL `GET` 均 200 → md 文件落盘且含 `change_ratio` 对应的真实数字
- 退出码 0/1；属集成测试（依赖 LLM+模型服务），不进 selftest

## D7 验证矩阵

| 层 | 手段 | 依赖 |
|---|---|---|
| 引擎纯逻辑 | selftest（合成 state + 假 LLM stub） | 零网络零 LLM |
| 真实产物 | CLI demo（multi_agent 跑一遍 → 引擎） | DeepSeek + 本地 COG |
| 链路 | curl 实测 SSE report 事件 + 下载路由 200 | 起服务 |
| UI | 浏览器实测报告卡片 + 下载 | 浏览器 |
| 端到端 | e2e_test.py | 全部 |

## 红旗预登记

① LLM 定性解读仍可能错误归因（如把季节波动说成趋势）——闸门只保数字不保观点，temp=0 缓解；② html/md 双模板内容漂移（改一处忘另一处）——selftest 加「两模板渲染章节标题集合一致」断言；③ Mapbox Static API 配额与网络依赖（demo 一次一图，selftest 不触网）；④ report_tool 内嵌完整管道，单次调用 30-60s（planner+analysis+LLM 叙事三重 LLM/模型延迟），chat 场景需用户有心理预期；⑤ supervisor 汇总文案需感知 report 存在与否（narrative_skipped / analysis_error 分支文案）。
