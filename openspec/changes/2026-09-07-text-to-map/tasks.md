# Tasks：text_to_map

## T1 引擎核心（scripts/text_to_map.py）
- [x] CartographyIR dataclass + validate_ir 三道闸（parse 闸在 extract_intent 内）+ 默认 IR 回退（fallback 标记）
- [x] extract_intent：DeepSeek temp=0 + response_format json_object + 单次重试（校验错误回填 prompt）
- [x] raster_to_grid 辅助函数（COG → 40×40 网格多边形 + ndvi 属性；rasterio `read(out_shape, Resampling.average)` 一步重采样，投影 CRS 四角 pyproj 转回 WGS84）
- [x] assemble_style 三主题分支（poi→match circle + W3 symbol / ndvi→interpolate fill / cog→raster-opacity），完整 Style Spec v8

## T2 selftest + CLI demo
- [x] --selftest 13 断言：IR 校验拒绝（非法 theme / label ⊄ visible / 越界 opacity / 白名单外 cog / 未注册语义词 / 未注册 color_intent）/ 三主题 assemble 结构断言 / match 键=class 名 / 语义覆盖 G 通道 / interpolate 断点升序 / Greens / raster-opacity+tiles URL——LLM 不在场全跑
- [x] CLI demo：3 条 NL → data/output/text_to_map/demo_{poi,ndvi,cog}.json 落盘 + 主题互异断言 + rationale 打印

## T3 Agent 工具接入
- [x] langchain_tools.py：text_to_map_tool（summary 前置 + map_style 尾置 + 异常→[工具错误]+theme 提示）+ SPATIAL_TOOLS 9→10
- [x] graph.py：SPATIAL_SYSTEM_PROMPT 4 条 → 5 条（新增制图路由规则）

## T4 SSE + 前端
- [x] main.py _stream：ToolMessage → map_style → yield _sse("style", 全量)
- [x] frontend/index.html：handleEvent style 分支 + applyStyle（清旧→t2m- 前缀注入→fitBounds）+ 欢迎语示例问法 + data() 加 t2m ids
- [x] **起服务实测**：curl /api/chat 制图请求 → style 事件可达、JSON 可 parse、layers[].source 为 id 引用（spec v8）；**浏览器实测**（agent-browser + Chrome 152）：POI 图（公园 #4C9F38 绿 + 地铁橙 + 107 标注）与 NDVI 网格图（Greens 1545 格）先后上图、二次提问旧 t2m 图层自动清除、既有 COG/POI 聚合层不破坏

## T5 回归 + 收尾
- [x] multi_agent demo 重跑：67.64% / 4.72s / 5 步不变（既有工具零影响）
- [x] README：阶段4 标题收官 + W4 进度行 + W4 关键数据块 + 项目结构（补 W3 漏掉的 auto_cartography 行）+ 快速开始 6.10 + 技术选型 + 里程碑
- [x] tasks.md 勾选回填证据 + proposal 状态 applied
- [x] memory 追加 + git 提交（代码 + README + openspec，无 data/.workbuddy 混入）

## 完成后（证据回填）

- **selftest**：13/13 PASS（初版 12/13——match 表达式断言索引写错（键在偶数位 [2] 非奇数位 [3]），修正测试后全过）
- **CLI demo**（真实 DeepSeek temp=0）：3 条 NL → poi/ndvi/cog 三份 Style JSON 主题互异、**零回退**；LLM 三处决策全对：公园→`color_semantics {"park":"植被"}`→#4C9F38、NDVI→`color_intent="植被"`→Greens、半透明→`raster_opacity=0.5`；poi 标注避让 placed 107/ dropped 281；ndvi 1545 有效网格（值域 [-0.223, 0.889]）；文件大小 531KB / 1.25MB / 689B
- **SSE 实测**：curl /api/chat 发制图请求 → 事件序列 status/district(无)/tool×2/result×2/style/answer/done；style 事件全量合法（theme=poi、semantic_hits、label_stats、fit_bounds、layer.source=id 引用）；planner 先调 data_retrieval 再调 text_to_map_tool（路由正确）
- **浏览器实测**：POI 图与 NDVI 网格图先后真实渲染（截图为证）；旧 t2m 层清除、既有卫星影像开关/POI 聚合层不破坏
- **偏差记录**（对 design）：① 前端 id 前缀改由**后端预置**（design D7 原写前端加前缀——后端统一生成更干净，前端零加工）② demo 落盘路径 `data/output/text_to_map/`（proposal 原写 `data/output/`，加子目录更整洁）③ W3 的 auto_cartography.py 在 README 项目结构区漏登（本轮回溯补上）
- **踩坑**：① `_EXTRACT_PROMPT` 含 JSON schema 示例不能用 `.format()`——`{"error": ...}` 的 `"error"` 被当占位符抛 `KeyError '"error"'`，demo 三条全部静默回退后单测 `_llm_json` 定位 → 改 replace 拼接 ② WorkBuddy 会话内 bash `&` 背景进程随 shell 退出被杀 → uvicorn 用 run_in_background
- **commit hash**：见 git log（本轮提交）

## verify 阶段（2026-09-07）

- 新增 `verify.py`（模式同 temporal_change_tool：direct 直调 + `--integration` SSE）
- **direct 23/23 PASS**（~5s，零 LLM）：引擎 selftest 复跑 13/13 + validate_ir 四类拒绝（非法 theme / 白名单外 cog / 未注册语义词 / label⊆）+ 工具注册 9→10 + graph 制图规则 + main.py style 发射 + 前端 style 分支/applyStyle + demo 三 JSON spec v8 结构（source=id 引用 / layer_ids 覆盖 sources+layers / fit_bounds / 三主题互异 / semantic_hits / label_stats 107 placed）
- **--integration 27/27 PASS**（SSE 实测 10.3s）：tool 事件 LLM 自主触发 text_to_map_tool → style 事件全量 JSON 可解析、载荷 spec v8 合法 → answer/done 收尾；事件统计 {status:1, tool:2, result:2, style:1, answer:1, done:1}
- **verify 脚本自身两处修正**（实现无恙）：① layer_ids 断言初版写窄——按设计含 sources+layers 两类 id（前端清旧层两者都删），改为超集断言 ② `ok = ... and palette.get("name")` 返回字符串 "Greens"（truthy）致 sum() 崩——bool() 包裹；另再次踩中 Edit 工具报成功磁盘未更新坑，heredoc patch 落盘
- proposal 状态维持 applied，待用户确认后归档
