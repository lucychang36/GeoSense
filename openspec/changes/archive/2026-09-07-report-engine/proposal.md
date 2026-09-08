# Proposal：report-engine（第11月 W1 报告生成引擎 + W2/W3/W4 集成收官）

> 状态：**archived（2026-09-08，verify direct 27/27 + integration 29/29；数字闸门零泄漏实证）**

## 背景

学习计划阶段4（第10-12月）的终点是「自然语言 → 自动分析 → 自动制图 → 自动报告」。第10月已完成前三环（multi_agent 四 worker + auto_symbology/cartography/text_to_map），但管道止于制图——**报告是缺失的最后一环**。计划中第11月的 W2（前端集成）/W3（后端集成）大部分已被阶段3/4 提前建成，本变更把月度重心收敛为：报告引擎 + 它的两条接入链路 + 端到端测试。

## 目标 / 验收标准

1. **引擎独立可跑**：`scripts/report_engine.py` + `--selftest` 断言全过（LLM 不在场，合成 state 驱动全流程）
2. **CLI demo**：以 multi_agent 一次真实运行的 state 为输入 → `data/output/reports/` 落盘 md + html + 统计图 PNG；md 中所有数字与 state 逐字一致（数字闸门生效）
3. **双链路接入**：multi_agent 图插入 report worker（cartography→report→supervisor，既有 4 worker 回归不破坏）；chat 链路 `report_tool` 注册进 SPATIAL_TOOLS 10→11，SSE 新事件 `report`
4. **交付物可下载**：前端聊天流内出现报告卡片；`GET /api/reports/{name}` 白名单路由可下载 md/html/图（路径净化，缺字段/越界路径不得 200）
5. **端到端**：`scripts/e2e_test.py` 起服务 → 一条自然语言 → 断言 style/report 事件到达 + 产物落盘（学习计划「终极示例交互」最小实现）

## 方案要点（详见 design.md）

- **路线 C（用户已确认）**：md 为第一交付物，html 为预览格式（Jinja2 双模板同 context 渲染，零新依赖）；PDF 为可选增强后补，不进本变更
- **数字闸门（防幻觉核心）**：LLM 叙事只收定性输入，原始数字一律由 Jinja2 从 state 注入
- **地图进报告**：Mapbox Static Image API（token 已有）+ Matplotlib 统计图，不引 Folium

## 非目标

- PDF 导出（需要系统级依赖，后需要时单独一行 pandoc/weasyprint 补）
- Folium 交互地图、CesiumJS 3D（计划列示但非报告必需）
- 第12月性能优化 / 文档 / 开源准备
- 报告多主题泛化（先做变化检测报告一种，模板留扩展位）
