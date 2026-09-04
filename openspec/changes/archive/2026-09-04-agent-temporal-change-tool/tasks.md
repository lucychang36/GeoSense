# Tasks：Spatial Agent temporal_change 工具

> 提案：proposal.md ｜ 设计：design.md
> 范围：langchain_tools.py + graph.py 两文件改动 + 验证
> 状态：**全部完成，apply 验证通过（2026-09-04）**

## 任务清单

- [x] 1. `backend/agent/langchain_tools.py`：新增 `temporal_change_tool`（D1-D3）+ 注册进 `SPATIAL_TOOLS`
- [x] 2. `backend/agent/graph.py`：`SPATIAL_SYSTEM_PROMPT` 加遥感规则（原 3 条重排为 4 条，D4）
- [x] 3. 直调验证：postclass ≈0.119 / spectral ≈0.67 / 失败路径 `[工具错误]` 不抛异常（验收 1/2/4）
- [x] 4. 集成验证：uvicorn + SSE `/api/chat`，断言 `tool` 事件 `name=temporal_change_tool`（验收 3）
- [x] 5. 收尾：README 补一行说明 + 今日 memory + git 提交（代码 + README，无 data/.workbuddy 混入）

## 验证证据（2026-09-04 实测）

| 验收 | 结果 |
|------|------|
| 1. postclass 直调 | `change_ratio=0.1187`（n_change=39705 / n_valid=334583）；transition 水→水 109284 / 植→城 23802 / 城→植 10620，与第8月 W2 线下数据一致 ✅ |
| 2. 失败路径 | `[工具错误] COG 不存在…可用带日期的 COG 文件：szbay_real_20230708.tif、szbay_real_20250727.tif`（不抛异常）✅ |
| 3. 集成 SSE | `tool` 事件 `{"name":"temporal_change_tool","args":{cog_a: szbay_real_20230708.tif, cog_b: szbay_real_20250727.tif, method: postclass}}`；result 摘要 0.1187；LLM answer 正确解读转换矩阵 ✅ |
| 4. spectral 直调 | `change_ratio=0.6764`（与 W1/W2 baseline ~67% 一致）✅ |

## 完成后

- [x] proposal.md 状态更新为 applied
- [x] 向用户汇报：验收 4 条证据 + 关键数值 + 本轮红旗

## verify 阶段（2026-09-04，用户触发 /opsx:verify）

- 新增可重跑验证脚本 **`verify.py`**（本目录）：直调 3 组断言 + `--integration` 走 SSE 让 LLM 自主触发。用法：
  - `.venv/bin/python openspec/changes/2026-09-04-agent-temporal-change-tool/verify.py`（直调，无需服务，~6s）
  - `... verify.py --integration`（先起 uvicorn backend.api.main:app --port 8000，再跑，~10-40s）
- **verify 抓到的真实缺陷（已修复）**：main.py 的 SSE `result` 事件对工具摘要做 `[:200]` 截断，而工具返回 JSON 原先把 `change_ratio` 排在字段尾部 → 前端事件流拿不到核心数字。修复：`langchain_tools.py` 返回字段**关键数值前置**（change_ratio/n_change/n_valid 在前），200 截断天然保住核心数字；LLM 拿完整 ToolMessage 不受影响。
- 最终结果：**7/7 PASS**（直调 4 断言 + 集成 3 断言），exit 0。
