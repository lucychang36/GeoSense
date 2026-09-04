# 变更提案：为 Spatial Agent 增加 temporal_change 工具（最小模型能力接入）

> 状态：**applied（2026-09-04，验证证据见 tasks.md）**
> 日期：2026-09-04
> 决策：路线 A（Spatial Agent @tool）+ 首个工具 = 变化检测 change

## 为什么（Why）

1. 第9月 W2/W3/W4 建的模型服务（`/api/model/*` + `model_service/` 函数层）经核查**无任何代码级消费方**——唯一消费是各周 curl 集成验证（见 2026-09-04 memory）。
2. Spatial Agent（第3月 W2，`/api/chat` 链路，前端在用）的 9 个工具**全是矢量工具**（测距/缓冲/POI/SQL/制图），没有栅格/遥感能力——用户问"深圳湾这两年变化多少"Agent 答不了。
3. "对比两期影像变化"是 GIS 自然语言最高频问题之一，数据现成（49QGE 双时相 COG），是让模型能力迎来第一个真实业务消费方的最低成本切口。

## 变更内容（What changes）

### 需求

1. **新增 `temporal_change_tool`**（`backend/agent/langchain_tools.py`，与现有 9 个 `@tool` 同构）：
   - 签名：`temporal_change_tool(cog_a: str, cog_b: str, method: str = "postclass") -> str`
   - docstring 第一句写明触发条件（当用户问"两期/两个年份对比变化"时使用）+ 参数为 `data/cogs/` 下文件名
   - **函数内 import** `..model_service.change`（lazy，不拖慢 agent 构建；与 `/api/model` 共享同进程 MPS 单例，零网络，**不打 HTTP 环路**）
2. **返回值 = 结构化文本摘要**（LLM 可消费）：`change_ratio / n_change / n_valid / method / transition 摘要`。**绝不回 base64 PNG**（文本模型看不懂且撑爆上下文）；异常捕获返回 `[工具错误] ...` 文本 + 可用 COG 列表（不抛异常，防炸 ReAct 循环）。
3. **注册**：加入 `SPATIAL_TOOLS` 列表（`langchain_tools.py`）。
4. **Prompt 规则**：`SPATIAL_SYSTEM_PROMPT`（`backend/agent/graph.py`）补一条——遥感影像时相对比用 `temporal_change_tool`，参数是文件名。

### 非目标（Non-goals，本轮不做）

- ❌ 不把 base64 PNG/图 URL 接进对话回复（前端展示图 = 第11月 W2 前端集成范畴）
- ❌ 不加 `segment_tool` / `detect_tool`（每个工具都有 LLM 调用可靠性成本，先只驯熟 1 个）
- ❌ 不动 multi_agent（路线 B 延后）
- ❌ 不改 `model_service` / `/api/model` 现有实现
- ❌ 不加认证/限流（沿用教学项目现状）

## 涉及（Affected）

| 文件 | 改动 |
|------|------|
| `backend/agent/langchain_tools.py` | +1 `@tool` 函数 +~55 行；注册进 `SPATIAL_TOOLS` |
| `backend/agent/graph.py` | `SPATIAL_SYSTEM_PROMPT` +1 条规则 |
| （只读复用）`backend/model_service/change.py` `change_cog` | 零改动 |

## 验收标准

1. **单测/直调**：`temporal_change_tool("szbay_real_20230708.tif", "szbay_real_20250727.tif")` → 返回 JSON 含 `change_ratio ≈ 0.119`（与第8月线下 11.9% / 服务端 11.87% 吻合）
2. **失败路径**：传不存在 cog → 返回 `[工具错误]` 文本（含可用列表），不抛异常
3. **集成（必须起服务 curl）**：uvicorn 起 `backend.api.main:app`，SSE POST `/api/chat` 问"对比 2023 和 2025 深圳湾水域变化" → 流中出现 `tool` 事件 `name=temporal_change_tool` 且 `result` 摘要正确
4. `method="spectral"` 分支可跑通（返回 67% 量级，与 baseline 一致）

## 风险与红旗

- LLM 可能编造/记错文件名 → 工具内白名单 + 错误信息带候选（沿用 multi_agent data_node 兜底思路）
- 一次只接 1 个工具，避免 ReAct 决策面变乱
- 结果可信度受伪标签 mIoU 0.927 上限约束（服务端已声明，工具层不重复处理）
