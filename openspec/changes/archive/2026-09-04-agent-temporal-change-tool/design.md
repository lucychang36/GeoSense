# Design：Spatial Agent temporal_change 工具

> 上游提案：proposal.md（2026-09-04，路线 A + change 首工具，用户已确认）
> 状态：design 完成，进入 task/apply

## 现状约束（代码核查结论）

1. `backend/model_service/change.py::change_cog(cog_a, cog_b, method="postclass", weight="unet_finetuned.pt", threshold=0.08)`：
   - 已返回完整数值字段：`change_ratio / n_change / n_valid / transition(3×3) / class_names / cog_a / cog_b / method`
   - **同时也返回 base64 PNG**（`change_mask_png` / `rgb_png`）→ 工具层必须丢弃（文本 LLM 看不懂 + 撑爆工具结果上下文）
   - 异常契约（已核实 loader.py）：`safe_cog_path` 抛 `FileNotFoundError`（不存在 / 是目录）/ `ValueError`（空名 / `../` 越权）；尺寸不一致抛 `ValueError`
2. 注册点：`SPATIAL_TOOLS`（langchain_tools.py L104-113）→ `build_spatial_agent()`（graph.py L54-56）直接消费该列表，append 即注册生效。
3. Prompt 注入点：`SPATIAL_SYSTEM_PROMPT`（graph.py L28-37）现为 3 条规则。
4. Agent 运行在 uvicorn 进程内（`/api/chat` 链路）→ 工具函数内 `import ..model_service` **与 `/api/model` 共享同一 MPS 单例**，零网络、不重复 load 权重、不打 HTTP 环路。

## 接口设计

```python
@tool
def temporal_change_tool(cog_a: str, cog_b: str, method: str = "postclass") -> str:
    """对 data/cogs/ 下两景同区域遥感影像（文件名）做变化检测，返回变化占比与类别转换统计。
    当用户问"两期/两个年份影像对比变化"时使用。
    例：cog_a=szbay_real_20230708.tif, cog_b=szbay_real_20250727.tif；
    method="postclass"（U-Net 分类后比较，默认）| "spectral"（光谱差分）。"""
```

- 参数仅 3 个（LLM 调用可靠性优先）；不暴露 `weight/threshold`（保持默认即最优）
- 返回：`json.dumps({cog_a, cog_b, method, change_ratio, n_change, n_valid, class_names, transition})`（`transition` 仅 postclass 有值，spectral 为 null，语义由 `class_names` 顺序说明）

## 关键决策

| # | 决策 | 理由 |
|---|------|------|
| D1 | **函数内 lazy import** `..model_service.change` | `build_spatial_agent()` 构建期不触发 torch/unet 导入，Agent 秒建；首次调用才有模型预热（与 W2 单例 lazy 同哲学） |
| D2 | **摘要化返回**：数值 + 3×3 转换矩阵；`change_cog` 返回 dict 里的 base64 PNG 字段直接不取 | 文本模型可消费；避免大 base64 撑爆 ReAct 工具结果 |
| D3 | **错误契约**：不抛异常，返回 `"[工具错误] {e}。可用带日期的 COG 文件：{hint}"`（hint = glob `data/cogs/*.tif` 中含 8 位日期者） | 抛异常会炸 ReAct 循环；带候选列表兜底 LLM 编造文件名（沿用 multi_agent data_node 思路） |
| D4 | **Prompt 规则 3**（graph.py）：遥感影像时相对比 → `temporal_change_tool`，参数是文件名并给示例 | 触发率靠 docstring + 显式规则，不靠模型悟性 |
| D5 | 捕获 `FileNotFoundError + ValueError`，不捕获其它 | 只兜用户输入类错误；内部 bug 让它炸出来可观测 |

## 边界情况

- LLM 传带目录路径（`data/cogs/foo.tif`）→ `safe_cog_path` 剥目录只取文件名 → 合法
- LLM 传不存在的名字 → `FileNotFoundError` → `[工具错误]` + 候选列表
- 两景尺寸不一致（不同 tile）→ `ValueError`（尺寸不一致提示）→ `[工具错误]` 返回给 LLM 解释
- `method="spectral"` → `transition=None`，摘要含 null；`change_ratio` ≈0.67 量级

## 验收映射（对照 proposal.md 验收标准）

| 验收 | 验证方式 | 预期 |
|------|---------|------|
| 1. postclass 直调 | `.venv/bin/python` 直调工具 | `change_ratio ≈ 0.119` |
| 2. 失败路径 | 直调不存在 cog | 返回 `[工具错误]` 文本含候选，**不抛异常** |
| 3. 集成 SSE | uvicorn 起服务，POST `/api/chat` | 流中 `tool` 事件 `name=temporal_change_tool`，result 摘要正确 |
| 4. spectral 分支 | 直调 method="spectral" | `change_ratio ≈ 0.67` 量级（与 baseline 一致） |
