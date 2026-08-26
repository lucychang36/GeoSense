# GeoSense Prompt 模式手册（W2 交付物）

> 5 种 GIS 场景的提示词模板，覆盖 4 种核心提示词技术。
> 所有模板的生产版本在 `backend/agent/prompts.py`，本文档解释设计原理。

## 四种核心技术速查

| 技术 | 一句话定义 | 适用场景 | GeoSense 用例 |
|------|-----------|---------|--------------|
| Zero-shot | 只给指令不给例子 | 模型已掌握的通用知识 | P1 知识问答 |
| Few-shot | 给 2~5 个输入/输出示例 | 需要锁定输出格式 | P2 NL2SQL、P3 意图解析 |
| CoT 思维链 | 要求"先推理后回答" | 多步推理、规划、计算 | P4 任务规划 |
| 结构化输出 | 强制 JSON 等格式契约 | 输出要被程序解析 | P2/P3/P5 |

**System Prompt 三层结构**：角色（Role）→ 规则（Rules）→ 输出契约（Contract）。

---

## P1 GIS 知识问答（Zero-shot）

**场景**：用户问"PostGIS 里 ST_DWithin 和 ST_Distance 有什么区别？"

**设计要点**：
- 用"角色"把模型限定在 GIS 专家视角，减少泛化闲聊；
- 用"规则"划边界：无关问题拒绝、坐标必须带 SRID、禁止编造函数名（GIS 函数名是幻觉重灾区）；
- 不给例子——知识问答考验的是模型已有知识，给例子反而限制发挥。

```text
你是 GeoSense，一名资深 GIS 专家助手……（完整版见 prompts.py: GIS_QA_SYSTEM）
```

---

## P2 自然语言 → PostGIS SQL（Few-shot + 结构化输出）

**场景**：用户说"查询南山区3公里内的所有学校"→ 生成可执行 SQL。

**设计要点（这是阶段1最关键的模板）**：
1. **Schema 注入**：模型看不见你的数据库，必须把表结构写进 Prompt（`{schema}` 占位符，运行时填充）；
2. **硬性规则防御**：只允许 SELECT——这是 SQL 注入防护的第一道（第二道是 W3 的 sql_validator）；
3. **::geography 陷阱**：EPSG:4326 下直接算距离单位是"度"不是"米"，必须在规则里强制转换，这是 GIS NL2SQL 最常见的错误；
4. **Few-shot 锁格式**：一个完整示例让模型稳定输出 `{"sql", "explanation"}` JSON；
5. **失败兜底**：`sql: null` 的显式退路，避免模型硬编一条错 SQL。

```text
用户：查询南山区3公里内的所有学校
输出：{"sql": "SELECT s.name, s.type FROM schools s, admin_boundary a
       WHERE a.name = '南山区' AND ST_DWithin(s.geom::geography, a.geom::geography, 3000);",
       "explanation": "用 ST_DWithin 做3公里缓冲区过滤……"}
```

---

## P3 空间查询意图解析（Few-shot + 结构化输出）

**场景**：Agent 拿到"帮我找福田区附近2公里的医院"，先拆成参数再调工具。

**设计要点**：
- 这是 **Function Calling 的前置形态**：意图解析 = 把非结构化输入变成工具参数；
- 示例里必须包含**负例**（"今天天气怎么样" → `unknown`），否则模型会对无关输入硬凑参数；
- `raw_query` 字段保留原句，便于日志追溯和后续纠错。

---

## P4 分析任务规划（Chain-of-Thought）

**场景**："分析深圳湾过去3年水质变化" → 拆成 数据检索→指数计算→变化分析→制图→报告。

**设计要点**：
1. **显式 CoT**：用 `<thinking>` 标签要求模型先分析"数据需求、依赖关系、工具选型"，再给计划——一步直接出计划容易漏步骤；
2. **依赖声明**：`depends_on` 字段让计划从"列表"升级为"DAG"，这是第10个月 LangGraph 多 Agent 编排的雏形；
3. **工具白名单**：限定 5 类工具，防止规划出系统没有的步骤。

---

## P5 自然语言 → 地图样式（结构化输出）

**场景**："用蓝色渐变显示人口密度" → Mapbox paint JSON（Text-to-Map 雏形）。

**设计要点**：
- 把**制图学知识**（ColorBrewer 配色原则：数值型用 sequential、分类型用 qualitative）写进规则——模型知道这些知识，但不约束就会乱用；
- 输出必须是合法的 Mapbox 表达式（`["interpolate", ...]`），前端可直接 `setPaintProperty`；
- `reason` 字段让模型解释配色依据，既是可解释性，也倒逼它遵守制图规则。

---

## 调试心法（经验沉淀）

1. **改 Prompt 先加规则，再加示例**：规则解决"类"问题，示例解决"个"问题；
2. **幻觉靠边界约束，不靠恳求**："禁止编造"不如"不知道就输出 null"；
3. **结构化输出失败时**：检查示例里 JSON 大括号是否转义（f-string 中 `{{` `}}`）；
4. **temperature 配套**：P2/P3/P5 这类结构化任务调用时设 `temperature=0`，P1 知识问答可用 0.3~0.7；
5. **每次改动留档**：Prompt 是代码的一部分，进 git、写 commit message。

## 验证方式

```bash
.venv/bin/python scripts/prompt_test.py        # 逐个模板实测（无 Key 时展示组装后的 messages）
```
