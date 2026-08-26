"""SQL 安全校验器 —— 让模型生成的 SQL 只能"读"，不能"写"。

核心概念：LLM 生成的 SQL 是"不可信输入"
- 模型可能被诱导生成 DROP/UPDATE、读系统表、绕过限制（提示注入攻击）；
- 执行前必须做"词法 → 结构 → 语义 → 资源"多层校验，这是 Agent 的安全边界
  （safety guardrail），也是阶段1验收标准"SQL 注入防护完整"的落点。

防护策略（纵深防御，多层叠加，任一层被绕过还有下一层）：
1. 词法层：先做 SQL 分词（tokenize），把字符串字面量、引号标识符、注释
   正确识别出来 —— 这是关键！直接对原始文本正则匹配，会被 `'a; DROP--'`
   这种藏在字符串里的关键字骗过。
2. 结构层：只允许单条 SELECT（首 token 必须是 SELECT），
   从词法上排除 INSERT/UPDATE/DELETE/DDL/DCL/多语句/注释注入。
3. 语义层：表白名单 —— 只允许查询明确开放的几张表（ALLOWED_TABLES）。
4. 资源层：强制 LIMIT 上限，防止全表扫描拖垮数据库。
5. 执行层：所有值走 psycopg 参数化（%s 占位），杜绝拼接注入（见 connection.py）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import ALLOWED_TABLES

MAX_ROWS = 1000

# 禁止出现的语句级关键字（写操作 + DDL + DCL + 事务控制）
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "MERGE",
    "DROP", "ALTER", "CREATE", "TRUNCATE", "REINDEX", "CLUSTER",
    "GRANT", "REVOKE", "COPY", "CALL", "DO", "VACUUM", "ANALYZE",
    "SET", "RESET", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
    "LOCK", "REFRESH", "LISTEN", "NOTIFY", "UNLISTEN", "IMPORT",
}

# 危险函数（读服务器文件 / 连外部库 / 睡死连接等）
DANGEROUS_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec",
    "pg_sleep", "pg_terminate_backend", "pg_cancel_backend",
}


@dataclass
class _Token:
    kind: str     # word / string / ident / symbol
    value: str


def tokenize(sql: str) -> list[_Token]:
    """把 SQL 切成 token，正确识别字符串 / 引号标识符 / 注释。

    这一步是校验器可靠性的根基：先正确切词，后续所有检查都在 token 上做，
    而不是在原始字符串上做正则 —— 后者会被字符串里的关键字绕过。
    """
    tokens: list[_Token] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            # 行注释 -- ... 到行尾
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            # 块注释 /* ... */
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if ch == "'":
            # 字符串字面量（'' 是 SQL 里转义的单引号，要跳过）
            j = i + 1
            while j < n:
                if sql[j] == "'" and j + 1 < n and sql[j + 1] == "'":
                    j += 2
                    continue
                if sql[j] == "'":
                    break
                j += 1
            tokens.append(_Token("string", sql[i:j + 1]))
            i = j + 1
            continue
        if ch == '"':
            # 双引号标识符（如 "schools"）
            j = i + 1
            while j < n and sql[j] != '"':
                j += 1
            tokens.append(_Token("ident", sql[i:j + 1]))
            i = j + 1
            continue
        if ch.isalnum() or ch == "_":
            # 关键字 / 标识符 / 函数名 / 数字（数字也要作为整体，LIMIT 校验需要它）
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            tokens.append(_Token("word", sql[i:j]))
            i = j
            continue
        # 其它符号（括号 / 逗号 / 运算符等）
        tokens.append(_Token("symbol", ch))
        i += 1
    return tokens


class SQLValidationError(ValueError):
    """SQL 未通过安全校验时抛出；错误信息会回喂给模型让它修正。"""


def validate(sql: str) -> str:
    """校验 SQL，通过则返回（可能补过 LIMIT 的）安全 SQL；否则抛 SQLValidationError。"""
    tokens = tokenize(sql)
    words = [t.value.upper() for t in tokens if t.kind == "word"]

    # 1. 必须非空且以 SELECT 开头（排除 WITH/CTE 与一切非查询语句）
    if not words:
        raise SQLValidationError("SQL 为空")
    if words[0] != "SELECT":
        raise SQLValidationError(f"只允许 SELECT 查询，收到以 {words[0]} 开头的语句")

    # 1.5 禁止堆叠多条语句（分号必须在括号外，且后面不能再有内容）：
    #     否则第二条语句会被数据库一并执行，绕开本文件其余所有基于
    #     "只有一条语句"假设的检查（如第 5 步的 LIMIT 上限）。
    _reject_stacked_statements(tokens)

    # 2. 禁止语句级危险关键字（注意：字符串里的内容已被分词器隔离，不会误伤）
    for w in words:
        if w in FORBIDDEN_KEYWORDS:
            raise SQLValidationError(f"检测到禁止的关键字 {w}，仅允许只读 SELECT")

    # 3. 禁止危险函数（词后紧跟左括号才算函数调用）
    for idx, t in enumerate(tokens):
        if t.kind == "word" and idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if (
                nxt.kind == "symbol" and nxt.value == "("
                and t.value.lower() in DANGEROUS_FUNCTIONS
            ):
                raise SQLValidationError(f"检测到危险函数 {t.value}")

    # 4. 表白名单：提取 FROM / JOIN 后的表名逐一校验
    tables = _extract_tables(tokens)
    if not tables:
        raise SQLValidationError("未识别到 FROM 子句，无法确认查询目标表")
    for tb in tables:
        if tb not in ALLOWED_TABLES:
            raise SQLValidationError(f"表 {tb} 不在白名单 {list(ALLOWED_TABLES)} 内")

    # 5. LIMIT 上限
    return _enforce_limit(tokens, sql, MAX_ROWS)


def _reject_stacked_statements(tokens: list[_Token]) -> None:
    """拒绝顶层（括号外）的分号后面还跟着内容的情况。

    允许单条语句结尾带一个分号（`SELECT ...;`），因为分号后没有其它 token；
    括号内的分号（如函数参数、子查询里几乎不会出现，但保险起见也放过）不算数。
    """
    depth = 0
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.kind == "symbol":
            if t.value == "(":
                depth += 1
            elif t.value == ")":
                depth = max(0, depth - 1)
            elif t.value == ";" and depth == 0 and i + 1 < n:
                raise SQLValidationError("检测到多条语句（分号后仍有内容），只允许单条 SELECT")


def _extract_tables(tokens: list[_Token]) -> list[str]:
    """提取 FROM / JOIN 后引用的表名。

    - 支持 schema.table（取最后一段）、FROM a, b 多表、子查询（跳过括号，
      子查询内部的 FROM 会被主循环再次扫到）；
    - 别名不影响提取（读表名即返回，alias 只是后面的普通 word）。
    """
    tables: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind == "word" and t.value.upper() in ("FROM", "JOIN"):
            i += 1
            name, i = _read_table_name(tokens, i)
            if name:
                tables.append(name)
            # 逗号分隔多表：FROM a, b, c
            while i < n and tokens[i].kind == "symbol" and tokens[i].value == ",":
                i += 1
                name, i = _read_table_name(tokens, i)
                if name:
                    tables.append(name)
            continue
        i += 1
    return tables


def _read_table_name(tokens: list[_Token], i: int) -> tuple[str | None, int]:
    """从当前位置读一个表引用，返回 (表名, 下一个位置)。

    - 跳过 ONLY 关键字；
    - 遇到 '(' 表示子查询/括号连接：跳过整个括号，返回 None（内层 FROM 交给主循环）；
    - 支持 schema.table 点号限定，取最后一段作为表名。
    """
    n = len(tokens)
    while i < n and tokens[i].kind == "word" and tokens[i].value.upper() == "ONLY":
        i += 1
    if i >= n:
        return None, i
    if tokens[i].kind == "symbol" and tokens[i].value == "(":
        return None, _skip_parens(tokens, i)
    name: str | None = None
    while i < n and tokens[i].kind == "word":
        name = tokens[i].value.lower()
        i += 1
        if i < n and tokens[i].kind == "symbol" and tokens[i].value == ".":
            i += 1
            continue
        break
    return name, i


def _skip_parens(tokens: list[_Token], i: int) -> int:
    """跳过一对平衡括号，返回括号后的位置。"""
    depth = 0
    n = len(tokens)
    while i < n:
        if tokens[i].kind == "symbol":
            if tokens[i].value == "(":
                depth += 1
            elif tokens[i].value == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return i


def _enforce_limit(tokens: list[_Token], sql: str, max_rows: int) -> str:
    """确保 LIMIT 存在且不超过上限：有则校验，无则自动追加。"""
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.kind == "word" and t.value.upper() == "LIMIT":
            # LIMIT 后应紧跟数字（容忍一层括号）
            j = i + 1
            if j < n and tokens[j].kind == "symbol" and tokens[j].value == "(":
                j += 1
            if j < n and tokens[j].kind == "word":
                v = tokens[j].value
                if v.upper() == "ALL":   # LIMIT ALL = 不限制行数，必须拦
                    raise SQLValidationError("LIMIT ALL 不受支持，请指定具体行数")
                if v.isdigit() and int(v) > max_rows:
                    raise SQLValidationError(f"LIMIT 不能超过 {max_rows}")
            return sql
    # 没有 LIMIT：在末尾追加（rstrip(';') 去掉可能的结尾分号）
    return sql.rstrip().rstrip(";") + f" LIMIT {max_rows}"
