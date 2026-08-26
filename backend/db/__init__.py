"""数据层：PostGIS 连接、表结构定义、SQL 安全校验。

分层（各司其职）：
- connection    —— 连接 + 查询执行（读 DATABASE_URL，参数化）
- schema        —— 表结构"单一事实来源"（DDL / 表白名单 / LLM 用的 schema 提示）
- sql_validator —— 只读安全校验（生成 SQL 执行前的最后一道闸）
"""
