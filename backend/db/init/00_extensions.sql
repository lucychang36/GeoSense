-- 容器首次启动时由 docker-entrypoint-initdb.d 自动执行（仅首次，数据卷为空时）。
-- 只启用 PostGIS 扩展（表结构由 scripts/seed_postgis.py 创建，更灵活、可反复重建）。
-- 注：pgvector 扩展待阶段2 RAG 迁移时启用（需先改用 docker/Dockerfile.postgis 镜像安装）。
CREATE EXTENSION IF NOT EXISTS postgis;
