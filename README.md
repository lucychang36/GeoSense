# GeoSense — 空间智能分析 Agent

> 自然语言驱动的空间智能分析平台。你说"分析深圳湾过去3年的水质变化"，
> GeoSense 自动获取卫星数据、运行 AI 模型、生成变化地图和分析报告。

基于《GeoAI_LLM_Agent_学习计划_增强版.md》分 4 个阶段、12 个月构建，采用"边构建边讲解"的学习模式：每个周节点产出可独立运行的交付物 + 核心概念说明。

## 当前进度

**第1月：LLM 基础 ✅（全部完成）**

| 周 | 主题 | 交付物 | 状态 |
|----|------|--------|------|
| W1 | 项目骨架 + LLM API 对接 | `scripts/llm_basics.py` | ✅ |
| W2 | Prompt 工程实战 | `backend/agent/prompts.py` + `prompt_patterns.md` | ✅ |
| W3 | Embedding 与向量相似度 | `scripts/embedding_search.py` | ✅ |
| W4 | Function Calling + GIS 工具 | `scripts/gis_tools.py` | ✅ |

**第2月：RAG 系统 ✅（全部完成）**

| 周 | 主题 | 交付物 | 状态 |
|----|------|--------|------|
| W1 | 向量数据库搭建 | `backend/rag/vector_store.py` + `scripts/vector_store.py` | ✅ |
| W2 | 文档处理流水线 | `backend/rag/doc_processor.py` + `scripts/doc_processor.py` | ✅ |
| W3 | RAG 检索链路 + 混合检索 | `backend/rag/retriever.py` + `scripts/rag_chain.py` | ✅ |
| W4 | RAG 评估与优化 | `backend/rag/evaluation.py` + `scripts/rag_evaluation.py` | ✅ |

**第3月：Agent 开发 ✅（全部完成）**

| 周 | 主题 | 交付物 | 状态 |
|----|------|--------|------|
| W1 | LangGraph + 单工具 Agent | `backend/agent/graph.py` + `scripts/basic_agent.py` | ✅ |
| W2 | 多工具 Agent + 空间分析工具集 | `backend/agent/spatial_tools.py` + `scripts/gis_agent.py` | ✅ |
| W3 | 多步推理工作流 + 错误处理 | `backend/agent/workflow.py` + `scripts/workflow_agent.py` | ✅ |
| W4 | Agent 评估 + 前端界面 | `backend/api/main.py` + `frontend/index.html` | ✅ |

**阶段1 补课：PostGIS SQL 生成/执行 ✅**

原计划阶段1 的「自然语言→PostGIS SQL 生成 + 安全执行」缺口已补齐：

| 模块 | 交付物 | 说明 |
|------|--------|------|
| 数据层 | `backend/db/` | 连接（psycopg3）+ schema 单一事实来源 + 只读安全校验 |
| SQL 工具 | `backend/agent/sql_tools.py` | `spatial_sql`：自然语言→SQL→校验→执行→GeoJSON |
| 数据源切换 | `backend/agent/spatial_tools.py` | `data_retrieval`/`spatial_query` 由内存 JSON 切到 PostGIS（签名不变） |
| 数据库 | `docker-compose.yml` + `docker/Dockerfile.postgis` | PostgreSQL 16 + PostGIS 3.4 + pgvector（预留） |
| 演示 | `scripts/seed_postgis.py` + `scripts/fetch_osm.py` + `scripts/postgis_sql_agent.py` | 样例数据 / OSM 真实数据导入 + 端到端演示与安全检查 |

> **数据说明**：`fetch_osm.py` 从 OpenStreetMap（Overpass API）拉取深圳 10 个区县的真实行政边界与
> 学校/医院/地铁站等 POI（约 3000+ 条），原始响应缓存于 `data/osm/`（`--refresh` 强制重下）。
> 边界为 OSM 原始绘制（沿海区含近海海域）；同名异地区（如海南龙华区）已在解析时按深圳范围过滤；
> 同一 relation 残留的多余外环（如龙岗区历史包含大鹏区域）按 relation 自带 label 节点筛除。

**阶段2 第4月：云原生遥感数据（COG）🚧（进行中）**

| 周 | 主题 | 交付物 | 状态 |
|----|------|--------|------|
| W1 | GDAL/Rasterio 基础 + COG 生成 | `scripts/make_demo_raster.py` + `scripts/cog_generator.py` | ✅ |
| W2 | 在线 COG 读取 + 瓦片服务 | `backend/data/cog_reader.py` + `scripts/tile_server.py` + `scripts/cog_range_demo.py` | ✅ |
| W3 | 遥感数据下载 + COG 仓库 | `scripts/satellite_download.py`（合成/真实双模式）+ `data/cogs/` + `manifest.json` | ✅ |
| W4 | 性能优化 + 可视化验证 | `scripts/cog_benchmark.py` + 前端 COG 栅格图层（含开关） | ✅ |

**阶段2 第5月：STAC + GeoParquet ✅（全部完成）**

| 周 | 主题 | 交付物 | 状态 |
|----|------|--------|------|
| W1 | STAC 规范 + pystac 实践 | `scripts/stac_catalog.py` + `data/stac/`（Catalog→Collection→Item 三层） | ✅ |
| W2 | STAC API 服务 | `stac_api/main:app` + `pystac-client`（/collections、/search 时间/空间过滤） | ✅ |
| W3 | GeoParquet + DuckDB Spatial | `scripts/export_geoparquet.py` + `scripts/duckdb_spatial.py` + `data/gpq/` | ✅ |
| W4 | 集成：STAC 检索 → COG 读取 → NDVI | `scripts/data_pipeline.py` + `data/output/ndvi_*.png`（高斯平滑出图） | ✅ |

## 项目结构

```
GeoSense/
├── backend/
│   ├── core/               # 核心能力层（与业务解耦）
│   │   ├── config.py       # 全局配置（.env 加载，12-Factor）
│   │   ├── llm.py          # LLM 客户端（chat / chat_stream / chat_with_tools）
│   │   └── embedding.py    # 本地 Embedding 引擎（bge-small-zh-v1.5）
│   ├── agent/              # Agent 模块
│   │   ├── prompts.py      # 5 个 GIS 场景系统提示词（W2）
│   │   ├── tools.py        # 3 个 GIS 工具 + Schema + 分发器（W4）
│   │   ├── spatial_tools.py # 7 个空间领域工具（第3月W2，数据已切 PostGIS）
│   │   ├── sql_tools.py    # spatial_sql：自然语言→PostGIS SQL→校验→执行（补课）
│   │   ├── langchain_tools.py # 工具 → LangChain @tool 适配（第3月W1）
│   │   ├── graph.py        # LangGraph ReAct / 多工具 Agent 构建
│   │   ├── workflow.py     # 手写 StateGraph 多步工作流 + 错误处理（第3月W3）
│   │   └── data/           # 深圳 POI 样例数据集（sz_poi.json）
│   ├── api/                # Web 接口层（第3月W4）
│   │   └── main.py         # FastAPI + SSE 流式 /chat 接口
│   ├── db/                 # 数据层（阶段1 补课）
│   │   ├── connection.py   # PostGIS 连接 + 参数化查询 + GeoJSON 转换
│   │   ├── schema.py       # 表结构单一事实来源（DDL / 表白名单 / LLM schema 提示）
│   │   ├── sql_validator.py # 只读 SQL 安全校验（注入防护）
│   │   └── init/           # docker 首次启动的建扩展 SQL
│   ├── data/               # 栅格读取层（阶段2 第4月 W2）
│   │   └── cog_reader.py   # COG 在线读取封装（rio-tiler：瓦片 / 局部读取 / 8位渲染）
│   └── rag/                # RAG 模块（第2月）
│       ├── vector_store.py # 向量库封装（Chroma，cosine 空间）
│       ├── doc_processor.py # 文档处理（PDF/MD/HTML 抽取 + 递归切片）
│       ├── retriever.py    # 混合检索（BM25 关键词 + 语义 + RRF 融合）
│       ├── evaluation.py   # 检索评估指标（HitRate/Recall/MRR）
│       └── data/           # 知识库种子数据 + 示例文档
│           ├── gis_knowledge.json  # 22 条 GIS 知识 chunk
│           ├── eval_dataset.json   # 12 条评估查询（含标注答案）
│           └── docs/       # 示例文档（postgis_intro.md / ogc_wms.html）
├── scripts/                # 各周交付物（可独立运行）
│   ├── llm_basics.py       # W1：LLM 四个基础实验
│   ├── prompt_test.py      # W2：5 模板实测（含 JSON 质量门禁）
│   ├── embedding_search.py # W3：GIS 术语语义搜索
│   ├── gis_tools.py        # W4：Function Calling 闭环
│   ├── vector_store.py     # 第2月 W1：向量库建库检索
│   ├── doc_processor.py    # 第2月 W2：文档切片 + 入库检索
│   ├── rag_chain.py        # 第2月 W3：混合检索 + RAG 问答
│   ├── rag_evaluation.py   # 第2月 W4：检索评估 + 优化
│   ├── basic_agent.py      # 第3月 W1：ReAct Agent
│   ├── gis_agent.py        # 第3月 W2：多工具空间分析 Agent
│   ├── workflow_agent.py   # 第3月 W3：多步工作流 + 错误处理
│   ├── seed_postgis.py     # 补课：PostGIS 建表 + 灌样例数据
│   ├── fetch_osm.py        # 补课：从 OSM 拉取深圳真实边界 + POI 并入库
│   ├── postgis_sql_agent.py # 补课：SQL Agent 演示（--check 安全检查 / 完整流水线）
│   ├── make_demo_raster.py # 阶段2 W1：合成演示栅格（普通 GeoTIFF）
│   ├── cog_generator.py    # 阶段2 W1：普通 GeoTIFF → COG（含对比报告）
│   ├── cog_range_demo.py   # 阶段2 W2：overview 对比 + HTTP Range 协议验证
│   ├── tile_server.py      # 阶段2 W2：COG 在线瓦片服务（FastAPI /tiles/{z}/{x}/{y}.png）
│   ├── satellite_download.py # 阶段2 W3：遥感仓库构建（合成/真实 Sentinel-2 双模式）
│   ├── cog_benchmark.py   # 阶段2 W4：普通 GeoTIFF vs COG 性能基准（4096 测试对）
│   ├── stac_catalog.py    # 第5月 W1：为 COG 影像创建 STAC 目录（pystac）
│   ├── export_geoparquet.py # 第5月 W3：PostGIS → GeoParquet 导出
│   ├── duckdb_spatial.py  # 第5月 W3：DuckDB 空间查询 + 百万级基准
│   └── data_pipeline.py   # 第5月 W4：STAC 检索 → COG 读取 → NDVI（统计用原始值，出图前高斯平滑）
├── stac_api/               # 第5月 W2：STAC API 服务（FastAPI，/collections、/search）
│   └── main.py             # 轻量 STAC API（读 data/stac，datetime/bbox/limit 过滤）
├── frontend/               # 前端（第3月W4）
│   └── index.html          # Vue3 + MapboxGL 对话界面（底图 POI 聚合 + 区边界高亮 + 查询结果上图）
├── docker/                 # 容器构建（补课）
│   └── Dockerfile.postgis  # PostGIS + pgvector 自建镜像
├── docker-compose.yml      # 数据基础设施（PostgreSQL 16 + PostGIS）
├── data/                   # 运行时数据（向量库 / 中间产物，git 忽略）
├── prompt_patterns.md      # W2 手册：Prompt 设计原理
├── requirements.txt
├── .env.example            # 配置模板（复制为 .env 后填 Key）
└── README.md
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖（建议 Python 3.11+）
python3 -m venv .venv
source .venv/bin/activate          # ⚠ 必须先激活，否则会用系统 Python（版本不一致会报错）
pip install -r requirements.txt

# 2. 配置 DeepSeek API Key（https://platform.deepseek.com/ 创建）
cp .env.example .env        # 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 按周运行交付物（均可独立运行）
python scripts/llm_basics.py          # W1：LLM 调用
python scripts/prompt_test.py         # W2：Prompt 模板
python scripts/embedding_search.py    # W3：语义搜索（首次下载 bge 模型）
python scripts/gis_tools.py           # W4：工具调用
python scripts/vector_store.py        # 第2月W1：向量库
python scripts/doc_processor.py       # 第2月W2：文档切片 + 入库检索
python scripts/rag_chain.py           # 第2月W3：混合检索 + RAG 问答
python scripts/rag_evaluation.py      # 第2月W4：检索评估 + 优化
python scripts/basic_agent.py         # 第3月W1：ReAct Agent
python scripts/gis_agent.py           # 第3月W2：多工具空间分析 Agent
python scripts/workflow_agent.py      # 第3月W3：多步工作流 + 错误处理

# 4. 启动 PostGIS 并灌入数据（阶段1 补课）
docker compose up -d                       # 起 PostgreSQL 16 + PostGIS（首次自动建扩展）
python scripts/seed_postgis.py             # （可选）建表 + 导入小样例数据
python scripts/fetch_osm.py                # 从 OpenStreetMap 拉取深圳 10 区真实边界 + 3000+ POI 并入库
python scripts/postgis_sql_agent.py --check   # 安全自检（连通 + 注入拦截，不调 LLM）

# 4.5 阶段2 第4月 W1：合成栅格 + 转 COG（无需网络）
python scripts/make_demo_raster.py                    # 生成合成演示栅格 data/demo/demo.tif
python scripts/cog_generator.py data/demo/demo.tif    # 转 COG 并打印「普通 vs COG」对比表
python scripts/cog_range_demo.py                      # W2：overview 对比 + Range 协议验证
uvicorn scripts.tile_server:app --host 127.0.0.1 --port 8001   # W2：COG 瓦片服务（/tiles/{z}/{x}/{y}.png）
python scripts/satellite_download.py          # W3：合成 5 景 + 真彩色预览 + manifest
python scripts/satellite_download.py --real   # W3：真实 Sentinel-2（earth-search）远程裁剪入库
python scripts/cog_benchmark.py               # W4：COG 性能基准（自动生成 4096 测试对）
python scripts/stac_catalog.py                # 第5月W1：为 COG 影像创建 STAC 目录（pystac 校验）
uvicorn stac_api.main:app --port 8002         # 第5月W2：STAC API 服务（/search 支持 datetime/bbox）
python scripts/export_geoparquet.py           # 第5月W3：PostGIS → GeoParquet（需容器运行）
python scripts/duckdb_spatial.py              # 第5月W3：DuckDB 空间查询 + 500万点基准
# 第5月W4：端到端流水线（需 STAC API :8002 先起）
uvicorn stac_api.main:app --port 8002 &        # 后台启 STAC API
python scripts/data_pipeline.py                # STAC 检索 → COG → NDVI（出图前高斯平滑，统计用原始值）
# W4 前端：打开 http://127.0.0.1:8000/，地图底图已叠加深圳湾卫星影像（可开关）

# 5. 启动 Web 界面（第3月W4）
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

> ⚠ **环境一致性**：项目统一用 `.venv`（Python 3.13.12）。若误用系统 Python 运行，
> 会因 chromadb 版本不一致（如系统 0.6.3 vs 项目 1.5.9）报 `KeyError: '_type'`。
> 如遇此类错误，删除 `data/vectorstore/` 后用 `.venv/bin/python` 重跑即可重建。
```

> 注：`embedding_search.py` / `vector_store.py` 使用本地开源 Embedding 模型
> （BAAI/bge-small-zh-v1.5，约 100MB，首次运行自动下载），无需 API Key。

## 技术选型

| 层 | 技术 | 说明 |
|----|------|------|
| LLM | DeepSeek（OpenAI 兼容协议） | 主力推理，Function Calling 强 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地开源，零成本离线 |
| 向量库（开发） | Chroma | 本地持久化，cosine 空间 |
| 向量库（生产） | pgvector | 直连 PostgreSQL，阶段2 RAG 迁移时切换（已随容器预留） |
| 空间数据库 | PostgreSQL 16 + PostGIS 3.4 | 空间查询 / SQL 生成执行（阶段1 补课引入） |
| 栅格处理 | rasterio + rio-cogeo | 栅格读写 / COG 转换（阶段2 第4月引入） |
| 矢量分析 | GeoParquet + DuckDB Spatial | 列式空间分析，百万级秒查（阶段2 第5月引入） |
| GIS 计算 | pyproj + shapely | 测地线距离 / 缓冲区 / 坐标转换 |
| Agent 框架 | LangGraph | 第3月引入（当前为手写主循环） |

## 里程碑

- [x] **第1月** 双模型 API（DeepSeek + 本地 bge）、结构化 Prompt（5/5 通过）、
  Function Calling（3/3 命中）、Token/上下文理解
- [x] **第2月** 100+ 文档 GIS 知识库、RAG 准确率 > 70%、混合检索、延迟 < 2s
- [x] **第3月** ReAct Agent、多工具空间分析、流式输出、前端界面
- [x] **补课** PostGIS SQL 生成 + 只读安全校验 + OSM 真实数据（10 区边界 + 3000+ POI）+ 前端地图可视化
- [ ] 第4-6月 STAC + COG + GeoParquet + DuckDB 空间大数据（第4月 ✅ 全4周；第5月 ✅ W1 STAC 目录 / W2 STAC API / W3 GeoParquet+DuckDB 500万点 748ms / **W4 STAC→COG→NDVI 端到端流水线**）
- [ ] 第7-9月 遥感 AI（分割/检测/变化检测）+ 模型服务化
- [ ] 第10-12月 多 Agent + 自动制图 + 自动报告 + 端到端平台
