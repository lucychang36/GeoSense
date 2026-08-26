# GeoAI + LLM Agent + 空间数据基础设施学习路线（增强版）

## 目标定位

从 GIS 全栈开发工程师升级为：

> GeoAI Engineer / Spatial Intelligence Engineer

核心能力：

- GIS系统架构
- 空间数据基础设施
- 遥感AI分析
- LLM Agent开发
- 自然语言驱动GIS分析
- 自动制图与空间智能应用

---

## 一、当前能力评估

已有优势：

| 领域 | 能力 | 可直接迁移到 |
|------|------|-------------|
| Vue WebGIS | ★★★★★ | Agent前端交互层 |
| MapboxGL/OpenLayers/CesiumJS | ★★★★★ | 自动制图可视化层 |
| PostgreSQL/PostGIS | ★★★★☆ | Agent工具层（空间查询工具） |
| Python GIS生态 | ★★★★☆ | Agent后端 + AI模型推理 |
| QGIS/ArcGIS/SuperMap | ★★★★☆ | 制图知识迁移到AI Cartography |

需要补强：

| 领域 | 目标 | 优先级 |
|------|------|--------|
| 深度学习 | 掌握GeoAI模型应用 | P0 |
| LLM工程 | 掌握Agent/RAG/工具调用 | P0 |
| 云原生空间数据 | 掌握STAC/COG/GeoParquet | P1 |
| 空间大数据 | 掌握分布式空间计算 | P2 |
| 自动制图 | 掌握AI Cartography | P1 |

---

## 二、总体学习路线（12个月）

```
阶段1: LLM工程基础        阶段2: 空间数据基础设施      阶段3: GeoAI模型应用        阶段4: GeoAI Agent综合系统
(0-3个月)                 (3-6个月)                   (6-9个月)                   (9-12个月)

┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Transformer原理 │      │  COG云原生栅格  │      │  PyTorch深度学习 │      │  多Agent协作    │
│  Prompt工程      │      │  STAC数据目录   │      │  语义分割模型    │      │  自动分析流水线  │
│  RAG系统         │  →   │  GeoParquet     │  →   │  目标检测模型    │  →   │  自动制图引擎    │
│  Function Call   │      │  DuckDB Spatial │      │  SAM遥感适配     │      │  报告自动生成    │
│  LangGraph Agent │      │  Apache Sedona  │      │  变化检测        │      │  端到端平台      │
└─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

# 阶段1：LLM Agent工程（0-3个月）

## 学习目标

能够开发：
- GIS智能问答
- 空间分析Agent
- GIS工具调用系统

---

## 第1个月：LLM基础

### 学习内容

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| Transformer原理 | Attention机制、编码器-解码器、位置编码 | [3Blue1Brown Transformer可视化](https://www.youtube.com/watch?v=wjZofJX0v4M)、[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| Token与Embedding | BPE分词、词向量、语义空间 | [OpenAI Tokenizer](https://platform.openai.com/tokenizer)、[Word2Vec原理](https://jalammar.github.io/illustrated-word2vec/) |
| Context Window | 上下文长度、注意力衰减、长文本策略 | 各模型官方文档 |
| Prompt Engineering | Zero-shot、Few-shot、CoT、结构化提示 | [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)、[Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) |
| Function Calling | 工具定义、参数schema、并行调用 | [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)、[LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/) |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | Transformer原理学习 + 本地跑通LLM API调用 | `llm_basics.ipynb`：调用OpenAI/Claude API完成对话 |
| W2 | Prompt工程实战：GIS场景提示词设计 | `prompt_patterns.md`：5种GIS场景的Prompt模板 |
| W3 | Embedding与向量相似度实践 | `embedding_search.py`：GIS术语语义搜索 |
| W4 | Function Calling基础 + GIS工具定义 | `gis_tools.py`：定义3个GIS工具函数（距离计算、缓冲区、坐标转换） |

### 里程碑检查

- [ ] 能用Python调用至少2个大模型API（OpenAI + 开源模型）
- [ ] 能设计结构化Prompt完成GIS知识问答
- [ ] 能定义Function Calling工具并让LLM正确调用
- [ ] 理解Token计费、上下文窗口限制

### 推荐模型

| 用途 | 模型 | 说明 |
|------|------|------|
| 主力模型 | GPT-4o / Claude 3.5 Sonnet | Function Calling能力强 |
| 开源替代 | Qwen2.5-72B / Llama 3.1 70B | 可本地部署 |
| 轻量测试 | Qwen2.5-7B / Phi-3 | 快速迭代 |
| Embedding | text-embedding-3-large / bge-m3 | 中英文支持好 |

---

## 第2个月：RAG系统

### 学习内容

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| 向量数据库 | ANN索引（HNSW/IVF）、相似度度量 | [Milvus官方教程](https://milvus.io/docs)、[pgvector文档](https://github.com/pgvector/pgvector) |
| Embedding模型 | 维度选择、多语言模型、领域适配 | [MTEB排行榜](https://huggingface.co/spaces/mteb/leaderboard) |
| 文档处理 | 切片策略、元数据、表格/图片处理 | [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) |
| 检索策略 | 语义检索、混合检索、重排序 | [Cohere Rerank](https://docs.cohere.com/docs/reranking)、[RAGFromScratch](https://github.com/langchain-ai/rag-from-scratch) |
| RAG评估 | 忠实度、相关性、召回率 | [RAGAS框架](https://github.com/explodinggradients/ragas) |

### 技术栈

| 组件 | 推荐方案 | 替代方案 |
|------|---------|---------|
| 框架 | LangChain | LlamaIndex |
| 向量库（开发） | FAISS / Chroma | pgvector（直接用PostgreSQL） |
| 向量库（生产） | Milvus | Qdrant / Weaviate |
| Embedding | OpenAI / BGE-M3 | Cohere |
| 重排序 | Cohere Rerank | BGE-Reranker |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | 向量数据库搭建 + Embedding实践 | `vector_store.py`：pgvector/Milvus存储GIS文档向量 |
| W2 | 文档处理流水线：GIS标准/论文切片 | `doc_processor.py`：支持PDF/Markdown/HTML切片 |
| W3 | RAG检索链路搭建 + 混合检索 | `rag_chain.py`：语义+关键词混合检索 |
| W4 | RAG评估与优化 | `rag_evaluation.py`：用RAGAS评估，优化切片/检索策略 |

### GIS知识库数据源

| 类型 | 数据源 | 获取方式 |
|------|--------|---------|
| GIS标准 | OGC标准文档（WMS/WFS/WMTS/WCS） | [OGC官网](https://www.ogc.org/standards) |
| 空间分析 | PostGIS官方文档、PySAL文档 | PostGIS Docs / PySAL Docs |
| 遥感论文 | IEEE TGRS、ISPRS顶会论文 | Google Scholar |
| 教程 | GeoPython、Automating GIS Processes | [AutoGIS](https://autogis-site.readthedocs.io/) |

### 里程碑检查

- [ ] 建成包含100+文档的GIS知识库
- [ ] RAG问答准确率 > 70%（人工评估20个问题）
- [ ] 检索延迟 < 2秒
- [ ] 实现混合检索（语义+关键词）

---

## 第3个月：Agent开发

### 学习内容

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| Agent架构 | ReAct、Plan-and-Execute、多Agent协作 | [Lilian Weng: LLM Agent](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| Tool Calling | 工具注册、参数验证、错误处理 | [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/) |
| Workflow编排 | 状态机、DAG、条件分支 | [LangGraph文档](https://langchain-ai.github.io/langgraph/) |
| Agent评估 | 轨迹评估、工具调用准确率 | [LangSmith](https://docs.smith.langchain.com/) |
| 生产化 | 流式输出、超时控制、成本监控 | LangSmith / Phoenix |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | LangGraph基础 + 单工具Agent | `basic_agent.py`：ReAct模式GIS问答Agent |
| W2 | 多工具Agent + 空间分析工具集 | `gis_agent.py`：集成PostGIS查询/缓冲分析/制图工具 |
| W3 | 多步推理工作流 + 错误处理 | `workflow_agent.py`：复杂分析多步编排 |
| W4 | Agent评估 + 前端界面 | `agent_ui/`：Vue + FastAPI的Agent交互界面 |

### GIS Agent工具集设计

```python
# 第3月需要实现的工具集
TOOLS = [
    "spatial_query",      # PostGIS空间查询
    "buffer_analysis",    # 缓冲区分析
    "overlay_analysis",   # 叠加分析
    "coordinate_transform", # 坐标转换
    "map_generation",     # 地图生成
    "data_retrieval",     # 数据检索
    "statistics_calc",    # 空间统计
]
```

### 里程碑检查

- [ ] Agent能正确理解"查询A市B区周边3公里学校"并调用工具
- [ ] 多步推理成功率 > 80%
- [ ] 实现流式输出
- [ ] 前端可交互展示Agent推理过程

---

# 阶段2：空间数据基础设施（3-6个月）

## 学习目标

掌握AI时代空间数据工程：从数据获取、存储、索引到大规模计算。

---

## 第4个月：云原生遥感数据

### COG（Cloud Optimized GeoTIFF）

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| COG原理 | 内部tiling、overview金字塔、HTTP Range读取 | [COG规范](https://www.cogeo.org/)、[rasterio COG指南](https://rasterio.readthedocs.io/en/latest/topics/cog.html) |
| GDAL/Rasterio | 读取、写入、重投影、重采样 | [GDAL Python API](https://gdal.org/api/python.html)、[rasterio教程](https://rasterio.readthedocs.io/) |
| 在线读取 | TiTiler、rio-tiler | [Titiler](https://developmentseed.org/titiler/)、[rio-tiler](https://cogeotiff.github.io/rio-tiler/) |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | GDAL/Rasterio基础 + COG生成 | `cog_generator.py`：普通GeoTIFF转COG |
| W2 | 在线COG读取 + rio-tiler | `tile_server.py`：用TiTiler提供瓦片服务 |
| W3 | 遥感数据下载 + COG仓库构建 | `satellite_warehouse/`：Sentinel-2数据COG仓库 |
| W4 | 性能优化 + 可视化验证 | WebGIS前端加载COG瓦片验证 |

### 推荐数据源

| 数据源 | 分辨率 | 获取方式 | 特点 |
|--------|--------|---------|------|
| Sentinel-2 | 10m | [Copernicus Data Space](https://dataspace.copernicus.eu/) | 免费开源，5天重访 |
| Landsat 8/9 | 30m | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) | 长时间序列 |
| MODIS | 250m-1km | [LP DAAC](https://lpdaac.usgs.gov/) | 每日数据 |
| Planet | 3-5m | [Planet Explorer](https://www.planet.com/explorer/) | 商业，有免费额度 |

---

## 第5个月：STAC + GeoParquet

### STAC（SpatioTemporal Asset Catalog）

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| STAC规范 | Item、Collection、Catalog、API | [STAC Spec](https://github.com/radiantearth/stac-spec) |
| pystac | Python创建/操作STAC | [pystac文档](https://pystac.readthedocs.io/) |
| stac-fastapi | STAC API服务 | [stac-fastapi](https://stac-utils.github.io/stac-fastapi/) |
| 公共STAC | Earth Search、Planetary Computer | [Earth Search](https://earth-search.aws.element84.com/v1)、[Planetary Computer](https://planetarycomputer.microsoft.com/) |

### GeoParquet

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| Apache Arrow | 列式内存格式、零拷贝 | [Arrow Python](https://arrow.apache.org/docs/python/) |
| GeoParquet | 空间列存储规范 | [GeoParquet Spec](https://geoparquet.org/) |
| DuckDB Spatial | 嵌入式空间分析数据库 | [DuckDB Spatial](https://duckdb.org/docs/extensions/spatial) |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | STAC规范学习 + pystac实践 | `stac_catalog/`：为Sentinel-2数据创建STAC目录 |
| W2 | stac-fastapi搭建STAC API | `stac_api/`：本地STAC API服务 |
| W3 | GeoParquet + DuckDB Spatial | `geo_parquet/`：千万级矢量数据DuckDB查询 |
| W4 | 集成：STAC检索 → COG读取 → 分析 | `data_pipeline.py`：端到端数据获取流水线 |

### 里程碑检查

- [ ] 能用STAC API检索指定时间/区域的Sentinel-2影像
- [ ] COG在线读取延迟 < 1秒/瓦片
- [ ] DuckDB Spatial千万级数据查询 < 5秒
- [ ] 实现数据获取自动化流水线

---

## 第6个月：空间大数据计算

### Apache Sedona + Spark GIS

| 主题 | 知识点 | 推荐资源 |
|------|--------|---------|
| Apache Sedona | 空间RDD、空间Join、空间索引 | [Sedona文档](https://sedona.apache.org/) |
| Spark GIS | 分布式空间计算、分区策略 | [Sedona Tutorial](https://sedona.apache.org/latest-snapshot/tutorial/) |
| 性能优化 | 空间分区、索引选择、Broadcast Join | Sedona性能调优文档 |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | Spark + Sedona环境搭建 | 本地Spark集群 + Sedona |
| W2 | 空间Join实战 | `spatial_join.py`：全国建筑物 × 人口网格 |
| W3 | 大规模空间聚合 | `spatial_aggregate.py`：按行政区聚合统计 |
| W4 | 性能对比 + 可视化 | `benchmark.py`：PostGIS vs DuckDB vs Sedona性能对比 |

### 推荐数据集

| 数据集 | 规模 | 获取方式 |
|--------|------|---------|
| Microsoft Global Building Footprints | 全球3亿+建筑 | [GitHub](https://github.com/microsoft/GlobalMLBuildingFootprints) |
| WorldPop人口网格 | 100m分辨率全球 | [WorldPop](https://www.worldpop.org/) |
| OpenStreetMap | 全球矢量 | [Geofabrik](https://download.geofabrik.de/) |
| GADM行政区 | 全球行政边界 | [GADM](https://gadm.org/) |

---

# 阶段3：GeoAI模型应用（6-9个月）

## 学习目标

成为AI空间应用工程师：能选型、部署、微调遥感AI模型。

---

## 第7个月：深度学习基础

| 主题 | 知识点 | 推荐资源 | 实践 |
|------|--------|---------|------|
| PyTorch基础 | Tensor、autograd、nn.Module | [PyTorch官方教程](https://pytorch.org/tutorials/) | 手写线性回归 + MLP |
| Dataset/DataLoader | 数据加载、增强、批处理 | [PyTorch Data](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) | 构建遥感影像Dataset |
| CNN | 卷积、池化、感受野、ResNet | [CS231n](https://cs231n.github.io/) | 用ResNet做地物分类 |
| 训练技巧 | 学习率、正则化、迁移学习 | [Fast.ai](https://course.fast.ai/) | 迁移学习微调 |

### 里程碑检查

- [ ] 能用PyTorch从头训练一个图像分类模型
- [ ] 理解反向传播、梯度下降、学习率调度
- [ ] 能使用预训练模型做迁移学习
- [ ] GPU训练环境搭建完成

### 推荐GPU环境

| 方案 | 说明 | 成本 |
|------|------|------|
| Google Colab Pro | 免费T4/A100，方便 | $10/月 |
| AutoDL / 矩池云 | 国内GPU租用 | ¥2-10/小时 |
| 本地RTX 4090 | 长期投入 | 一次性¥15k+ |

---

## 第8个月：遥感AI模型

### 语义分割

| 模型 | 特点 | 适用场景 | 资源 |
|------|------|---------|------|
| U-Net | 经典，小数据友好 | 地物提取 | [ segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) |
| DeepLab v3+ | 空间金字塔池化 | 复杂场景 | [PyTorch实现](https://github.com/jfzhang95/pytorch-deeplab-xception) |
| SegFormer | Transformer，无位置编码 | 高精度分割 | [官方实现](https://github.com/NVlabs/SegFormer) |

### 目标检测

| 模型 | 特点 | 适用场景 | 资源 |
|------|------|---------|------|
| YOLOv8/v11 | 实时检测 | 建筑/船舶/车辆 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| Faster R-CNN | 高精度 | 小目标检测 | [Detectron2](https://github.com/facebookresearch/detectron2) |

### 基础模型（Foundation Models）

| 模型 | 能力 | 资源 |
|------|------|------|
| Segment Anything (SAM) | 通用分割 | [SAM](https://github.com/facebookresearch/segment-anything)、[SAM-Geo](https://github.com/developmentseed/sam-geo) |
| CLIP | 图文对齐 | [OpenCLIP](https://github.com/mlfoundations/open_clip) |
| Prithvi (NASA-IBM) | 遥感基础模型 | [HuggingFace](https://huggingface.co/ibm-nasa-geospatial) |
| SatCLIP | 遥感位置嵌入 | [GitHub](https://github.com/microsoft/satclip) |

### 周计划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1 | 语义分割模型搭建 | `unet_segmentation.py`：U-Net建筑提取 |
| W2 | 模型训练 + 评估 | `train_segmentation.py`：完整训练流水线 |
| W3 | SAM遥感适配 | `sam_geo.py`：SAM自动建筑分割 |
| W4 | 目标检测实战 | `yolo_detection.py`：YOLOv8船舶检测 |

### 推荐遥感AI数据集

| 数据集 | 任务 | 规模 | 获取 |
|--------|------|------|------|
| [SpaceNet](https://spacenet.ai/) | 建筑提取/道路 | 万级 | AWS Open Data |
| [xView2](https://xview2.org/) | 灾后损伤评估 | 850k+ | DICE |
| [BigEarthNet](https://bigearth.net/) | 多标签分类 | 59万 | [HuggingFace](https://huggingface.co/datasets/BigEarthNet) |
| [SEN12MS](https://mediatum.ub.tum.de/1474000) | 多源分割 | 18万 | TUM |
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 城乡地物分割 | 5987 | GitHub |

---

## 第9个月：变化检测 + 模型服务化

### 变化检测

| 方法 | 说明 | 资源 |
|------|------|------|
| 差异图法 | 影像差值/比值 → 阈值 | 基础方法 |
| 深度学习变化检测 | Siamese网络 | [CDRL](https://github.com/likyoo/Change-Detection-Review) |
| 时序分析 | LSTM/Transformer时序建模 | [T-CycleNet](https://github.com/Banson-AI/T-CycleNet) |

### 模型服务化

| 组件 | 方案 | 说明 |
|------|------|------|
| 模型服务 | TorchServe / Triton | 生产部署 |
| API封装 | FastAPI | 接口定义 |
| 异步推理 | Celery + Redis | 长任务处理 |
| 监控 | Prometheus + Grafana | 服务监控 |

### 周计划

| W1 | 变化检测模型 | `change_detection.py`：Siamese网络城市扩张检测 |
| W2 | 模型服务化 | `model_service/`：FastAPI + TorchServe |
| W3 | 异步推理流水线 | `async_inference.py`：大影像异步分块推理 |
| W4 | 阶段3集成项目 | 蓝藻监测系统原型 |

---

# 阶段4：GeoAI Agent综合系统（9-12个月）

## 学习目标

构建完整空间智能平台：自然语言 → 自动分析 → 自动制图 → 自动报告。

---

## 第10个月：多Agent协作 + 自动制图

### 多Agent协作

| 主题 | 知识点 | 资源 |
|------|--------|------|
| 多Agent架构 | 分工协作、消息传递、冲突解决 | [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/) |
| Agent角色设计 | 规划Agent、数据Agent、分析Agent、制图Agent | 自研 |
| 状态管理 | 共享状态、检查点、回滚 | LangGraph State |

### AI Cartography（自动制图）

| 主题 | 知识点 | 资源 |
|------|--------|------|
| 自动符号化 | 根据数据类型自动选择符号/颜色 | [ColorBrewer](https://colorbrewer2.org/)、[CartoCSS](https://carto.com/developers/styling/cartocss/) |
| 自动标注 | 标注冲突检测与避让 | MapboxGL标注策略 |
| 地图综合 | 缩放级别自适应简化 | [geojson-simplify](https://github.com/maxogden/simplify-geojson)、PostGIS ST_Simplify |
| Text-to-Map | 自然语言→地图样式 | [Mapbox Style Spec](https://docs.mapbox.com/mapbox-gl-js/style-spec/)、[Kepler.gl](https://kepler.gl/) |
| AI配色 | LLM生成地图配色方案 | GPT-4 + ColorBrewer知识 |

### 周计划

| W1 | 多Agent架构设计 | `multi_agent/`：规划+数据+分析+制图4个Agent |
| W2 | 自动符号化引擎 | `auto_symbology.py`：根据数据类型自动配色 |
| W3 | 自动标注 + 地图综合 | `auto_cartography.py`：完整制图流水线 |
| W4 | Text-to-Map原型 | `text_to_map.py`：自然语言→Mapbox样式JSON |

---

## 第11个月：报告生成 + 系统集成

### 自动报告生成

| 组件 | 方案 | 说明 |
|------|------|------|
| 报告模板 | Markdown + Jinja2 | 结构化模板 |
| 图表生成 | Matplotlib + Folium | 统计图 + 地图 |
| LLM分析文本 | GPT-4o / Claude | 自然语言分析报告 |
| PDF导出 | WeasyPrint / Pandoc | 格式化输出 |

### 系统集成

| 层 | 技术 | 说明 |
|----|------|------|
| 前端 | Vue3 + MapboxGL + CesiumJS | 交互 + 2D/3D地图 |
| API | FastAPI | RESTful + WebSocket |
| Agent | LangGraph | 多Agent编排 |
| 工具层 | PostGIS / DuckDB / GDAL | 空间计算 |
| AI模型 | PyTorch + TorchServe | 模型推理 |
| 数据层 | STAC / COG / GeoParquet | 数据管理 |
| 部署 | Docker + Kubernetes | 容器化 |

### 周计划

| W1 | 报告生成引擎 | `report_generator/`：自动分析报告 |
| W2 | 前端集成 | `frontend/`：Agent交互 + 地图展示 |
| W3 | 后端集成 | `backend/`：API + Agent + 工具层 |
| W4 | 端到端测试 | `e2e_test.py`：完整流程测试 |

---

## 第12个月：优化 + 开源 + 求职

| 周次 | 任务 |
|------|------|
| W1 | 性能优化（推理加速、缓存、CDN） |
| W2 | 文档完善 + 开源准备 |
| W3 | 写技术博客（2-3篇深度文章） |
| W4 | 项目展示视频 + 求职准备 |

---

# 技术栈目标

```
Frontend:    Vue3 + MapboxGL + CesiumJS + Kepler.gl
Backend:     Python + FastAPI + WebSocket
Spatial:     PostGIS + DuckDB Spatial + Apache Sedona
Data:        STAC + COG + GeoParquet + Apache Arrow
AI:          PyTorch + Ultralytics + SAM + segmentation_models
LLM:         LangGraph + RAG + pgvector/Milvus + LangSmith
Deployment:  Docker + Kubernetes + Cloud (AWS/阿里云)
Monitoring:  Prometheus + Grafana + LangSmith
```

---

# 推荐项目路线

| 项目 | 时间 | 核心技能 | 交付物 |
|------|------|---------|--------|
| P1: GIS Chat Assistant | 1-3月 | LLM + RAG + Function Calling | GIS知识问答 + SQL生成Agent |
| P2: 遥感数据智能平台 | 3-6月 | STAC + COG + DuckDB | 数据目录 + 在线分析 |
| P3: GeoAI分析平台 | 6-9月 | PyTorch + 分割/检测 + 服务化 | 蓝藻监测系统 |
| P4: GeoAI Agent | 9-12月 | 多Agent + 自动制图 + 报告 | 空间智能分析助手 |

---

# 学习资源汇总

## 必读书单

| 书名 | 领域 | 优先级 |
|------|------|--------|
| 《深度学习》Ian Goodfellow | 深度学习理论 | P1 |
| 《动手学深度学习》李沐 | PyTorch实战 | P0 |
| 《大模型应用开发极简入门》黄佳 | LLM工程 | P0 |
| 《LangChain实战》 | Agent开发 | P0 |
| 《Python空间数据处理》 | Python GIS | P1 |
| 《遥感数字图像处理》 | 遥感基础 | P1 |

## 在线课程

| 课程 | 平台 | 说明 |
|------|------|------|
| [CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/) | Stanford | 计算机视觉经典 |
| [Fast.ai Practical Deep Learning](https://course.fast.ai/) | Fast.ai | 实战深度学习 |
| [AutoGIS](https://autogis-site.readthedocs.io/) | Helsinki | Python GIS自动化 |
| [EO-Tutorials](https://github.com/developmentseed/eo-tutorials) | DevSeed | 遥感数据工程 |
| [Deep Learning for RS](https://github.com/syamkakarla98/Deep-Learning-for-Remote-Sensing-Data) | GitHub | 遥感AI实战 |

## 必关社区

| 社区 | 说明 |
|------|------|
| [Awesome GeoAI](https://github.com/awesome-list/awesome-geoai) | GeoAI资源汇总 |
| [r/remoteSensing](https://reddit.com/r/remoteSensing) | 遥感社区 |
| [GIS Stack Exchange](https://gis.stackexchange.com/) | GIS问答 |
| [HuggingFace Spaces](https://huggingface.co/spaces) | AI模型Demo |
| [Development Seed Blog](https://developmentseed.org/blog/) | 空间数据工程前沿 |

---

# 长期职业目标

目标岗位：
- GeoAI Engineer
- Spatial Intelligence Engineer
- Geospatial AI Architect
- Digital Twin Architect

核心竞争力：

> 将GIS、AI、大模型、空间数据基础设施融合，构建下一代空间智能系统。

---
---

# 三、练习开发任务：GeoSense — 空间智能分析Agent

## 项目概述

**GeoSense** 是一个基于自然语言的空间智能分析平台。用户用自然语言提问，系统通过多Agent协作，自动完成数据检索、空间分析、AI推理、自动制图和报告生成。

**一句话描述**：你说"分析深圳湾过去3年的水质变化"，GeoSense自动获取卫星数据、运行水质AI模型、生成变化地图和分析报告。

> 本项目贯穿12个月学习路线，每个阶段产出可独立运行的模块，最终集成为完整系统。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户界面 (Vue3 + MapboxGL)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 对话面板  │  │ 地图视图  │  │ 分析结果  │  │ 报告查看  │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
└───────┼─────────────┼─────────────┼─────────────┼──────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI + WebSocket)                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌───────────────────┐ ┌─────────────────┐ ┌──────────────┐
│  LangGraph        │ │  报告生成服务    │ │  文件管理     │
│  多Agent编排       │ │  (Jinja2+LLM)  │ │  (COG/报告)   │
└───────┬───────────┘ └─────────────────┘ └──────────────┘
        │
   ┌────┼────┬────────┬────────┬────────┐
   ▼    ▼    ▼        ▼        ▼        ▼
┌──────┐┌────┐┌────────┐┌────────┐┌────────┐
│规划   ││数据 ││分析    ││制图    ││报告    │
│Agent  ││Agent││Agent  ││Agent  ││Agent  │
└───┬──┘└──┬─┘└───┬────┘└───┬────┘└───┬───┘
    │      │      │         │         │
    │      ▼      │         │         │
    │ ┌────────┐  │         │         │
    │ │STAC API│  │         │         │
    │ │COG读取 │  │         │         │
    │ └────────┘  │         │         │
    │      │      │         │         │
    │      ▼      ▼         │         │
    │ ┌──────────────────┐  │         │
    │ │  AI模型推理服务   │  │         │
    │ │ (TorchServe)     │  │         │
    │ └──────────────────┘  │         │
    │                       │         │
    │ ┌──────────────────┐  │         │
    │ │  PostGIS / DuckDB│◄─┘         │
    │ │  空间计算引擎     │            │
    │ └──────────────────┘            │
    │                       │         │
    │ ┌──────────────────┐  │         │
    │ │  RAG知识库        │◄─────────┘
    │ │ (pgvector)       │
    │ └──────────────────┘
```

---

## 分阶段交付计划

### 阶段1交付：GeoSense Chat（第1-3月）

**目标**：自然语言GIS问答 + 空间SQL生成 + 知识库检索

#### 功能需求

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| 对话界面 | 自然语言对话 + 流式输出 | Vue3 + FastAPI + WebSocket |
| GIS知识库 | GIS标准/PostGIS文档检索 | pgvector + RAG |
| SQL生成 | 自然语言→PostGIS SQL | LLM Function Calling |
| SQL执行 | 安全执行生成的SQL | PostGIS + SQL验证层 |
| 地图展示 | 查询结果可视化 | MapboxGL + GeoJSON |

#### 示例交互

```
用户：查找深圳南山区3公里内的所有学校
Agent：[工具调用] spatial_query → 生成PostGIS SQL → 执行查询
Agent：[结果] 找到47所学校，已在地图上标注。
用户：把它们按距离排序，最近的高亮显示
Agent：[工具调用] 重新排序 → 更新地图样式
```

#### 数据准备

```sql
-- 示例PostGIS数据
CREATE TABLE schools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200),
    type VARCHAR(50),       -- 小学/中学/大学
    geom GEOMETRY(Point, 4326)
);

CREATE TABLE admin_boundary (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    level VARCHAR(20),      -- 省/市/区
    geom GEOMETRY(Polygon, 4326)
);

-- 插入测试数据（可用OSM数据导入）
```

#### 交付物清单

```
geosense/
├── backend/
│   ├── main.py              # FastAPI入口
│   ├── agent/
│   │   ├── graph.py         # LangGraph Agent定义
│   │   ├── tools.py         # GIS工具函数
│   │   └── prompts.py       # 系统提示词
│   ├── rag/
│   │   ├── indexer.py       # 文档索引
│   │   ├── retriever.py     # 检索器
│   │   └── data/            # GIS知识文档
│   ├── db/
│   │   ├── connection.py    # PostGIS连接
│   │   └── sql_validator.py # SQL安全验证
│   └── api/
│       ├── chat.py          # 对话接口
│       └── map.py           # 地图数据接口
├── frontend/
│   ├── src/
│   │   ├── views/Chat.vue   # 对话界面
│   │   ├── components/
│   │   │   ├── MapView.vue  # 地图组件
│   │   │   └── MessageList.vue
│   │   └── api/client.ts    # API客户端
│   └── package.json
├── docker-compose.yml       # PostgreSQL+pgvector+后端
└── README.md
```

#### 验收标准

- [ ] 自然语言查询成功率 > 80%
- [ ] SQL注入防护完整
- [ ] 查询结果正确可视化
- [ ] 知识库检索准确率 > 70%
- [ ] 流式响应延迟 < 500ms首token

---

### 阶段2交付：GeoSense DataHub（第4-6月）

**目标**：STAC遥感数据管理 + COG在线分析 + 大规模空间计算

#### 新增功能

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| STAC目录 | 遥感数据检索与浏览 | stac-fastapi + pystac |
| COG服务 | 在线栅格瓦片读取 | TiTiler + rio-tiler |
| 数据浏览 | 影像目录地图浏览 | MapboxGL + STAC API |
| 空间计算 | DuckDB大规模空间分析 | DuckDB Spatial |
| 数据流水线 | STAC→COG→分析自动化 | Python pipeline |

#### 示例交互

```
用户：搜索2024年深圳湾的Sentinel-2影像，云量小于20%
Agent：[工具调用] stac_search → STAC API检索
Agent：找到23景影像，时间范围2024-01至2024-12，已在地图上标注覆盖范围。
用户：选最近的一景，计算NDWI水体指数
Agent：[工具调用] cog_read + ndwi_calc → COG在线读取 + 计算
Agent：NDWI计算完成，水体区域已用蓝色高亮显示。
```

#### 新增工具定义

```python
NEW_TOOLS_PHASE2 = [
    "stac_search",          # STAC数据检索
    "cog_read",             # COG在线读取
    "raster_calc",          # 栅格计算（NDVI/NDWI等）
    "duckdb_spatial_query", # DuckDB空间查询
    "raster_stats",         # 栅格分区统计
    "image_preview",        # 影像预览生成
]
```

#### 交付物增量

```
geosense/
├── backend/
│   ├── data/
│   │   ├── stac_api.py         # STAC API服务
│   │   ├── cog_reader.py       # COG读取服务
│   │   ├── raster_engine.py    # 栅格计算引擎
│   │   └── duckdb_engine.py    # DuckDB空间分析
│   ├── agent/
│   │   └── tools.py            # + 新增阶段2工具
│   └── api/
│       ├── stac.py             # STAC接口
│       └── raster.py           # 栅格接口
├── frontend/
│   └── src/
│       └── components/
│           ├── StacBrowser.vue  # STAC浏览器
│           ├── RasterLayer.vue  # 栅格图层
│           └── DataCatalog.vue  # 数据目录
└── data/
    ├── stac/                    # STAC目录数据
    └── cogs/                    # COG影像存储
```

#### 验收标准

- [ ] STAC检索响应 < 2秒
- [ ] COG瓦片加载 < 1秒
- [ ] NDWI等指数计算正确
- [ ] DuckDB千万级查询 < 5秒
- [ ] Agent能串联STAC检索+COG读取+分析

---

### 阶段3交付：GeoSense AI（第7-9月）

**目标**：遥感AI模型推理 + 变化检测 + 模型服务化

#### 新增功能

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| 地物分割 | 建筑/水体/植被自动提取 | U-Net / SAM |
| 目标检测 | 船舶/车辆检测 | YOLOv8 |
| 变化检测 | 多时相变化分析 | Siamese网络 |
| 模型服务 | 异步推理API | TorchServe + Celery |
| 结果可视化 | AI结果叠加地图 | MapboxGL |

#### 示例交互

```
用户：对这幅深圳湾影像做建筑提取
Agent：[工具调用] ai_inference(building_segmentation) → 异步推理
Agent：建筑提取完成，检测到12,347栋建筑，总面积8.2km²。
用户：和2022年对比，哪些区域新增了建筑？
Agent：[工具调用] change_detection → 多时相变化检测
Agent：检测到3个主要扩张区域，已用红色标注在地图上。新增建筑面积1.5km²，主要分布在XX片区。
```

#### 新增工具定义

```python
NEW_TOOLS_PHASE3 = [
    "ai_inference",         # AI模型推理
    "change_detection",     # 变化检测
    "object_detection",     # 目标检测
    "area_calculation",     # 面积统计
    "model_list",           # 可用模型列表
]
```

#### AI模型清单

| 模型 | 任务 | 输入 | 输出 |
|------|------|------|------|
| U-Net (建筑) | 语义分割 | Sentinel-2 RGB+NIR | 建筑mask |
| SAM-Geo | 通用分割 | 高分影像 | 多类地物mask |
| YOLOv8 | 目标检测 | 高分影像 | 船舶/车辆bbox |
| Siamese-CD | 变化检测 | 两期影像 | 变化mask |

#### 交付物增量

```
geosense/
├── ai/
│   ├── models/
│   │   ├── unet_building.py     # 建筑分割模型
│   │   ├── sam_geo.py           # SAM遥感适配
│   │   ├── yolo_detector.py     # 目标检测
│   │   └── change_detector.py   # 变化检测
│   ├── service/
│   │   ├── inference_api.py     # 推理API
│   │   ├── task_queue.py        # Celery异步队列
│   │   └── model_registry.py    # 模型注册中心
│   └── weights/                 # 模型权重
├── backend/
│   └── api/
│       └── ai.py                # AI推理接口
└── frontend/
    └── src/components/
        ├── AiResult.vue          # AI结果展示
        └── InferencePanel.vue    # 推理面板
```

#### 验收标准

- [ ] 建筑分割IoU > 0.75（测试集）
- [ ] 推理服务支持并发
- [ ] 变化检测能识别城市扩张
- [ ] AI结果正确叠加地图
- [ ] 异步推理状态实时更新

---

### 阶段4交付：GeoSense Full System（第10-12月）

**目标**：多Agent协作 + 自动制图 + 自动报告 = 完整空间智能平台

#### 新增功能

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| 多Agent编排 | 规划→数据→分析→制图→报告 | LangGraph多Agent |
| 自动制图 | AI驱动符号化+标注+综合 | Mapbox Style + LLM |
| 自动报告 | 分析报告自动生成 | Jinja2 + LLM + PDF |
| Text-to-Map | 自然语言→地图样式 | LLM + Mapbox Style Spec |
| 端到端 | 自然语言→完整分析报告 | 全栈集成 |

#### 终极示例交互

```
用户：分析深圳过去5年的城市扩张情况，重点关注南山区和宝安区

[规划Agent] 分解任务：
  1. 获取2019和2024年深圳Sentinel-2影像
  2. 运行建筑分割模型
  3. 进行变化检测
  4. 按行政区统计扩张面积
  5. 生成扩张专题地图
  6. 生成分析报告

[数据Agent] 检索STAC → 下载COG → 读取影像
[分析Agent] AI推理 → 变化检测 → 面积统计
[制图Agent] 自动配色 → 标注 → 生成专题图
[报告Agent] 生成Markdown报告 → 导出PDF

GeoSense：分析完成！
  📊 深圳城市扩张分析报告（2019-2024）
  - 总新增建筑面积：23.6 km²
  - 南山区扩张：5.2 km²（主要在XX片区）
  - 宝安区扩张：12.1 km²（主要在XX片区）
  - 扩张速率：4.7 km²/年
  
  📁 交付物：
  - 扩张变化地图（已展示）
  - 分析报告PDF（可下载）
  - 原始数据（可下载）
```

#### 多Agent架构设计

```python
# LangGraph多Agent状态图
from langgraph.graph import StateGraph, MessagesState, START, END

# Agent节点
def planner_agent(state):
    """规划Agent：分解用户请求为子任务"""
    pass

def data_agent(state):
    """数据Agent：STAC检索 + COG读取"""
    pass

def analysis_agent(state):
    """分析Agent：AI推理 + 空间分析"""
    pass

def cartography_agent(state):
    """制图Agent：自动制图 + 样式生成"""
    pass

def report_agent(state):
    """报告Agent：报告生成 + PDF导出"""
    pass

# 构建状态图
graph = StateGraph(MessagesState)
graph.add_node("planner", planner_agent)
graph.add_node("data", data_agent)
graph.add_node("analysis", analysis_agent)
graph.add_node("cartography", cartography_agent)
graph.add_node("report", report_agent)

# 定义流转
graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", route_to_subagent)
graph.add_edge("data", "analysis")
graph.add_edge("analysis", "cartography")
graph.add_edge("cartography", "report")
graph.add_edge("report", END)
```

#### 自动制图引擎设计

```python
class AutoCartographyEngine:
    """AI驱动自动制图引擎"""

    def generate_style(self, data_type, analysis_result):
        """根据数据类型和分析结果生成地图样式"""
        # 1. LLM分析数据特征 → 推荐可视化方案
        # 2. 选择配色方案（ColorBrewer规则）
        # 3. 选择符号化方式（点/线/面/热力图）
        # 4. 生成Mapbox Style JSON

    def auto_label(self, features, zoom_level):
        """自动标注：冲突检测 + 避让"""

    def auto_generalize(self, geometry, zoom_level):
        """自动综合：根据缩放级别简化"""

    def text_to_style(self, natural_language):
        """自然语言 → 地图样式"""
        # "用红色渐变显示人口密度" → Mapbox Style JSON
```

#### 自动报告模板

```markdown
# {{ title }}

## 概述
{{ summary }}

## 分析区域
{{ region_description }}

## 数据来源
{{ data_sources }}

## 分析方法
{{ methodology }}

## 分析结果
{{ results }}

### 统计数据
{{ statistics_table }}

### 变化地图
{{ map_image }}

## 结论与建议
{{ conclusions }}

## 附录
{{ appendix }}
```

#### 交付物增量

```
geosense/
├── backend/
│   ├── agent/
│   │   ├── multi_agent.py       # 多Agent编排
│   │   ├── planner.py           # 规划Agent
│   │   ├── data_agent.py        # 数据Agent
│   │   ├── analysis_agent.py    # 分析Agent
│   │   ├── cartography_agent.py # 制图Agent
│   │   └── report_agent.py      # 报告Agent
│   ├── cartography/
│   │   ├── style_engine.py      # 样式引擎
│   │   ├── auto_label.py        # 自动标注
│   │   └── text_to_map.py       # Text-to-Map
│   └── report/
│       ├── generator.py         # 报告生成器
│       ├── templates/           # 报告模板
│       └── pdf_export.py        # PDF导出
├── frontend/
│   └── src/
│       ├── views/Analysis.vue   # 分析工作台
│       └── components/
│           ├── AgentFlow.vue    # Agent流程展示
│           ├── ReportViewer.vue # 报告查看器
│           └── StyleEditor.vue  # 样式编辑器
└── docker/
    ├── Dockerfile.backend
    ├── Dockerfile.frontend
    └── docker-compose.prod.yml
```

#### 最终验收标准

- [ ] 多Agent协作成功率 > 85%
- [ ] 自动制图结果美观、信息清晰
- [ ] 报告内容准确、结构完整
- [ ] 端到端响应 < 5分钟（含AI推理）
- [ ] 系统可Docker一键部署
- [ ] 完整文档 + 演示视频

---

## 技术选型总表

| 层 | 技术 | 版本 | 用途 |
|----|------|------|------|
| 前端框架 | Vue3 | 3.4+ | SPA框架 |
| 地图引擎 | MapboxGL JS | 3.x | 2D地图 |
| 三维引擎 | CesiumJS | 1.115+ | 3D地图 |
| 后端框架 | FastAPI | 0.110+ | RESTful API |
| Agent框架 | LangGraph | 0.2+ | 多Agent编排 |
| LLM | GPT-4o / Claude 3.5 | - | 推理引擎 |
| 向量数据库 | pgvector | 0.7+ | RAG检索 |
| 空间数据库 | PostgreSQL + PostGIS | 16 + 3.4 | 空间计算 |
| 嵌入式分析 | DuckDB Spatial | 1.0+ | 大规模分析 |
| 栅格处理 | rasterio + GDAL | 1.3+ | COG处理 |
| STAC | stac-fastapi | 3.x | 数据目录 |
| 瓦片服务 | TiTiler | 0.18+ | COG瓦片 |
| AI框架 | PyTorch | 2.2+ | 模型训练/推理 |
| 模型服务 | TorchServe | 0.11+ | 模型部署 |
| 分割模型 | segmentation_models.pytorch | 0.3+ | U-Net等 |
| 检测模型 | Ultralytics YOLOv8 | 8.2+ | 目标检测 |
| 基础模型 | SAM | - | 通用分割 |
| 异步队列 | Celery + Redis | 5.3+ | 异步推理 |
| 容器化 | Docker + Compose | 24+ | 部署 |
| 监控 | LangSmith | - | Agent追踪 |

---

## 开发节奏建议

| 时间 | 开发重点 | 学习配合 |
|------|---------|---------|
| 第1月 | 搭建项目骨架 + LLM API对接 | Transformer + Prompt |
| 第2月 | RAG知识库 + GIS文档索引 | RAG系统 |
| 第3月 | Agent工具链 + 对话界面 | LangGraph Agent |
| 第4月 | STAC数据目录 + COG服务 | COG + GDAL |
| 第5月 | DuckDB分析 + 数据流水线 | STAC + GeoParquet |
| 第6月 | 阶段2集成 + 性能优化 | Sedona空间计算 |
| 第7月 | PyTorch基础 + 模型训练环境 | 深度学习基础 |
| 第8月 | 分割/检测模型 + 推理服务 | 遥感AI模型 |
| 第9月 | 变化检测 + 阶段3集成 | 变化检测 |
| 第10月 | 多Agent + 自动制图 | 多Agent + AI Cartography |
| 第11月 | 报告生成 + 全系统集成 | 系统集成 |
| 第12月 | 优化 + 文档 + 开源 | 求职准备 |

---

## 项目亮点（求职用）

1. **全栈AI系统**：从自然语言到空间分析到报告生成，完整闭环
2. **多Agent协作**：LangGraph编排5个专业Agent协同工作
3. **云原生空间数据**：STAC + COG + GeoParquet现代空间数据栈
4. **遥感AI**：语义分割 + 目标检测 + 变化检测多模型集成
5. **自动制图**：AI驱动地图样式自动生成
6. **生产级架构**：Docker部署 + 异步推理 + 监控告警
