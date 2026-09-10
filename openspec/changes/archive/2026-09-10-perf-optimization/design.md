# design：第12月 W1 性能优化

## D1 基准先行（本变更第一原则）

`scripts/bench.py`，4 项测量，**先跑基线落盘 `data/output/bench/baseline.json`，再做任何优化**，最后跑同脚本输出对比表：

| 项 | 测法 | 教学点 |
|---|---|---|
| B1 瓦片冷/热延迟 | 对同一 COG 请求 N 个瓦片 ×2 轮（第一轮冷、第二轮热） | 缓存命中前后的量级差 |
| B2 推理延迟 | temporal_change postclass 在固定小 COG 上跑 1 次（含模型加载与否分开计时） | 推理 vs 加载 vs IO 的占比 |
| B3 模型冷启动 | 进程内首次 get_unet() 耗时 | lazy load 的代价 |
| B4 载荷体积 | /api/reports md 与区界 geojson 的原始 vs gzip 字节数 | 传输层收益上界 |

计时用 `time.perf_counter()`；输出人类可读表格 + JSON 落盘（对比用）。

## D2 推理层（收益不确定 → 全部由 bench 裁决）

- `torch.no_grad() → torch.inference_mode()`（change.py / segment.py）：省 version counter 与 view tracking，是 no_grad 的严格超集升级
- **fp16 实验**：`model.half()` + 输入 half，bench B2 对比 fp32；**负收益或输出数值异常 → 回退并记录数字**（回退是合法结论，不是失败）
- **warmup**：loader 首次 get_unet/get_yolo 后空推理一次（1×4×64×64 随机张量），消 MPS kernel 编译抖动——冷启动移到启动时而非首个用户请求

## D3 服务缓存层（tile_server 磁盘 LRU）

- 目录 `data/cache/tiles/`（gitignore；可整体删除重建）
- 键 = `{md5(abs_cog_path)[:12]}_{mtime_ns}_{z}_{x}_{y}_{bands or auto}`——**mtime_ns 进键**：COG 覆盖后键变 → 自动失效（最简失效解；「同 mtime 覆盖」理论漏网，文档化说明）
- 读：命中 → 直接回 PNG 文件字节；未命中 → rio-tiler 生成 → 写缓存 → 返回
- LRU 上限 512 文件，超限按 atime 删最旧（`os.utime` 维护；简单粗暴够用，不引 cachetools）
- 教学点：**不可变内容寻址是缓存的理想场景**——键里带内容版本（mtime），值永远正确，无需 TTL 猜测

## D4 传输层

- 主服务 `main.py`：`app.add_middleware(GZipMiddleware, minimum_size=1024)`——区界 geojson（492 顶点）、report md/html、SSE 文本全受益
- **gzip × SSE 风险点**：Starlette GZipMiddleware 对 StreamingResponse 逐块压缩不整体缓冲，但事件到达节奏可能变——e2e 回归实测事件完整性兜底
- tile_server `/info`：ETag = `md5(mtime_ns + size)`，请求带 `If-None-Match` 匹配 → `304`（协商缓存教学：max-age 是「别问」，ETag 是「问了但没变就别传」）

## D5 CDN 教学（诚实降级）

不接真 CDN。在 tile_server 注释 + README 讲清：**Cache-Control（强缓存）+ ETag/304（协商缓存）+ 不可变内容寻址（键即版本）= CDN 的全部语义内核**；上 CDN 只是把这些响应头原样透传、把源站换成边缘节点。避免「为了计划名词引入假部署」。

## D6 沙箱与权威数字

WorkBuddy 会话内跑 python 受 sitecustomize 注入干扰（~73ms/open），**bench 的权威数字必须在用户干净终端跑**；agent 侧只做「同环境前后相对对比」（同一干扰前后抵消，比例仍有效）。verify 同理只断言相对改进方向。

## D7 验证矩阵

1. bench 基线/优化后同表对比（B1 热延迟应有量级下降；B2/B3 记录；B4 体积下降）
2. 缓存失效单测：生成 COG → 请求瓦片 → 重写 COG（mtime 变）→ 再请求 → 缓存必须 miss
3. ETag/304：curl 二次请求带 If-None-Match → 304
4. gzip：响应头 `content-encoding: gzip`
5. 回归：e2e 7/7（gzip×SSE）+ 全部既有 selftest
6. verify.py：direct（缓存失效/ETag/gzip 头/bench 对比断言）+ `--integration`（复用 e2e_test.py，同 report-engine 模式）

## D8 回收的旧伏笔

- 第9月 `safe_cog_path` 路径净化 → 缓存键的 abs path md5 同样杜绝 `../` 与别名重复
- 第9月 lazy load 单例 → warmup 是它的自然补全（冷启动从首个用户请求移到启动时）
- W2「数据性质 ≠ 制图语义」的同款思维 → 「缓存命中 ≠ 数据正确」，失效语义是缓存的正确性问题
