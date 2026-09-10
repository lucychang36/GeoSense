# proposal：第12月 W1 性能优化（基准先行 + 三层优化）

> 状态：**applied（2026-09-10，bench 前后对比 + 失效/ETag 实测 + 回归全绿，证据见 tasks.md）**

## 背景

学习计划第12月 W1：性能优化（推理加速、缓存、CDN）。现状核实（2026-09-08 explore）：

- 全项目唯一性能设施 = 模型 MPS 单例 lazy load（第9月）+ tile_server 给浏览器的 `Cache-Control: max-age=3600`
- 服务端**零缓存**（无 lru_cache / 磁盘缓存，每瓦片请求都重新 rio-tiler 开窗）
- 推理用 `torch.no_grad()`（非 `inference_mode`）、无 fp16、无 warmup
- 无任何基准工具：e2e 18s / report_tool 30-60s，时间花在哪没人说得清

## 目标（验收标准）

1. **基准先行**：`scripts/bench.py` 可重复运行，输出「基线 vs 优化后」同表对比（瓦片冷/热延迟、推理延迟、模型冷启动、载荷体积）
2. **推理层**：`inference_mode` 落地；MPS fp16 实验有结论（采纳或负收益回退，均记录数字）；模型加载后 warmup 一次
3. **服务缓存层**：tile_server 磁盘缓存，同一瓦片二次请求显著加速（bench 数字为准）；**COG 被覆盖后必须取到新瓦片**（mtime 失效语义测试）
4. **传输层**：`/info` 加 ETag/304；主服务加 gzip（geojson/报告载荷体积可测下降）
5. **回归**：e2e 7/7 不破坏（重点验证 gzip × SSE 流式实时性）；既有 selftest 全过

## 非目标（范围克制）

- Redis / Celery 级分布式缓存（单机学习项目，磁盘 LRU 足够）
- 真 CDN 接入（无线上部署；教「HTTP 缓存语义 = CDN 内核」即可，见 design D5）
- 模型 int8 量化、蒸馏、前端打包优化
- LLM 响应缓存（DeepSeek 调用是延迟大头，但缓存语义复杂且温度=0 下重复请求少，收益不成比例）

## 红旗预登记

① MPS fp16 在 base=16 小 UNet 上可能负收益（kernel 转换开销 > 半精度收益）——bench 裁决，回退也是合法结论；② 瓦片缓存失效语义：mtime 键是最简解，COG「同 mtime 覆盖」理论上可漏（概率极低，文档化说明）；③ gzip × SSE：中间件若缓冲事件会破坏流式实时性，e2e 实测兜底；④ bench 计时受沙箱 PYTHONPATH 干扰，权威数字需在用户干净终端跑，verify 只做同环境相对对比；⑤ 第9月单例化已吃掉最大收益，本月整体收益递减——预期管理。
