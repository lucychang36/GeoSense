# tasks：perf-optimization

## T1 基准工具 + 基线（design D1/D6）
- [ ] `scripts/bench.py`：B1 瓦片冷/热 ×2 轮 / B2 推理延迟 / B3 模型冷启动 / B4 载荷体积；perf_counter 计时；表格 + JSON 落盘 `data/output/bench/`
- [ ] 优化前跑基线落盘 `baseline.json`（agent 侧跑相对基线，权威数字标注「以用户干净终端为准」）

## T2 推理层（design D2）
- [ ] change.py / segment.py：`no_grad` → `inference_mode`
- [ ] fp16 实验：model.half() 分支 + bench B2 对比；负收益/数值异常 → 回退并记录数字
- [ ] loader warmup：首次加载后空推理一次（64×64 随机张量）

## T3 服务缓存层（design D3）
- [ ] tile_server 磁盘缓存：键 = md5(path)[:12]_mtime_ns_z_x_y_bands；读→写→回；LRU 512 文件按 atime 淘汰
- [ ] 失效测试：COG 重写（mtime 变）→ 缓存必须 miss 取新瓦片
- [ ] `data/cache/` 进 gitignore

## T4 传输层 + 回归（design D4/D7）
- [ ] main.py `GZipMiddleware(minimum_size=1024)`；tile_server `/info` ETag/304
- [ ] 实测：gzip 头生效、二次 If-None-Match → 304
- [ ] 回归：e2e 7/7（重点 gzip×SSE 事件完整性）+ 既有 selftest 全过

## T5 收尾
- [ ] bench 优化后对比表（数据进 README/tasks.md）
- [ ] README：第12月 W1 进度行、快速开始、CDN 语义教学段
- [ ] tasks.md 证据回填 + proposal 状态流转；memory；git 提交（代码 + README + openspec）

## 完成后（证据回填）

## 完成后（证据回填）

- **T1 基线（baseline.json）**：B3 冷启动 143.0ms；B2 fp32 first 244.7ms / steady 6.6ms；fp16 first 1223.2ms / steady 5.4ms / agreement 0.9994；B1 raw r1 17.9ms / r2 13.0ms；B4 geojson 413.9KB→81.5KB（×0.197）
- **T2 推理层**：inference_mode ×2（change.py/segment.py）+ loader warmup（64×64 空推理）；**fp16 裁决：负收益回退**——稳态 ~18%（6.6→5.4ms，绝对值 1.2ms）不值首次编译 5×（245→1223ms）+ 精度损失（agreement 0.9994），理由与数字写进 loader.py 注释
- **T3 瓦片缓存**：`scripts/tile_cache.py`（键 md5(路径)[:12]+mtime_ns+z/x/y/波段；原子写 tmp→replace；LRU 512 按 atime；fail-open）；tile_server tile() 改走缓存；**失效实测**：os.utime 改 mtime → 键变（2c858b2aab1a_17883… → _17890…）→ 旧缓存不冒充；after.json **热路径 0.1ms vs raw 13.6ms（136×）**
- **T4 传输层**：main.py GZipMiddleware(minimum_size=1024)；/info ETag=md5(mtime+size)，TestClient 实测 200+etag → If-None-Match 命中 304 → 指纹不符 200；**回归**：4 个 selftest 全过（text_to_map 14/14、report_engine 10、auto_symbology 8/8、auto_cartography 10/10）+ **e2e 7/7**（gzip×SSE 下 7 种事件完整：status/tool×2/report/result×2/answer/done——红旗③解除）
- **T5**：bench 对比表入 README 第12月区块；本变更 proposal → applied
- **踩坑/偏差**：① bench 首跑缺 sys.path 注入 + _tile_coords 里写坏一段表达式（Write 产物自查清理）② 失效测试初版裸文件名传 _resolve_cog 404（它要完整路径）、硬编码瓦片坐标出界（复用 bench._tile_coords）③ fp16 结论与 explore 红旗①预判一致 ④ baseline/after 同环境跑，B3 差异实为 warmup 成本转移（143→318ms 加载 + 首请求 245→120ms，总账基本持平，语义更优）
- **commit hash**：见 git log（本轮提交）
