# 任务 — 2026-09-23-region-data-inventory

> 约定：每个任务完成后在此回填证据（命令/输出摘要）；selftest 断言一律显式 `bool()` 包裹 `and` 链（项目铁律）。

## T1 STAC 泛化重建（去硬编码 + region 自动分 collection）

- [x] `scripts/stac_catalog.py`：COLLECTION_ID/TITLE/DESCRIPTION 去硬编码 → 按 manifest `region` 字段分组生成 collection；REGION_META 注册表（D1）提供中文名/描述，未注册自动生成；老条目 region 缺失回落 `szbay`（红旗 R2）
- [x] docstring 声明「export view, source of truth = manifest.json」（D2）
- [x] 重建落盘 `data/stac/`：szbay + zhengzhou_hightech 双 collection；校验 items 日期/波段/bbox 与 manifest 逐字段 diff 为零
- [x] 回填证据：`区域分组：{'szbay': 8, 'zhengzhou_hightech': 2}` / `pystac 校验：通过 ✓` / `manifest 一致性 ✓（10/10）`；坑 1：同 bbox 多景触发 STAC spatial extent 重复数组校验失败（去重修复）；坑 2：szbay 4 波段 vs 郑州 6 波段 → summaries 按实际波段并集

## T2 admin_boundary 全国化（下载器 + 缓存 + 导入）

- [x] `scripts/fetch_admin_boundaries.py`：DataV GeoAtlas 省级 + 地市级下载 → `data/admin/*.geojson` 缓存；`--from-cache` 离线重放（红旗 R3）
- [x] `seed_postgis.py` 扩展 `--admin-geojson` 导入：level 枚举 district/city/province（country 预留），UNIQUE(name,level) + ON CONFLICT upsert，不 TRUNCATE
- [x] 实测：`admin_boundary` 查「金水」命中郑州市金水区（level=district）；「朝阳区」返回候选列表（红旗 R1）
- [x] 回填证据：34 省 / 363 市 / **2840 区县**（10.2 MB）；`seed_postgis --admin-geojson` 入库 3237 条；朝阳区歧义对实测 [220104 长春, 110105 北京]；**坑：直辖市（京津冀沪渝）的 110000_full 里区县直接是 district 级，省市两级路由漏掉北京朝阳区 → collect_children 加 side_catch 顺路收层后修复重下**

## T3 区域解析 + inventory 工具（核心）

- [x] `scripts/data_inventory.py`：`resolve_region_bbox` 三级回退（精确/模糊+消歧/词表，D3）+ `inventory_query` 空间求交与 coverage 分档（D4）+ `recommendation` 最近邻推荐 + `manifest_mtime` 附带（D2）
- [x] `backend/agent/langchain_tools.py` 注册 `data_inventory_tool(region, bbox)`（bbox 优先）
- [x] selftest：full/partial/none/歧义候选/未解析五分支 + 金水区真实查询 + 杭州诚实降级，全 `bool()` 断言
- [x] 回填证据：selftest **9/9 PASS**；**设计修订（D4 实测驱动）**：纯面积阈值无法区分深圳湾自定义框（0.336）与金水区相交（0.337）→ 改为 **region_known 身份判定**（数据自属区域 partial 不拒绝，非自属区域 partial 警告+推荐替代）；坑：scenes 输出投影漏 path 字段致 data_node KeyError（补上，LLM 亦可直读文件名）

## T4 agent 接入（prompt 纪律 + 路由改造）

- [x] data_node 工具集挂 `data_inventory_tool`；SYSTEM 注入「区域问句先探查再下结论」纪律（D7）
- [x] `REGION_KEYWORDS` 降级为回退路径：inventory 求交优先，返回 `region` 字段喂 analysis 节点
- [x] 拒绝话术模板落地（推荐替代区域 + 日期清单 + 询问切换）
- [x] 回填证据：data_node 四问句实测——金水区→gap+推荐（34% 覆盖）/深圳湾→szbay_real 2023+2025/郑州高新区→zhengzhou_real 2023+2025/**杭州→gap 诚实降级（旧代码静默回落深圳湾的坑被堵死）**；supervisor 缺口话术含「数据情况/你可以」两段；report_worker data_gap 短路实测

## T5 e2e 与回归

- [x] `scripts/e2e_test.py` 新增 `--jinshui` 场景：回复含「高新区/郑州高新区」+「2023」「2025」、推荐句无幻觉面积数字；深圳湾回归场景全绿
- [x] 全部既有 selftest 回归通过（thematic / report_engine / text_to_map / symbology / cartography / inventory）
- [x] README 进度同步（第12月 W2 变更三条目）
- [x] 回填证据：e2e **--jinshui 8/8**（data_inventory_tool 在场 / 诚实说明 2015 无数据 / 无幻觉 km 数字 / 无报告事件）+ 深圳湾 **7/7** + 郑州 **17/17**；selftest 回归 thematic 29 / report 22 / t2m 14 / symbology 8 / cartography 10 / inventory 9 全绿；**端口预检再次生效**（用户残留 uvicorn PID 52375 占 8010 → 先 kill 再跑）

## 影响面（供 apply 前检查）

`scripts/stac_catalog.py` / `scripts/fetch_admin_boundaries.py`（新） / `scripts/data_inventory.py`（新） / `scripts/seed_postgis.py` / `backend/agent/langchain_tools.py` / `backend/agent/prompts.py` / `backend/agent/graph.py` / `backend/agent/multi_agent/agents.py` / `backend/agent/multi_agent/state.py` / `scripts/e2e_test.py` / `README.md` / `data/stac/**`（重建） / `data/admin/**`（新缓存）

