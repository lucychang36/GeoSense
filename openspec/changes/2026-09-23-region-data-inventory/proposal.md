# 2026-09-23-region-data-inventory — 区域数据清单与智能兜底（region 开放解析）

> 状态：**applied（2026-09-23，e2e --jinshui 8/8 + 深圳湾 7/7 + 郑州 17/17 / selftest 9/9 + 五套回归全绿 / admin_boundary 全国 3237 条入库 / STAC 双 collection 10 items，证据见 tasks.md）**

## 背景与动机

W2 郑州建筑用地变化报告落地后，用户实测提问「对比郑州金水区 2015 和 2025 的建筑用地变化」，agent 诚实拒绝（正确行为），但拒绝理由出现事实偏差：声称「可用影像仅为深圳湾」——**漏掉了系统自己上周刚跑过的郑州高新区两期影像**。

explore 阶段核对了三层现状：

| 组件 | 现状 | 缺口 |
|---|---|---|
| admin_boundary 表 | 已建、已入 ALLOWED_TABLES、agent 可查 | **只有深圳 10 区**（OSM 种子数据），全国不可用 |
| STAC 目录 | `stac_catalog.py`（第5月 W1）完整实现、`data/stac/` 已落盘 | **collection 硬编码 szbay**，W2 新增的郑州两景不在目录（从未重建）；collection 元数据手写 |
| agent 接入 | `data_retrieval_tool` 只认固定数据集、`spatial_sql_tool` 只查 Doris 三张表 | **没有任何工具可达影像清单**（manifest.json / STAC 均不在 agent 视野内） |

根因：影像元数据活在 manifest.json / STAC 里，agent 的工具层够不着——它只能查库，库里没有影像覆盖信息。

## 方案概述

**区域清单工具（inventory）+ 空间求交解析 + manifest 唯一真相**：

1. **`data_inventory_tool(region, bbox)` 新工具**：任意地名（金水区/杭州/东京）→ 解析 bbox（admin_boundary 精确/模糊匹配优先，未命中则返回候选/要求提供 bbox）→ 与 manifest 影像 bbox **空间求交**（shapely）→ 返回「完全/部分/无覆盖 + 可用日期/云量/波段」。
2. **region 开放语义（本提案核心约束，用户指定）**：`region` 字段不是封闭枚举——**全国乃至全球范围靠空间求交天然支持**，无需逐区域注册 preset；REGION_META 注册表只承担「中文名/描述的显示美化」，未注册区域自动生成元数据（开放-封闭，与 W2 THEMES/SEMANTIC 同构）。
3. **STAC 泛化重建**：`stac_catalog.py` 去硬编码，按 manifest 的 `region` 字段自动分 collection；重建后 szbay + zhengzhou_hightech 双 collection 落盘。**明确 manifest 为唯一真相、STAC 为标准化导出视图**（避免双真相漂移——本次郑州缺目就是漂移的实证）。
4. **admin_boundary 全国化**：下载全国省市两级 GeoJSON 导入（`fetch_admin_boundaries.py`），深圳 10 区保留；世界级**机制支持但本次不导入**（scope 控制）。
5. **agent 拒绝-兜底智能化**：data_node prompt 注入「先查 inventory 再下结论」纪律；找不到数据时能主动说「金水区没有，但郑州高新区有 2023/2025 两期，是否改用？」

## 验收标准

1. `data_inventory_tool("郑州金水区")` 返回：无影像覆盖 + 推荐 `zhengzhou_hightech`（2023-06-26/2025-06-27，云 5.2%/4.4%）+ 附 bbox 相交证据。
2. `data_inventory_tool("杭州")` 返回「无法解析或无数据」的**诚实降级**话术（含可提供 bbox 的指引），不编造。
3. admin_boundary 查询「金水」返回郑州金水区记录（全国省市两级入库，~370 面）。
4. `data/stac/` 重建后含两个 collection，郑州 items 的六波段/日期/bbox 与 manifest 一致（自动校验脚本零 diff）。
5. e2e 新增金水区问句场景：agent 回复含「郑州高新区 + 两期日期」推荐、无幻觉数字；深圳湾回归场景全绿。
6. 全部 selftest 回归通过（inventory / report_engine / thematic 等既有套件）。

## 范围外（non-goals）

- **不做**世界级边界数据导入（机制开放，数据导入留需要出现时再加——第三次重复才抽象纪律）
- **不做**前端可视化接入（区域覆盖卡片/时间轴属增值项，agent 回答质量不依赖它）
- **不做**影像按行政区精确裁剪下载（bbox 近似维持，前案红旗延续）
- **不做** STAC API server / pystac-client 服务化（文件目录够用）
- **不做**数据下载自动化（「无数据区域→自动触发下载」留服务化阶段）

## 红旗预登记

1. **地名歧义**：「朝阳区」存在北京/长春两解——inventory 必须返回候选列表消歧，禁止静默取第一个。
2. **manifest 历史条目 region 缺失**（szbay 老数据无 region 字段）——回落 `szbay` 并在 inventory 输出标注「(legacy)」。
3. **admin_boundary 数据源可用性**：阿里云 DataV GeoAtlas 免费接口偶发限流，下载失败须降级为本地缓存文件（脚本支持 `--from-cache`）。
4. **全国边界 GeoJSON 体积 ~10 MB / 370 面**：内存求交（shapely 逐面 intersects）秒级可行，但边界面读入须缓存为单一 GeoJSON 文件避免每次请求重读下载。
5. **STAC 重建会覆盖现有 data/stac/**：旧目录仅 szbay 单 collection，重建前确认无手工编辑痕迹（diff 校验）。
6. **inventory 与 temporal_change 的区域判定口径必须一致**：都基于 manifest bbox 求交，禁止 inventory 说有、analysis 说没有的第三次「双真相」。
