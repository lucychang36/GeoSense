"""第5月 W4 交付物：data_pipeline.py —— STAC 检索 → COG 读取 → NDVI 分析（端到端）

核心概念：数据流水线（pipeline）
- 把"找数据 → 读数据 → 算指标 → 出结果"四步串成一条自动链路，
  是阶段2（空间数据基础设施）的验收之作：从"人翻文件"升级为"程序按条件取数"。
- 每一步都复用前几周的成果：
    ① STAC 检索  —— pystac-client 查 STAC API（第5月 W2，:8002）
    ② COG 读取   —— rio-tiler 读 COG 资产（第4月 W2，asset 由 ① 返回）
    ③ 指标计算   —— NDVI（植被）/ NDWI（水体），B4/B8 波段
    ④ 结果输出   —— 统计 JSON + 色带图 PNG

核心概念：资产引用（asset href）解析
- STAC Item 的 assets["cog"].href 是相对路径（W1 落盘时是相对项目根的）；
- 本地演示直接按项目根 resolve；生产环境这里会是 https://... 的完整 URL，
  但流水线代码不变 —— 这正是"元数据与数据分离"的好处。

运行方式：
  .venv/bin/python scripts/data_pipeline.py   （需 STAC API :8002 在跑）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pystac import Item
from pystac_client import Client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.core.config import PROJECT_ROOT

console = Console()
STAC_URL = "http://127.0.0.1:8002"
COLLECTION = "sentinel2-szbay"
# 合成影像波段顺序（manifest）：B2=1, B3=2, B4=3, B8=4
B4_IDX, B8_IDX, B3_IDX = 3, 4, 2  # rio-tiler read 用 1-based


# ---------------------------------------------------------------------------
# 步骤 1：STAC 检索
# ---------------------------------------------------------------------------
def step1_search(datetime_range: str, bbox: list[float], cloud_max: float) -> list[Item]:
    """按 时间 + 空间 + 云量 检索影像，返回 STAC Item 列表。"""
    catalog = Client.open(STAC_URL)
    results = catalog.search(
        collections=[COLLECTION],
        datetime=datetime_range,
        bbox=bbox,
    )
    items = [it for it in results.items() if it.properties.get("cloud_cover", 0) <= cloud_max]
    console.print(f"[dim]  检索 {datetime_range} 命中 {len(items)} 景（云量 ≤ {cloud_max}%）[/]")
    return items


# ---------------------------------------------------------------------------
# 步骤 2：COG 读取（rio-tiler 只读需要的波段）
# ---------------------------------------------------------------------------
def step2_read_cog(item: Item) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """读 Item 的 COG 资产，返回 (B4, B8, B3, 元信息)。"""
    from rio_tiler.io import COGReader

    href = item.assets["cog"].href
    path = (PROJECT_ROOT / href.lstrip("./")).resolve()  # 相对项目根解析
    with COGReader(str(path)) as cog:
        arr = cog.read([B4_IDX, B8_IDX, B3_IDX]).data  # (3, h, w)：B4, B8, B3（rio-tiler 6.x 用 .data）
        meta = {"scene": item.id, "path": str(path.relative_to(PROJECT_ROOT)),
                "size": f"{cog.width}x{cog.height}", "dtype": cog.dataset.dtypes[0]}
    return arr[0], arr[1], arr[2], meta


# ---------------------------------------------------------------------------
# 步骤 3：NDVI / NDWI 计算与统计
# ---------------------------------------------------------------------------
def step3_analyze(b4: np.ndarray, b8: np.ndarray, b3: np.ndarray | None = None) -> dict:
    """NDVI=(B8-B4)/(B8+B4)；NDWI=(B3-B8)/(B3+B8)（McFeeters，正值为水体）。"""
    b4f, b8f = b4.astype(np.float32), b8.astype(np.float32)
    ndvi = np.divide(b8f - b4f, b8f + b4f, out=np.zeros_like(b4f), where=(b8f + b4f) != 0)
    stats = {
        "ndvi_mean": round(float(ndvi.mean()), 4),
        "ndvi_p10": round(float(np.percentile(ndvi, 10)), 4),
        "ndvi_p90": round(float(np.percentile(ndvi, 90)), 4),
        "vegetation_ratio": round(float((ndvi > 0.3).mean()), 4),   # 浓植被占比
    }
    if b3 is not None:
        b3f = b3.astype(np.float32)
        ndwi = np.divide(b3f - b8f, b3f + b8f, out=np.zeros_like(b3f), where=(b3f + b8f) != 0)
        stats["ndwi_mean"] = round(float(ndwi.mean()), 4)
        stats["water_ratio"] = round(float((ndwi > 0).mean()), 4)   # 水体占比
    return stats


# ---------------------------------------------------------------------------
# 步骤 4：结果输出（统计 JSON + 色带图）
# ---------------------------------------------------------------------------
def step4_output(ndvi: np.ndarray, meta: dict, stats: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"ndvi_{meta['scene']}.png"

    # 遥感标准做法：合成数据逐像素噪声在除法里被放大（B4+B8≈0 → 极端值），
    # 出图前做轻度高斯平滑（sigma=2 ≈ 3×3 像素窗口），消除"马赛克"感；
    # 统计仍用原始 NDVI，保证指标不被平滑改变。
    from scipy.ndimage import gaussian_filter
    ndvi_smooth = gaussian_filter(ndvi, sigma=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(ndvi_smooth, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
    ax.set_title(f"NDVI · {meta['scene']}\nmean={stats['ndvi_mean']} · vegetation {stats['vegetation_ratio']:.0%}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png


def main() -> None:
    console.rule("[bold cyan]GeoSense 数据流水线：STAC → COG → NDVI")
    # 检索条件：2020~2022 年、深圳湾、云量 ≤ 5%
    items = step1_search("2020-01-01T00:00:00Z/2022-12-31T23:59:59Z",
                         [113.88, 22.46, 114.10, 22.60], cloud_max=5)

    if not items:
        console.print("[red]无命中影像，检查 STAC API 或检索条件[/]")
        return

    table = Table(title="流水线逐景结果")
    table.add_column("场景")
    table.add_column("影像")
    table.add_column("NDVI均值", justify="right")
    table.add_column("植被占比", justify="right")
    table.add_column("NDWI均值", justify="right")
    table.add_column("水体占比", justify="right")
    table.add_column("出图")

    for item in items:
        b4, b8, b3, meta = step2_read_cog(item)
        stats = step3_analyze(b4, b8, b3)
        ndvi = np.divide(b8.astype(np.float32) - b4.astype(np.float32),
                         b8.astype(np.float32) + b4.astype(np.float32),
                         out=np.zeros_like(b4, dtype=np.float32),
                         where=(b8.astype(np.float32) + b4.astype(np.float32)) != 0)
        png = step4_output(ndvi, meta, stats)
        table.add_row(meta["scene"], f"{meta['size']}·{meta['dtype']}",
                      f"{stats['ndvi_mean']}", f"{stats['vegetation_ratio']:.0%}",
                      f"{stats['ndwi_mean']}", f"{stats['water_ratio']:.0%}",
                      png.name)

    console.print(table)
    console.print(Panel(
        "流水线闭环 ✓：STAC 检索（按时间/空间/云量）→ 资产解析 → rio-tiler 读 COG → "
        "NDVI/NDWI 统计 → 色带图落盘 data/output/",
        title="端到端结果", border_style="green"))


if __name__ == "__main__":
    main()
