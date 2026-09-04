#!/usr/bin/env python3
"""第10月 W2：自动符号化引擎 —— 数据类型 → 配色方案（ColorBrewer 规则，纯函数零 LLM）

为什么是它
----------
W1 复盘红旗③「制图靠人肉」：项目内 4+ 处硬编码配色（蓝藻 6 处 RGBA /
多 Agent 变化纯红 / 光谱差分 cmap=gray / render_mask 默认采样），
颜色从哪来没有决策记录。本引擎把「数据性质 → 配色方案」做成显式规则层：

    探查(profile_data) → 决策(choose_symbology) → 渲染(to_matplotlib) / 前端(to_mapbox_style)

知识核心：ColorBrewer 三类 palette 与数据类型的对应
  categorical（定性分类）→ Qualitative + 语义约定覆盖（水=蓝 / 植=绿 / 城=灰）
  sequential（单极连续） → Sequential（浅→深 = 低→高）
  diverging（双极连续）  → Diverging（中点=无变化，两端=正负方向）
  binary（二值事件）     → 透明背景 + 事件色

语义约定层 = 领域知识显式注入：命中行业惯例色（如水体必蓝）优先于机械 palette，
未命中回落机械色并如实标注。刻意零 LLM —— ColorBrewer 是有限规则表，
规则可复现、可自测；LLM 选色是 W4（Text-to-Map）的范畴。

实现裁决（design D3/D5 内部矛盾的最小裁定）：sequential 触发词只认「植被」，
NDVI 落缺省 YlGnBu —— 与 D5 demo 预期和 selftest 断言一致，且三联色族区分更明显。

用法（项目 .venv，WorkBuddy 会话内加 env -u PYTHONPATH）：
  .venv/bin/python scripts/auto_symbology.py --selftest   # 决策表自测（秒级，零数据依赖）
  .venv/bin/python scripts/auto_symbology.py              # 真实数据三联 demo（~10s）
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rich.console import Console  # noqa: E402

con = Console()
OUT_DIR = PROJECT_ROOT / "data" / "output"
COGS_DIR = PROJECT_ROOT / "data" / "cogs"
UNET_PT = PROJECT_ROOT / "data" / "models" / "unet_finetuned.pt"
PAIR = ("szbay_real_20230708.tif", "szbay_real_20250727.tif")

# ---------------------------------------------------------------------------
# ColorBrewer 内置 palette（硬编码 hex，零第三方依赖）
# ---------------------------------------------------------------------------
PALETTES: dict[str, list[str]] = {
    "Set2":    ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "Dark2":   ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"],
    "YlGnBu":  ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#0C2A84"],
    "Greens":  ["#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476", "#41AB5D", "#238B45", "#006D2C"],
    "OrRd":    ["#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84", "#FC8D59", "#EF6548", "#D7301F", "#B30000"],
    "RdBu":    ["#67001F", "#B2182B", "#D6604D", "#F4A582", "#F7F7F7", "#92C5DE", "#4393C3", "#2166AC", "#053061"],
    "BrBG":    ["#543005", "#8C510A", "#BF812D", "#DFC27D", "#F5F5F5", "#80CDC1", "#35978F", "#01665E", "#003C30"],
    "cividis": ["#00224E", "#123570", "#3B496C", "#575D6D", "#707173", "#8A8678", "#A59C74", "#C3B369", "#E1CF62", "#FEE838"],
}
# 色盲安全白名单（ColorBrewer 官方标注；定性 Set2/Dark2 无 CB-safe 认证 → 诚实标 False）
CB_SAFE = {"YlGnBu", "OrRd", "RdBu", "BrBG", "Greens", "cividis"}

# 语义约定表（GIS 行业惯例显式化；命中覆盖机械 palette）
SEMANTIC: dict[str, str] = {
    "水": "#3787C0", "水体": "#3787C0",
    "植被": "#4C9F38", "绿植": "#4C9F38",
    "城市": "#8C8C8C", "建成区": "#8C8C8C",
    "新增": "#E24B4A", "变化": "#E24B4A",
    "消退": "#3787C0", "持续": "#E6B93C",
}


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class DataProfile:
    """一次数据探查的结果：符号化决策的输入。"""
    kind: str            # "categorical" | "sequential" | "diverging" | "binary"
    n_classes: int       # categorical: unique 数；其它 0
    vmin: float
    vmax: float
    labels: list[str] = field(default_factory=list)   # 类别名 / 连续数据的语义提示
    class_values: list = field(default_factory=list)  # categorical: 原始类别值（mapbox match 用）


@dataclass
class SymbologyPlan:
    """一次符号化决策的结果：渲染参数 + 可审计的理由。"""
    kind: str
    palette_name: str
    colors: list[str]        # hex（#RRGGBB / #RRGGBBAA 透明）
    colorblind_safe: bool
    reason: str              # 一句话决策理由（日志 / LLM 消费用）
    semantic_hits: dict = field(default_factory=dict)  # 审计：哪些类被语义覆盖
    class_values: list = field(default_factory=list)
    vmin: float = 0.0        # 连续值域（to_mapbox_style breaks 缺省用）
    vmax: float = 1.0


# ---------------------------------------------------------------------------
# 探查：数据 → DataProfile
# ---------------------------------------------------------------------------
def profile_data(values, labels=None, force_kind=None) -> DataProfile:
    """判定数据类型（优先级：force_kind → bool→binary → 整数 unique≤10→categorical
    → float 跨 0→diverging → sequential）。NaN 用 nanmin/nanmax 忽略。"""
    arr = np.asarray(values)
    lab = list(labels) if labels else []
    if force_kind:
        kind = force_kind
    elif arr.dtype == bool:
        kind = "binary"
    elif np.issubdtype(arr.dtype, np.integer) and np.unique(arr).size <= 10:
        kind = "categorical"
    else:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        kind = "diverging" if vmin < 0 < vmax else "sequential"

    if kind == "categorical":
        uniq = np.unique(arr)
        return DataProfile(kind, int(uniq.size), float(uniq.min()), float(uniq.max()),
                           lab, [v.item() if hasattr(v, "item") else v for v in uniq])
    return DataProfile(kind, 0, float(np.nanmin(arr)), float(np.nanmax(arr)), lab, [])


# ---------------------------------------------------------------------------
# 决策：DataProfile → SymbologyPlan
# ---------------------------------------------------------------------------
def choose_symbology(profile: DataProfile) -> SymbologyPlan:
    kind = profile.kind
    if kind == "binary":
        return SymbologyPlan(
            kind, "semantic(binary)", ["#00000000", SEMANTIC["变化"]], False,
            "二值事件：透明背景 + 语义事件色（变化=红）；透明由消费方按需使用",
            {}, profile.class_values, profile.vmin, profile.vmax)

    if kind == "categorical":
        colors: list[str | None] = []
        hits: dict[str, str] = {}
        used_sem = False
        set2 = iter(PALETTES["Set2"])
        for i in range(profile.n_classes):
            lab = profile.labels[i] if i < len(profile.labels) else None
            if lab and lab in SEMANTIC:
                hits[lab] = SEMANTIC[lab]
                colors.append(SEMANTIC[lab])
                used_sem = True
            else:
                colors.append(next(set2))
        name = "semantic" if used_sem and len(hits) == profile.n_classes else (
            "Set2+semantic" if used_sem else "Set2")
        reason = (f"定性分类 {profile.n_classes} 类：语义命中 {list(hits) or '无'}，"
                  f"未命中回落 Set2 机械色" if used_sem else
                  f"定性分类 {profile.n_classes} 类（无语义标签）：Set2 机械色")
        return SymbologyPlan(kind, name, colors, False, reason, hits, profile.class_values, profile.vmin, profile.vmax)

    if kind == "diverging":
        return SymbologyPlan(
            kind, "RdBu", list(PALETTES["RdBu"]), True,
            f"连续值域 [{profile.vmin:.3g}, {profile.vmax:.3g}] 跨 0：中点=无变化，两端=正负方向",
            {}, [], profile.vmin, profile.vmax)

    # sequential：按语义提示选族（触发词只认「植被」——见 docstring 实现裁决）
    text = "".join(profile.labels)
    if "植被" in text:
        name = "Greens"
    elif any(k in text for k in ("风险", "密度", "热度")):
        name = "OrRd"
    else:
        name = "YlGnBu"
    return SymbologyPlan(
        kind, name, list(PALETTES[name]), True,
        f"连续单极 [{profile.vmin:.3g}, {profile.vmax:.3g}]：浅→深=低→高（{name}）",
        {}, [], profile.vmin, profile.vmax)


# ---------------------------------------------------------------------------
# 渲染：SymbologyPlan → matplotlib / Mapbox Style 骨架
# ---------------------------------------------------------------------------
def to_matplotlib(plan: SymbologyPlan, n: int | None = None):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    if plan.kind in ("categorical", "binary"):
        cols = plan.colors[:n] if n else plan.colors
        return ListedColormap(cols)
    return LinearSegmentedColormap.from_list(plan.palette_name, plan.colors)


def _css_color(hexcolor: str) -> str:
    """#RRGGBBAA → rgba()（Mapbox fill-color 语法）；hex6 原样。"""
    if len(hexcolor) == 9:
        r, g, b = int(hexcolor[1:3], 16), int(hexcolor[3:5], 16), int(hexcolor[5:7], 16)
        return f"rgba({r},{g},{b},{int(hexcolor[7:9], 16) / 255:.3f})"
    return hexcolor


def to_mapbox_style(plan: SymbologyPlan, field: str, breaks=None) -> dict:
    """Mapbox Style JSON 骨架（仅 fill-color 表达式；source/layers 归 W4）。
    categorical/binary → match 表达式；连续 → interpolate 表达式。"""
    if plan.kind in ("categorical", "binary"):
        expr = ["match", ["get", field]]
        for v, c in zip(plan.class_values, plan.colors):
            expr += [v, _css_color(c)]
        expr.append("#CCCCCC")
        return {"fill-color": expr}
    if breaks is None:
        breaks = list(np.linspace(plan.vmin, plan.vmax, 5))
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    cm = LinearSegmentedColormap.from_list("_tmp", plan.colors)
    cols = [to_hex(cm(float(x))) for x in np.linspace(0, 1, len(breaks))]
    expr = ["interpolate", ["linear"], ["get", field]]
    for b, c in zip(breaks, cols):
        expr += [round(float(b), 4), c]
    return {"fill-color": expr}


# ---------------------------------------------------------------------------
# 自测：决策表 8 条断言（design D7，零数据依赖）
# ---------------------------------------------------------------------------
def selftest() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    checks: list[tuple[str, bool, str]] = []

    p = profile_data(np.array([[True, False]]))
    s = choose_symbology(p)
    checks.append(("bool → binary", p.kind == "binary" and s.kind == "binary", s.palette_name))

    p = profile_data(np.array([[0, 1, 2]], dtype=np.uint8), labels=["水", "城", "植"])
    s = choose_symbology(p)
    r, g, b = mcolors.to_rgb(s.colors[0])
    checks.append(("语义命中：水 → 蓝域（B 通道最大）",
                   bool(s.semantic_hits.get("水")) and b > r and b > g, str(s.semantic_hits)))

    p = profile_data(np.array([[0, 1, 2]], dtype=np.uint8))
    s = choose_symbology(p)
    checks.append(("无标签 3 类 → Set2 机械色", s.palette_name == "Set2", s.palette_name))

    p = profile_data(np.linspace(-1, 1, 50))
    s = choose_symbology(p)
    checks.append(("值域 [-1,1] 跨 0 → diverging RdBu", s.palette_name == "RdBu", s.palette_name))

    p = profile_data(np.linspace(0, 0.8, 50))
    s = choose_symbology(p)
    checks.append(("值域 [0,0.8] 单极 → sequential YlGnBu", s.palette_name == "YlGnBu", s.palette_name))

    p = profile_data(np.linspace(0, 0.8, 50), force_kind="diverging")
    s = choose_symbology(p)
    checks.append(("force_kind=diverging 覆盖 sequential", s.palette_name == "RdBu", s.palette_name))

    s_c = choose_symbology(profile_data(np.array([[0, 1, 2]], dtype=np.uint8), labels=["水", "城", "植"]))
    st1 = to_mapbox_style(s_c, "class")
    colors_ok = all(c.startswith("#") or c.startswith("rgba") for c in st1["fill-color"][3::2])
    st2 = to_mapbox_style(choose_symbology(profile_data(np.linspace(-1, 1, 50))), "value")
    checks.append(("mapbox match/interpolate 结构合法",
                   st1["fill-color"][0] == "match" and colors_ok and st2["fill-color"][0] == "interpolate",
                   f"match={st1['fill-color'][0]}, interpolate={st2['fill-color'][0]}"))

    s_rdbu = choose_symbology(profile_data(np.linspace(-1, 1, 10)))
    s_set2 = choose_symbology(profile_data(np.array([[0, 1, 2]], dtype=np.uint8)))
    checks.append(("CB-safe 白名单标注一致", s_rdbu.colorblind_safe and not s_set2.colorblind_safe,
                   f"RdBu={s_rdbu.colorblind_safe}, Set2={s_set2.colorblind_safe}"))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        con.print(f"  [{'green]PASS' if ok else 'red]FAIL'}[/] {name}  [dim]{detail}[/]")
    con.print(f"\n[bold]自测：{n_pass}/{len(checks)} PASS[/]")
    return 0 if n_pass == len(checks) else 1


# ---------------------------------------------------------------------------
# demo：真实数据三联（sequential / categorical+语义 / diverging）
# ---------------------------------------------------------------------------
def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False   # 中文字体缺 U+2212 负号字形
    import matplotlib.pyplot as plt
    import rasterio
    import torch

    from unet_segmentation import N_CLASSES, UNet, infer_full

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for name in PAIR:
        if not (COGS_DIR / name).exists():
            con.print(f"[red]COG 不存在：{COGS_DIR / name}[/]")
            return 1

    def read_ndvi(path):
        with rasterio.open(path) as ds:
            b = ds.read().astype(np.float32) / 10000.0
        return b, (b[3] - b[2]) / (b[3] + b[2] + 1e-6)

    # 联1：NDVI（制图语义 = 单极植被活性 → force sequential）
    # 自动判定会得 diverging（水面 NDVI 为负，值域跨 0）——但那是「水的物理性质」
    # 不是「反向植被活性」。数据性质 ≠ 制图语义，此处用 force_kind 注入领域知识（诚实记录）。
    b25, ndvi25 = read_ndvi(COGS_DIR / PAIR[1])
    ndvi25[~(b25.sum(0) > 0.02)] = np.nan
    p1 = profile_data(ndvi25, labels=["NDVI"], force_kind="sequential")
    s1 = choose_symbology(p1)

    # 联2：U-Net 分类（categorical + 语义命中；labels 用语义表全称）
    model = UNet(in_ch=4, out_ch=N_CLASSES, base=16).to(device)
    model.load_state_dict(torch.load(UNET_PT, map_location=device))
    model.eval()
    with torch.no_grad():
        mask = infer_full(model, torch.from_numpy(b25[None]).to(device), device)
    p2 = profile_data(mask, labels=["水", "城市", "植被"])
    s2 = choose_symbology(p2)

    # 联3：NDVI 差值 2025−2023（无云交集，交集外 NaN → diverging）
    b23, ndvi23 = read_ndvi(COGS_DIR / PAIR[0])
    ndvi23[~(b23.sum(0) > 0.02)] = np.nan
    common = (b25.sum(0) > 0.02) & (b23.sum(0) > 0.02) & \
             ~(b25.min(0) > 0.25) & ~(b23.min(0) > 0.25)
    diff = ndvi25 - ndvi23
    diff[~common] = np.nan
    p3 = profile_data(diff)
    s3 = choose_symbology(p3)

    con.print(f"[bold]三联决策[/]")
    for i, s in enumerate((s1, s2, s3), 1):
        con.print(f"  联{i}: {s.kind:12s} → [bold]{s.palette_name}[/]  {s.reason}")
    con.print(f"  联2 语义命中：{s2.semantic_hits}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].imshow(ndvi25, cmap=to_matplotlib(s1), vmin=p1.vmin, vmax=p1.vmax)
    axes[0].set_title(f"联1 NDVI ({PAIR[1][10:18]})\n{s1.palette_name} — {s1.kind}", fontsize=10)
    axes[1].imshow(mask, cmap=to_matplotlib(s2), vmin=0, vmax=p2.n_classes - 1, interpolation="nearest")
    axes[1].set_title(f"联2 U-Net 分类（语义覆盖）\n{s2.palette_name} — 水/城/植", fontsize=10)
    vmax3 = float(np.nanmax(np.abs(diff)))
    cm3 = to_matplotlib(s3)
    cm3.set_bad("#DDDDDD")
    axes[2].imshow(diff, cmap=cm3, vmin=-vmax3, vmax=vmax3)
    axes[2].set_title(f"联3 NDVI 差值 2025-2023\n{s3.palette_name} — 红=变褐 蓝=变绿", fontsize=10)
    for ax in axes:
        ax.axis("off")
    for i, s in enumerate((s1, s2, s3)):
        fig.text(0.06 + i * 0.32, 0.045, f"决策理由：{s.reason}", fontsize=8, wrap=True)
    fig.suptitle("GeoSense 自动符号化引擎：数据类型 → 配色方案（ColorBrewer 规则）", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "symbology_demo.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    con.print(f"\n[green]三联图已保存：{out}[/]")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="决策表自测（零数据依赖）")
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else main())
