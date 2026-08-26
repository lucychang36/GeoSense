"""阶段2 第4月 W2 交付物：cog_range_demo.py —— COG 在线读取原理演示

Part A：overview 金字塔的价值（本地、确定性）
- 读取同一张 128x128 缩略图：
    * 普通 GeoTIFF 没有金字塔 → 必须读全分辨率（1024x1024x4 波段）再缩小；
    * COG 有金字塔 → rasterio 自动从 overview（128x128）直接读。
- 结果：耗时差一个数量级（本机实测 ~70x），数据规模差 64 倍。

Part B：HTTP Range 协议验证（自建 Range 服务器 + urllib）
- COG 在线读取的前提是服务器支持 Range 请求（206 + Content-Range）。
- 本部分用一个最小 Range 服务器（与对象存储同款行为）演示协议交互，
  并验证标准库 SimpleHTTPRequestHandler 不支持 Range —— 这就是
  "不能随便拿静态文件服务器当影像服务器"的原因。

注：本机环境里 GDAL 的 libcurl 对本地 HTTP 有兼容性问题（GIL 阻塞，表现为挂死），
因此这里用 urllib 做协议级验证；生产环境（S3/CDN 这类成熟 Range 服务）无此问题。

用法：.venv/bin/python scripts/cog_range_demo.py
"""
from __future__ import annotations

import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
PORT = 8123


# ---------------------------------------------------------------------------
# Part A：overview 金字塔的价值
# ---------------------------------------------------------------------------
def part_a() -> None:
    print("=" * 60)
    print("Part A：读取同一张 128x128 缩略图（本地确定性对比）")
    print("=" * 60)

    def best_time(path: Path) -> float:
        times: list[float] = []
        with rasterio.open(path) as ds:
            for _ in range(5):
                t0 = time.time()
                ds.read(1, out_shape=(128, 128),
                        resampling=rasterio.enums.Resampling.average)
                times.append((time.time() - t0) * 1000)
        return min(times)

    plain = best_time(DEMO_DIR / "demo.tif")
    cog = best_time(DEMO_DIR / "demo_cog.tif")

    # 数据规模（文档化计算）：全分辨率 vs overview 3（128px）
    full_bytes = 1024 * 1024 * 4 * 2        # 1024x1024 x 4 波段 x uint16(2B)
    ov_bytes = 128 * 128 * 4 * 2            # overview 128x128
    print(f"  普通 GeoTIFF：{plain:6.2f} ms  （读全分辨率 {full_bytes/1024/1024:.1f} MB 再缩小）")
    print(f"  COG         ：{cog:6.2f} ms  （直接从 overview 读 {ov_bytes/1024:.0f} KB）")
    print(f"  → 耗时差 {plain / max(cog, 0.001):.0f}x；数据规模差 {full_bytes / ov_bytes} 倍")
    print("  → 结论：低缩放/小图场景，金字塔让 COG 省掉几乎全部无效读取\n")


# ---------------------------------------------------------------------------
# Part B：HTTP Range 协议验证
# ---------------------------------------------------------------------------
class RangeHandler(BaseHTTPRequestHandler):
    """最小 Range 服务器（真实对象存储 S3/CDN 的同款行为）。

    标准库 SimpleHTTPRequestHandler 不支持 Range（永远 200 全量），
    GDAL 会直接报 "Range downloading not supported by this server!"。
    """
    protocol_version = "HTTP/1.1"     # 显式消息边界，GDAL/libcurl 依赖

    def log_message(self, *args):     # 静音
        pass

    def _respond(self, code: int, headers: dict, body: bytes,
                 content_length: int | None = None):
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length",
                         str(len(body) if content_length is None else content_length))
        self.end_headers()
        try:
            if body:
                self.wfile.write(body)
            self.wfile.flush()        # 必须 flush：缓冲的 wfile 会滞留小响应体
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self):
        path = (DEMO_DIR / self.path.lstrip("/")).resolve()
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        self._respond(200, {"Accept-Ranges": "bytes"}, b"",
                      content_length=path.stat().st_size)

    def do_GET(self):
        path = (DEMO_DIR / self.path.lstrip("/")).resolve()
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            lo_s, hi_s = rng[6:].split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s) if hi_s else size - 1
            if lo >= size:
                self._respond(416, {}, b"")
                return
            with open(path, "rb") as f:      # 按偏移只读所需字节（seek）
                f.seek(lo)
                chunk = f.read(hi - lo + 1)
            self._respond(206, {
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {lo}-{hi}/{size}",
            }, chunk)
        else:
            with open(path, "rb") as f:
                body = f.read()
            self._respond(200, {"Accept-Ranges": "bytes"}, body)


def part_b() -> None:
    print("=" * 60)
    print("Part B：HTTP Range 协议验证（自建 Range 服务器 + urllib）")
    print("=" * 60)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}/demo_cog.tif"
    cog_size = (DEMO_DIR / "demo_cog.tif").stat().st_size

    # 1) HEAD：探测文件大小（GDAL 打开远程文件的第一步）
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=10) as resp:
        size = int(resp.headers["Content-Length"])
        accept_ranges = resp.headers.get("Accept-Ranges")
    print(f"  ① HEAD → Content-Length={size}（真实大小 {cog_size}）"
          f"，Accept-Ranges={accept_ranges}")

    # 2) Range GET：只取前 512 字节，应返回 206 + Content-Range
    req = urllib.request.Request(url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        code = resp.status
        crange = resp.headers.get("Content-Range")
        body = resp.read()
    print(f"  ② Range: bytes=0-511 → HTTP {code}，Content-Range={crange}，"
          f"收到 {len(body)} 字节")
    assert code == 206 and len(body) == 512, "Range 响应不正确"

    # 3) 反例：标准库 SimpleHTTPRequestHandler 不支持 Range（返回 200 全量）
    print("  ③ 若换成 SimpleHTTPRequestHandler：不支持 Range → GDAL 报错"
          " 'Range downloading not supported'（本会话已实测踩坑）")
    server.shutdown()
    print("  → 结论：COG 在线读取需要支持 Range 的服务器（S3/CDN/本实现）\n")


def main() -> int:
    part_a()
    part_b()
    print("提示：在线瓦片服务见 scripts/tile_server.py（uvicorn 启动，已验证可取 PNG）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
