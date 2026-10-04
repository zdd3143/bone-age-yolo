"""下载 YOLOv8 预训练权重（带重试 + 备用镜像）。

为什么要单独写这个：
    ultralytics 内部用的是 requests 直接下载 GitHub Release，
    在国内网络下经常 RemoteDisconnected。它自己不带重试，
    一失败整个训练就中断了。

    所以这里手动下载到当前目录，ultralytics 看到本地已有同名文件就不会再下载。
"""
import sys
import time
from pathlib import Path

import requests

NAME = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
TAG = "v8.4.0"
DST = Path(__file__).resolve().parent.parent / NAME

URLS = [
    f"https://github.com/ultralytics/assets/releases/download/{TAG}/{NAME}",
    f"https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/{TAG}/{NAME}",
    f"https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/{TAG}/{NAME}",
    f"https://ghfast.top/https://github.com/ultralytics/assets/releases/download/{TAG}/{NAME}",
    f"https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download/{TAG}/{NAME}",
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
}


def try_download(url: str, attempt: int) -> bool:
    print(f"  [{attempt}] {url[:96]}", flush=True)
    try:
        with requests.get(url, stream=True, headers=HEADERS,
                          timeout=(15, 120), allow_redirects=True) as r:
            if r.status_code != 200:
                print(f"        HTTP {r.status_code}")
                return False
            total = int(r.headers.get("content-length", 0))
            tmp = DST.with_suffix(DST.suffix + ".part")
            got = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
            if got < 100_000:
                print(f"        文件太小（{got} 字节），判定失败")
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(DST)
            print(f"        ✓ 下载成功 {got / 1024 / 1024:.1f} MB")
            return True
    except Exception as e:                              # noqa: BLE001
        print(f"        {type(e).__name__}: {str(e)[:80]}")
        return False


def main():
    if DST.exists() and DST.stat().st_size > 100_000:
        print(f"已存在：{DST}（{DST.stat().st_size / 1024 / 1024:.1f} MB）")
        return

    print(f"目标：{DST}")
    print(f"要下载：{NAME}\n")

    for round_no in range(1, 4):
        for url in URLS:
            if try_download(url, round_no):
                print(f"\n✅ 完成：{DST}")
                return
        print(f"\n第 {round_no} 轮全部失败，等 3 秒重试…\n")
        time.sleep(3)

    raise SystemExit(
        "\n❌ 所有镜像都失败。\n"
        "   备用方案：从能上网的机器下载 yolov8n.pt，\n"
        f"   放到 {DST} 即可。\n"
        "   下载地址：https://github.com/ultralytics/assets/releases/"
        f"download/{TAG}/{NAME}"
    )


if __name__ == "__main__":
    main()
