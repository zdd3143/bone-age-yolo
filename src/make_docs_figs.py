"""把关键结果图压缩后放进 docs/，供 README 引用。

为什么要压缩：
    训练产物里的 PNG 都是高分辨率（1 MB 左右），
    直接放进 git 会让仓库变得很重。压到 200 KB 左右足够 README 展示。
"""
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs"
DOCS.mkdir(exist_ok=True)

ITEMS = [
    # (源文件, 目标文件名, 最大边长, 质量)
    ("outputs/data_report/01_class_dist.png",  "class_dist.png",        1100, 88),
    ("outputs/data_report/04_box_scale.png",   "box_scale.png",         1200, 88),
    ("outputs/runs/full/results.png",          "training_curves.png",   1400, 85),
    ("outputs/runs/full/BoxPR_curve.png",      "pr_curve.png",          1000, 88),
    ("outputs/data_samples/sample_4_2511.png", "sample_annotated.png",   900, 85),
    ("outputs/joint_filter/rus13_14714.png",   "rus13_result.png",      1000, 88),
]

for src_rel, dst_name, max_side, quality in ITEMS:
    src = PROJECT / src_rel
    if not src.exists():
        print(f"  跳过（不存在）：{src_rel}")
        continue
    im = Image.open(src).convert("RGB")     # PNG → 存成 JPEG，体积小很多
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    dst = DOCS / dst_name.replace(".png", ".jpg")
    im.save(dst, "JPEG", quality=quality, optimize=True)
    print(f"  {dst_name.replace('.png', '.jpg'):<24} "
          f"{im.size[0]}x{im.size[1]}  {dst.stat().st_size / 1024:6.0f} KB")

total = sum(f.stat().st_size for f in DOCS.iterdir() if f.is_file())
print(f"\ndocs/ 合计 {total / 1024:.0f} KB")
