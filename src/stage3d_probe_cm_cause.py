"""验证假设：conf=0.001 时，低置信度的"错类框"会抢走匹配。

假设链条
--------
1. ultralytics 的混淆矩阵用的是 `self.args.conf`，而 val 的默认 conf 是 0.001。
2. 混淆矩阵的匹配是【不看类别、只按 IoU 贪心】的（见 metrics.py:454-461）。
3. 而 mAP 的计算是【按类别分开匹配】的。
4. 所以：只要在同一个位置存在一个低置信度但 IoU 略高的错类框，
   它就会在混淆矩阵里"赢"，被记成一次类别混淆；
   但在 mAP 里它根本不影响（mAP 按类别分开算）。

验证方法：数一下同一张图在不同 conf 下输出多少个框。
"""
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
VAL_IMG = PROJECT / "data" / "bone_age" / "images" / "val"
WEIGHTS = PROJECT / "outputs" / "runs" / "full" / "weights" / "best.pt"

model = YOLO(str(WEIGHTS))
imgs = sorted(VAL_IMG.glob("*.png"))[:8]

print("=" * 72)
print("同一批图，不同 conf 下的输出框数")
print("=" * 72)
print(f"  {'conf':>8}{'总框数':>9}{'平均每图':>10}{'真值框数':>10}{'倍数':>8}")
print("  " + "-" * 46)

GT_TOTAL = 0
for ip in imgs:
    lp = PROJECT / "data" / "bone_age" / "labels" / "val" / f"{ip.stem}.txt"
    GT_TOTAL += len(lp.read_text(encoding="utf-8").strip().splitlines())

for conf in (0.001, 0.05, 0.25, 0.5):
    total = 0
    for ip in imgs:
        r = model.predict(str(ip), conf=conf, imgsz=640, device=0,
                          verbose=False)[0]
        total += 0 if r.boxes is None else len(r.boxes)
    print(f"  {conf:>8}{total:>9}{total / len(imgs):>10.1f}"
          f"{GT_TOTAL:>10}{total / max(GT_TOTAL, 1):>7.1f}x")

print("""
解读
----
  如果 conf=0.001 时的框数是真值的十几倍，说明模型在每个关节位置
  同时输出了很多【低置信度的其它类别】的框。
  mAP 按类别分开匹配 -> 不受影响；
  混淆矩阵不分类别、按 IoU 贪心 -> 被这些框抢走匹配，对角线就崩了。
""")

# ---- 直接证据：找一张图，看同一个 GT 上有几个不同类别的框 ----
print("=" * 72)
print("直接证据：某个真实关节位置上，模型输出了几个不同类别的框")
print("=" * 72)

ip = imgs[0]
lp = PROJECT / "data" / "bone_age" / "labels" / "val" / f"{ip.stem}.txt"
W, H = Image.open(ip).size
gt = []
for line in lp.read_text(encoding="utf-8").strip().splitlines():
    c, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    gt.append((int(c), (cx - w / 2) * W, (cy - h / 2) * H,
               (cx + w / 2) * W, (cy + h / 2) * H))

r = model.predict(str(ip), conf=0.001, imgsz=640, device=0, verbose=False)[0]
xyxy = r.boxes.xyxy.cpu().numpy()
cls = r.boxes.cls.cpu().numpy().astype(int)
conf = r.boxes.conf.cpu().numpy()
names = r.names


def iou1(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0


print(f"\n图片 {ip.stem}（真值 {len(gt)} 个框，预测 {len(xyxy)} 个框）")
print(f"\n  取前 6 个真实框，列出所有 IoU>0.45 的预测框：")
for gi, g in enumerate(gt[:6]):
    hits = []
    for pi in range(len(xyxy)):
        v = iou1(g[1:], xyxy[pi])
        if v > 0.45:
            hits.append((v, names[cls[pi]], float(conf[pi]),
                         "✓同类" if cls[pi] == g[0] else "✗错类"))
    hits.sort(reverse=True)
    gt_name = names[g[0]]
    print(f"\n  真值[{gi}] {gt_name}  —— 有 {len(hits)} 个预测框 IoU>0.45")
    for v, n, c, tag in hits[:6]:
        mark = "  ★贪心会选它" if hits and (v, n, c, tag) == hits[0] else ""
        print(f"      IoU={v:.3f}  {n:<18} conf={c:.4f}  {tag}{mark}")
