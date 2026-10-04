"""决定性测试：手工匹配在 conf=0.001 / 0.25 下各跑一遍全量 val，看能不能复现那 0.47。

如果手工匹配在 conf=0.001 下也掉到 0.47，那就完全确认了：
    混淆矩阵低对角线的根因 = val 默认 conf=0.001 + 匹配不分类别
    （低置信度的错类框，只要 IoU 略高，就会把正确框挤掉）
而 mAP 是按类别分开匹配的，所以不受影响。
"""
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
VAL_IMG = PROJECT / "data" / "bone_age" / "images" / "val"
VAL_LBL = PROJECT / "data" / "bone_age" / "labels" / "val"
WEIGHTS = PROJECT / "outputs" / "runs" / "full" / "weights" / "best.pt"

CLASSES = ["DistalPhalanx", "MCP", "MCPFirst", "MiddlePhalanx",
           "ProximalPhalanx", "Radius", "Ulna"]
IOU_THR = 0.45


def iou_matrix(gt, pr):
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)))
    g, p = gt[:, None, :], pr[None, :, :]
    x1 = np.maximum(g[..., 0], p[..., 0]); y1 = np.maximum(g[..., 1], p[..., 1])
    x2 = np.minimum(g[..., 2], p[..., 2]); y2 = np.minimum(g[..., 3], p[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    ap = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    u = ag[:, None] + ap[None, :] - inter
    return np.divide(inter, u, out=np.zeros_like(inter), where=u > 0)


def greedy(iou, thr):
    pairs, ug, up = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gi, pi in order:
        if iou[gi, pi] < thr:
            break
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi)
        pairs.append((int(gi), int(pi)))
    return pairs


model = YOLO(str(WEIGHTS))
imgs = sorted(VAL_IMG.glob("*.png"))

for CONF in (0.001, 0.25):
    n = len(CLASSES)
    mat = np.zeros((n + 1, n + 1))
    n_pred = 0

    for ip in imgs:
        lp = VAL_LBL / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        W, H = Image.open(ip).size

        gt = []
        for line in lp.read_text(encoding="utf-8").strip().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            gt.append((int(c), (cx - w / 2) * W, (cy - h / 2) * H,
                       (cx + w / 2) * W, (cy + h / 2) * H))

        r = model.predict(str(ip), conf=CONF, imgsz=640, device=0,
                          verbose=False)[0]
        pr = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c in zip(xyxy, cls):
                pr.append((int(c), x1, y1, x2, y2))
        n_pred += len(pr)

        gtb = np.array([[b[1], b[2], b[3], b[4]] for b in gt])
        prb = (np.array([[b[1], b[2], b[3], b[4]] for b in pr])
               if pr else np.zeros((0, 4)))
        pairs = greedy(iou_matrix(gtb, prb), IOU_THR)
        pmap = dict(pairs)
        for gi, (c, *_r) in enumerate(gt):
            if gi in pmap:
                mat[pr[pmap[gi]][0], c] += 1
            else:
                mat[n, c] += 1

    col = mat.sum(axis=0)
    print("=" * 74)
    print(f"手工匹配 · conf={CONF} · IoU>={IOU_THR} · {len(imgs)} 张图")
    print("=" * 74)
    print(f"  预测框总数 {n_pred}   平均每图 {n_pred / len(imgs):.1f}   "
          f"真实框总数 {int(col[:n].sum())}")
    print(f"\n  {'类别':<18}{'真实框':>8}{'判对':>7}{'对角线':>9}")
    print("  " + "-" * 44)
    d = []
    for i, cn in enumerate(CLASSES):
        g_, ok = int(col[i]), int(mat[i, i])
        dd = ok / g_ if g_ else 0
        d.append(dd)
        print(f"  {cn:<18}{g_:>8}{ok:>7}{dd:>9.2f}")
    print(f"  {'平均':<18}{'':>8}{'':>7}{np.mean(d):>9.2f}")
    print(f"  background 列（误检数）：{int(mat[:, n].sum())}")
    print()
