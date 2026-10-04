"""手工重算混淆矩阵，绕开 ultralytics 的 val pipeline。

为什么要手工算：
    ultralytics 的混淆矩阵和它自己报的 mAP 用了【不同的匹配口径】，
    导致热力图对角线只有 0.30，而 mAP 的 R 是 0.99。为了搞清楚到底谁对，
    这里用完全自己写的逻辑复现一遍，并且【同时用两个 IoU 阈值】对照：

        IoU >= 0.45   （ultralytics 混淆矩阵的默认值）
        IoU >= 0.70   （更严格，接近 mAP50-95 的口径）

    如果 0.45 下对角线也是 0.30、0.70 下反而更高，
    就说明问题出在【匹配阈值太松，相邻关节互相抢框】。
"""
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
VAL_IMG = PROJECT / "data" / "bone_age" / "images" / "val"
VAL_LBL = PROJECT / "data" / "bone_age" / "labels" / "val"
WEIGHTS = PROJECT / "outputs" / "runs" / "smoke" / "weights" / "best.pt"

CLASSES = ["DistalPhalanx", "MCP", "MCPFirst", "MiddlePhalanx",
           "ProximalPhalanx", "Radius", "Ulna"]
N_IMG = 60
CONF = 0.25


def iou_matrix(gt, pr):
    """gt/pr: (N,4) xyxy。返回 (len(gt), len(pr)) 的 IoU 矩阵。"""
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)))
    g = gt[:, None, :]
    p = pr[None, :, :]
    x1 = np.maximum(g[..., 0], p[..., 0])
    y1 = np.maximum(g[..., 1], p[..., 1])
    x2 = np.minimum(g[..., 2], p[..., 2])
    y2 = np.minimum(g[..., 3], p[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    ap = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    union = ag[:, None] + ap[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match(iou, thr):
    """按 IoU 从高到低做一对一匹配，返回 [(gt_idx, pr_idx), ...]。"""
    pairs = []
    used_g, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gi, pi in order:
        if iou[gi, pi] < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((int(gi), int(pi)))
    return pairs


def main():
    imgs = sorted(VAL_IMG.glob("*.png"))[:N_IMG]
    print(f"用 {len(imgs)} 张验证图，conf={CONF}\n")

    model = YOLO(str(WEIGHTS))

    # ---------- 收集所有图的 GT 和预测 ----------
    data = []
    for i, ip in enumerate(imgs, 1):
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

        data.append((ip.stem, gt, pr))
        if i % 20 == 0:
            print(f"  已处理 {i}/{len(imgs)}", flush=True)

    # ---------- 在两个 IoU 阈值下各算一次 ----------
    for thr in (0.45, 0.70):
        n = len(CLASSES)
        mat = np.zeros((n + 1, n + 1))       # 行=预测，列=真实，最后一列/行=background

        for _stem, gt, pr in data:
            if not gt:
                continue
            gt_boxes = np.array([[b[1], b[2], b[3], b[4]] for b in gt])
            pr_boxes = (np.array([[b[1], b[2], b[3], b[4]] for b in pr])
                        if pr else np.zeros((0, 4)))
            iou = iou_matrix(gt_boxes, pr_boxes)
            pairs = greedy_match(iou, thr)

            matched_gt = {gi for gi, _ in pairs}
            for gi, (c, *_rest) in enumerate(gt):
                if gi in matched_gt:
                    pi = dict(pairs)[gi]
                    mat[pr[pi][0], c] += 1
                else:
                    mat[n, c] += 1           # 漏检 -> background

        col = mat.sum(axis=0)
        print(f"\n{'=' * 74}")
        print(f"IoU 阈值 = {thr}   （{len(data)} 张图）")
        print("=" * 74)
        print(f"  {'类别':<18}{'真实框':>8}{'判对':>7}{'对角线':>9}")
        print("  " + "-" * 44)
        diags = []
        for i, cname in enumerate(CLASSES):
            g = int(col[i])
            ok = int(mat[i, i])
            d = ok / g if g else 0
            diags.append(d)
            print(f"  {cname:<18}{g:>8}{ok:>7}{d:>9.2f}")
        print(f"  {'平均':<18}{'':>8}{'':>7}{np.mean(diags):>9.2f}")

    print(f"""
{'=' * 74}
结论怎么看
{'=' * 74}
  如果 IoU=0.70 的对角线明显高于 IoU=0.45，说明：
      模型其实把关节找得挺准，只是【相邻关节的框有重叠】，
      在宽松阈值下会互相抢匹配，于是看起来像"类别判错"。
      这属于【评估口径】造成的假象，不是模型真的分不清。

  如果两个阈值下都只有 0.3 左右，说明：
      模型确实分不清相邻的指骨类别 —— 那是真问题。
""")


if __name__ == "__main__":
    main()
