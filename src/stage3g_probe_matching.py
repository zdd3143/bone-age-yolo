"""复现实验：用 ultralytics 完全相同的匹配代码，看能不能复现它的对角线。

背景
----
    已确认的事实：
        ultralytics val 内置混淆矩阵  conf=0.001 → 对角线 0.53
                                    conf=0.25  → 对角线 1.00
        我自己写的贪心匹配          conf=0.001 → 对角线 0.97   ← 对不上
                                    conf=0.25  → 对角线 1.00

    同样是 conf=0.001，为什么 ultralytics 给 0.53 而我给 0.97？

    怀疑点：匹配在【IoU 并列】时的排序方式不同。

        低 conf 下，multi_label=True 会产生「同一个框、不同类别」的多条检测
        —— 它们的 IoU 和同一个 GT 完全相同（因为框几乎一模一样）。
        这时"谁赢"完全取决于排序实现：
            ultralytics:  matches[:,2].argsort()[::-1]   （升序后反转）
            我的实现:      np.argsort(-iou)               （直接降序）
        两者在并列时的顺序是【相反】的。

    本脚本：把 ultralytics 的匹配代码【一字不改】抄过来，喂同样的检测结果，
    看能否复现 0.53；再用我自己的贪心跑一遍，看是否得到 0.97。
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

PROJECT = Path(__file__).resolve().parent.parent
VAL_IMG = PROJECT / "data" / "bone_age" / "images" / "val"
VAL_LBL = PROJECT / "data" / "bone_age" / "labels" / "val"
WEIGHTS = PROJECT / "outputs" / "runs" / "full" / "weights" / "best.pt"

CLASSES = ["DistalPhalanx", "MCP", "MCPFirst", "MiddlePhalanx",
           "ProximalPhalanx", "Radius", "Ulna"]
IOU_THR = 0.45


# ============================================================
# 方案 A：ultralytics 的匹配代码（从 metrics.py:454-486 抄过来的）
# ============================================================
def match_ultralytics(gt_cls, gt_boxes, det_cls, det_boxes, nc):
    """返回 (矩阵, 预测框数)。逻辑和 ConfusionMatrix.process_batch 一致。"""
    mat = np.zeros((nc + 1, nc + 1))
    gt_classes = [int(c) for c in gt_cls]
    det_classes = [int(c) for c in det_cls]

    if len(gt_classes) == 0:
        for dc in det_classes:
            mat[dc, nc] += 1
        return mat
    if len(det_classes) == 0:
        for gc in gt_classes:
            mat[nc, gc] += 1
        return mat

    iou = box_iou(torch.as_tensor(gt_boxes), torch.as_tensor(det_boxes))

    x = torch.where(iou > IOU_THR)
    if x[0].shape[0]:
        m = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
    else:
        m = np.zeros((0, 3))

    m0, m1, _ = m.transpose().astype(int)
    gt_match = np.full(len(gt_classes), -1)
    gt_match[m0] = m1
    matched_det = set(m1.tolist())
    for i, gc in enumerate(gt_classes):
        if (di := gt_match[i].item()) >= 0:
            mat[det_classes[di], gc] += 1
        else:
            mat[nc, gc] += 1
    for i, dc in enumerate(det_classes):
        if i not in matched_det:
            mat[dc, nc] += 1
    return mat


# ============================================================
# 方案 B：我自己的贪心匹配（stage3e 用的）
# ============================================================
def match_mine(gt_cls, gt_boxes, det_cls, det_boxes, nc):
    mat = np.zeros((nc + 1, nc + 1))
    if len(gt_cls) == 0:
        for dc in det_cls:
            mat[dc, nc] += 1
        return mat
    if len(det_cls) == 0:
        for gc in gt_cls:
            mat[nc, gc] += 1
        return mat

    iou = box_iou(torch.as_tensor(gt_boxes), torch.as_tensor(det_boxes)).numpy()
    pairs, ug, up = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gi, pi in order:
        if iou[gi, pi] < IOU_THR:
            break
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi)
        pairs.append((int(gi), int(pi)))
    pmap = dict(pairs)
    for gi, gc in enumerate(gt_cls):
        if gi in pmap:
            mat[det_cls[pmap[gi]], gc] += 1
        else:
            mat[nc, gc] += 1
    for pi, dc in enumerate(det_cls):
        if pi not in up:
            mat[dc, nc] += 1
    return mat


def diagonal(mat, nc):
    col = mat.sum(axis=0)
    d = []
    for i in range(nc):
        g = col[i]
        d.append(mat[i, i] / g if g else 0)
    return np.mean(d), col


def main():
    model = YOLO(str(WEIGHTS))
    imgs = sorted(VAL_IMG.glob("*.png"))
    nc = len(CLASSES)

    for conf in (0.001, 0.25):
        mat_u = np.zeros((nc + 1, nc + 1))
        mat_m = np.zeros((nc + 1, nc + 1))
        n_det = 0

        for ip in imgs:
            lp = VAL_LBL / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            W, H = Image.open(ip).size
            gt_cls, gt_boxes = [], []
            for line in lp.read_text(encoding="utf-8").strip().splitlines():
                c, cx, cy, w, h = line.split()
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                gt_cls.append(int(c))
                gt_boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                                 (cx + w / 2) * W, (cy + h / 2) * H])

            r = model.predict(str(ip), conf=conf, iou=0.7, max_det=300,
                              imgsz=640, device=0, verbose=False)[0]
            det_cls, det_boxes = [], []
            if r.boxes is not None and len(r.boxes):
                for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy().astype(int)):
                    det_cls.append(int(c)); det_boxes.append(list(map(float, b)))
            n_det += len(det_cls)

            mat_u += match_ultralytics(gt_cls, gt_boxes, det_cls, det_boxes, nc)
            mat_m += match_mine(gt_cls, gt_boxes, det_cls, det_boxes, nc)

        du, col = diagonal(mat_u, nc)
        dm, _ = diagonal(mat_m, nc)
        print("=" * 74)
        print(f"conf = {conf}   预测框总数 {n_det}   真实框总数 {int(col[:nc].sum())}")
        print("=" * 74)
        print(f"  {'类别':<18}{'真实':>7}{'ultralytics':>13}{'我的贪心':>11}")
        print("  " + "-" * 50)
        for i, c in enumerate(CLASSES):
            g = int(col[i])
            print(f"  {c:<18}{g:>7}{mat_u[i,i]/g:>13.2f}{mat_m[i,i]/g:>11.2f}")
        print(f"  {'平均对角线':<18}{'':>7}{du:>13.2f}{dm:>11.2f}")
        print(f"  误检列合计            {int(mat_u[:, nc].sum()):>10}"
              f"{int(mat_m[:, nc].sum()):>11}")
        print()


if __name__ == "__main__":
    main()
