"""给 ultralytics 的 val 加探针：直接打印它【内部】的混淆矩阵。

目的
----
之前我从 PNG 热力图上读数字，只能读到"对角线 0.3~0.6"。
现在把 `metrics.confusion_matrix.matrix` 原样打出来，
就能知道确切的对角线值，并且可以在不同 conf 下对比。

如果 conf=0.25 时对角线明显变高 —— 那就证明
「val 默认 conf=0.001 + 匹配不分类别」就是根因。
"""
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
CFG = PROJECT / "configs" / "bone_age.yaml"
WEIGHTS = PROJECT / "outputs" / "runs" / "full" / "weights" / "best.pt"

CLASSES = ["DistalPhalanx", "MCP", "MCPFirst", "MiddlePhalanx",
           "ProximalPhalanx", "Radius", "Ulna"]


def report(tag, m):
    mat = m.confusion_matrix.matrix
    n = len(CLASSES)
    col = mat.sum(axis=0)
    print(f"\n{'=' * 74}")
    print(f"conf = {tag}")
    print(f"{'=' * 74}")
    print(f"  矩阵形状 {mat.shape}   真实框总数 {int(col[:n].sum())}")
    print(f"\n  {'类别':<18}{'真实框':>8}{'判对':>7}{'对角线':>9}")
    print("  " + "-" * 42)
    diags = []
    for i, c in enumerate(CLASSES):
        g = int(col[i])
        ok = int(mat[i, i])
        d = ok / g if g else 0
        diags.append(d)
        print(f"  {c:<18}{g:>8}{ok:>7}{d:>9.2f}")
    print(f"  {'平均':<18}{'':>8}{'':>7}{np.mean(diags):>9.2f}")
    print(f"\n  误检（background 列）合计：{int(mat[:, n].sum())}")
    print(f"  mAP 报告：P={m.box.mp:.4f}  R={m.box.mr:.4f}  "
          f"mAP50={m.box.map50:.4f}")


def main():
    model = YOLO(str(WEIGHTS))
    print("=" * 74)
    print("ultralytics 内部混淆矩阵 · 不同 conf 对比")
    print("=" * 74, flush=True)

    for conf in (0.001, 0.25):
        m = model.val(data=str(CFG), conf=conf, plots=True,
                      project=str(PROJECT / "outputs" / "runs"),
                      name=f"cm_probe_{conf}", exist_ok=True)
        report(conf, m)
        print(flush=True)


if __name__ == "__main__":
    main()
