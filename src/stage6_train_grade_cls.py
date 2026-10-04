"""阶段 6：训练关节发育等级分类器。

数据
----
    arthrosis/<关节名>/<等级>/*.png
    9 类关节 × 10~14 个等级，共 8,210 张切图。
    等级就是 RUS-CHN 的发育等级（1 级最早、等级越高越成熟）。

设计
----
    · 每个关节单独训一个小 CNN —— 因为不同关节的等级数不一样
      （Radius 14 级、MCP 10 级），而且等级的含义也不同。
    · 输入统一缩放到 96x96 灰度，把 8,210 张全部载入内存（约 75 MB）。
    · 用【加权交叉熵】应对类别不平衡 —— 最少的等级只有 6 个样本。
    · 少数样本极少的等级：全部放进训练集，不放进验证集（否则 val 里只有 1 张，没有意义）。

⚠️ 必须说明的局限
----------------
    arthrosis/ 的切图来自【另一批 X 光片】（编号 6~7 位，与我们那 881 张的 5 位编号对不上）。
    所以：
        · 在这个数据集内部的 val 指标，只说明"它学会了这批数据的等级规律"
        · 迁移到我们检测框裁出来的图上，会有【域偏移】（裁剪方式、成像参数都可能不同）
    这一点在报告里会写明。

产出
----
    outputs/cls/<关节名>.pt          每个关节的模型权重
    outputs/cls/grade_cls_report.json 训练指标汇总
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parent.parent
DATA = Path(r"D:\AAA\骨龄-pdf讲义\arthrosis")
OUT = PROJECT / "outputs" / "cls"
REPORT = PROJECT / "outputs" / "data_report"

IMG_SIZE = 128
BATCH = 64
EPOCHS = 50
LR = 1e-3
SEED = 42
MIN_VAL_PER_CLASS = 5       # 少于这个数的等级，全部进训练集


class SmallCNN(nn.Module):
    """4 个卷积块 + 全局平均池化 + 单层分类头。参数量很小，训练快。"""

    def __init__(self, n_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def load_split(joint_dir):
    """载入某个关节的全部切图到内存。返回 (images, labels, grade_names)。"""
    grades = sorted([p for p in joint_dir.iterdir() if p.is_dir()],
                    key=lambda p: int(p.name))
    names = [g.name for g in grades]
    imgs, labels = [], []
    for gi, g in enumerate(grades):
        for f in g.glob("*"):
            try:
                im = Image.open(f).convert("L").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            except Exception:
                continue
            imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
            labels.append(gi)
    if not imgs:
        return None, None, None
    x = torch.from_numpy(np.stack(imgs)).unsqueeze(1)     # (N,1,H,W)
    y = torch.tensor(labels, dtype=torch.long)
    # 注意：这里【不】做归一化 —— 归一化统计量必须只用训练集算，
    #       否则就是把验证集的信息泄漏进了训练（data leakage）。
    return x, y, names


def split_train_val(y, n_classes):
    """分层划分：少数类全部进训练集。"""
    rng = random.Random(SEED)
    tr, va = [], []
    for c in range(n_classes):
        idx = [i for i in range(len(y)) if int(y[i]) == c]
        rng.shuffle(idx)
        if len(idx) < MIN_VAL_PER_CLASS:
            tr += idx
        else:
            k = max(1, int(len(idx) * 0.2))
            va += idx[:k]
            tr += idx[k:]
    return tr, va


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), f1s


def ordinal_metrics(y_true, y_pred):
    """有序分类真正该看的指标。

    等级是【有序变量】—— 1 级和 2 级的差距，
    远小于 1 级和 14 级的差距。
    但准确率把"差 1 级"和"差 13 级"算成一样错，所以它会严重低估模型。

    这里补三个指标：
        ±1 准确率   预测落在真实等级 ±1 以内（临床上这通常可接受）
        MAE         平均绝对等级误差
        Pearson r   预测等级与真实等级的相关性（衡量单调趋势）
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    within1 = float(np.mean(np.abs(t - p) <= 1))
    mae = float(np.mean(np.abs(t - p)))
    if t.std() > 1e-9 and p.std() > 1e-9:
        r = float(np.corrcoef(t, p)[0, 1])
    else:
        r = float("nan")
    return within1, mae, r


def train_one(joint, x, y, names, device, epochs):
    n_classes = len(names)
    tr, va = split_train_val(y, n_classes)
    x_tr, y_tr = x[tr], y[tr]
    has_val = len(va) > 0
    if has_val:
        x_va, y_va = x[va], y[va]

    # ★ 归一化统计量【只用训练集】算，然后同时应用到训练和验证。
    #   用全数据集的统计量是数据泄漏。
    #   推理时必须用同一组 mean/std —— 所以存进 checkpoint。
    mean = float(x_tr.mean())
    std = float(x_tr.std() + 1e-6)
    x_tr = (x_tr - mean) / std
    if has_val:
        x_va = (x_va - mean) / std

    # 加权交叉熵：按类别频率的倒数加权
    cnt = Counter(int(v) for v in y_tr)
    w = torch.tensor([len(y_tr) / (n_classes * max(cnt.get(c, 1), 1))
                      for c in range(n_classes)], dtype=torch.float32,
                     device=device)

    model = SmallCNN(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=w)

    dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH,
                    shuffle=True, drop_last=len(y_tr) > BATCH)

    best_f1, best_state, best_ep = -1.0, None, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                pred = model(x_va.to(device)).argmax(1).cpu().tolist()
            yt = y_va.tolist()
            acc = sum(1 for t, p in zip(yt, pred) if t == p) / len(yt)
            f1, _ = macro_f1(yt, pred, n_classes)
            w1, mae, r = ordinal_metrics(yt, pred)
            hist.append({"epoch": ep, "train_loss": round(tot / max(nb, 1), 4),
                         "val_acc": round(acc, 4), "val_macro_f1": round(f1, 4),
                         "val_within1": round(w1, 4), "val_mae": round(mae, 3),
                         "val_pearson": round(r, 4)})
            # 以 ±1 准确率作为选模型的依据 —— 它才反映"等级判得准不准"
            if w1 > best_f1:
                best_f1 = w1
                best_ep = ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_ep = ep

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 最终指标（用最好的那个 checkpoint）
    res = {"joint": joint, "n_classes": n_classes, "grades": names,
           "n_total": len(y), "n_train": len(tr), "n_val": len(va),
           "best_epoch": best_ep, "history": hist,
           "norm_mean": round(mean, 6), "norm_std": round(std, 6)}
    if has_val:
        with torch.no_grad():
            pred = model(x_va.to(device)).argmax(1).cpu().tolist()
        yt = y_va.tolist()
        acc = sum(1 for t, p in zip(yt, pred) if t == p) / len(yt)
        f1, per_f1 = macro_f1(yt, pred, n_classes)
        w1, mae, r = ordinal_metrics(yt, pred)
        res.update({"val_acc": round(acc, 4), "val_macro_f1": round(f1, 4),
                    "val_within1": round(w1, 4), "val_mae": round(mae, 3),
                    "val_pearson": round(r, 4)})
        # 混淆矩阵
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(yt, pred):
            cm[t, p] += 1
        res["confusion"] = cm.tolist()
        res["per_class_f1"] = {names[i]: round(per_f1[i], 4)
                               for i in range(n_classes)}
        res["per_class_support"] = {names[i]: int((y_va == i).sum())
                                    for i in range(n_classes)}

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "n_classes": n_classes, "grades": names,
                "img_size": IMG_SIZE, "joint": joint,
                "norm_mean": mean, "norm_std": std},
               OUT / f"{joint}.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        raise SystemExit(
            f"找不到等级分类数据：{data_root}\n"
            "用 --data 指定 arthrosis 目录（结构：<关节名>/<等级>/*.png）"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    joints = sorted([d for d in data_root.iterdir() if d.is_dir()])

    print("=" * 78)
    print(f"阶段 6：发育等级分类器训练（{len(joints)} 个关节）")
    print(f"  设备 {device}   输入 {IMG_SIZE}x{IMG_SIZE} 灰度   "
          f"batch {BATCH}   epochs {args.epochs}")
    print("=" * 78, flush=True)

    results = []
    t0 = time.time()
    for j in joints:
        x, y, names = load_split(j)
        if x is None:
            print(f"\n[{j.name}] 没读到图片，跳过")
            continue
        cnt = Counter(int(v) for v in y)
        print(f"\n[{j.name}] {len(y)} 张 / {len(names)} 个等级  "
              f"({', '.join(f'{names[c]}:{cnt[c]}' for c in sorted(cnt))})",
              flush=True)
        r = train_one(j.name, x, y, names, device, args.epochs)
        results.append(r)
        if "val_acc" in r:
            print(f"    → 准确率 {r['val_acc']:.4f}   "
                  f"±1等级 {r['val_within1']:.4f}   "
                  f"MAE {r['val_mae']:.2f}   r {r['val_pearson']:.3f}   "
                  f"best epoch {r['best_epoch']}/{args.epochs}")
        else:
            print(f"    → 无验证集（等级样本太少），训练 {r['best_epoch']} 轮")

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  {'关节':<11}{'等级数':>7}{'样本':>7}{'准确率':>9}{'±1等级':>9}"
          f"{'MAE':>7}{'Pearson r':>11}")
    print("  " + "-" * 62)
    for r in results:
        if "val_acc" not in r:
            print(f"  {r['joint']:<11}{r['n_classes']:>7}{r['n_total']:>7}"
                  f"{'（无验证集）':>18}")
            continue
        print(f"  {r['joint']:<11}{r['n_classes']:>7}{r['n_total']:>7}"
              f"{r['val_acc']:>9.4f}{r['val_within1']:>9.4f}"
              f"{r['val_mae']:>7.2f}{r['val_pearson']:>11.3f}")

    with_val = [r for r in results if "val_acc" in r]
    if with_val:
        print(f"\n  {'平均':<11}{'':>7}{'':>7}"
              f"{np.mean([r['val_acc'] for r in with_val]):>9.4f}"
              f"{np.mean([r['val_within1'] for r in with_val]):>9.4f}"
              f"{np.mean([r['val_mae'] for r in with_val]):>7.2f}"
              f"{np.mean([r['val_pearson'] for r in with_val]):>11.3f}")

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "grade_cls_report.json").write_text(
        json.dumps({"img_size": IMG_SIZE, "epochs": args.epochs,
                    "device": device, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  模型：{OUT}\\<关节>.pt  （共 {len(results)} 个）")
    print(f"  报告：{REPORT / 'grade_cls_report.json'}")
    print(f"  耗时：{(time.time() - t0) / 60:.1f} 分钟")
    print("""
⚠️ 这个指标只说明"它学会了 arthrosis 这批数据的等级规律"。
   迁移到我们检测框裁出来的图上会有域偏移，见脚本开头的说明。""")


if __name__ == "__main__":
    main()
