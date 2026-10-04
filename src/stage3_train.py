"""阶段 3：用 YOLOv8 训练手骨关节检测模型。

设计原则：把参数全部显式写出来，不要用默认值糊过去 ——
         因为面试官一定会问"你为什么这么设"。

关键参数说明：
    model     预训练权重。用 yolov8n（nano，最小最快）先跑通，不是因为它最好，
              而是因为它【最快能告诉你数据有没有问题】。
    imgsz     训练分辨率。640 是 YOLO 默认档位；再大显存吃不住，再小关节看不清。
    batch     批大小。8 GB 显存 + yolov8n@640，8 是安全值（16 可能 OOM）。
    epochs    训练轮数。100 对 705 张图是合理量级，配合 patience 早停。
    patience  早停耐心值。30 轮指标不涨就停，省时间也防过拟合。
    workers   数据加载进程数。Windows 上不能设太大，否则反而更慢。
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CFG = PROJECT / "configs" / "bone_age.yaml"
RUNS = PROJECT / "outputs" / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="0=第一块 GPU，cpu=用 CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="bone_age")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，只加载已有权重做评估")
    args = ap.parse_args()

    # 关掉 ultralytics 的联网同步，避免训练被网络问题打断
    try:
        from ultralytics import settings
        settings.update({"sync": False})
    except Exception:
        pass

    from ultralytics import YOLO

    print("=" * 70)
    print("阶段 3：YOLOv8 训练")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<10} {v}")
    print("=" * 70, flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        weights = RUNS / args.name / "weights" / "best.pt"
        if not weights.exists():
            raise SystemExit(f"找不到权重：{weights}")
        model = YOLO(str(weights))
        print(f"已加载权重：{weights}")
    else:
        model = YOLO(args.model)
        model.train(
            data=str(CFG),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            project=str(RUNS),
            name=args.name,
            exist_ok=True,
            plots=True,          # 混淆矩阵 / PR 曲线 / 训练曲线
        )

    # ---------- 在验证集上评估 ----------
    print("\n" + "=" * 70)
    print("在验证集上评估")
    print("=" * 70, flush=True)
    metrics = model.val(data=str(CFG), imgsz=args.imgsz,
                        batch=args.batch, device=args.device,
                        project=str(RUNS), name=f"{args.name}_val",
                        exist_ok=True)

    names = metrics.names
    per_class = {}
    for i, cid in enumerate(metrics.ap_class_index):
        per_class[names[int(cid)]] = {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "AP50": float(metrics.box.ap50[i]),
            "AP50_95": float(metrics.box.ap[i]),
        }

    result = {
        "args": vars(args),
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": per_class,
    }

    out = RUNS / args.name / "metrics_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"mAP@0.5        {result['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   {result['metrics']['mAP50_95']:.4f}")
    print(f"精确率         {result['metrics']['precision']:.4f}")
    print(f"召回率         {result['metrics']['recall']:.4f}")
    print("\n分类别：")
    for name, m in sorted(per_class.items()):
        print(f"  {name:<18} P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"AP50={m['AP50']:.4f}")
    print(f"\n汇总已存：{out}")
    print(f"全部产物：{RUNS / args.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
