"""阶段 4：批量推理 + 结果缓存。

为什么单独做这一步
------------------
推理一次不便宜：881 张图、单张 1~3 秒，总共约 2~3 分钟。
而后面有好几个脚本都要用检测结果：
    · stage5_select13.py   13 关节筛选
    · stage5b_draw13.py    结果可视化
    · 各种统计（每类框数分布、几何关系等）

所以把推理结果缓存成一份 JSON，后续脚本直接读，
**从"每次都跑 3 分钟"变成"秒开"**。

缓存内容（每张图一条）：
    {
      "14714": {
        "split": "train",
        "detections": [
            {"class": "Radius", "conf": 0.79, "bbox": [x1, y1, x2, y2]},
            ...
        ]
      },
      ...
    }

用法
----
    python src/stage4_infer.py             # 有缓存就跳过，没有就跑
    python src/stage4_infer.py --force     # 强制重新推理
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT / "outputs" / "runs" / "full" / "weights" / "best.pt"
IMG_DIRS = {
    "train": PROJECT / "data" / "bone_age" / "images" / "train",
    "val": PROJECT / "data" / "bone_age" / "images" / "val",
}
CACHE = PROJECT / "outputs" / "detections" / "all_detections.json"

CONF = 0.25          # 和训练时的验证保持一致的置信度阈值


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略缓存，重新推理")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--weights", default=str(WEIGHTS))
    args = ap.parse_args()

    if CACHE.exists() and not args.force:
        n = len(json.loads(CACHE.read_text(encoding="utf-8")))
        print(f"已有缓存（{n} 张），跳过推理。要重跑请加 --force")
        print(f"  {CACHE}")
        return

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"找不到权重：{weights}\n"
            "先跑 python src/stage3_train.py 训练，或指定 --weights 指向已有模型"
        )

    imgs = []
    for split, d in IMG_DIRS.items():
        if not d.exists():
            raise SystemExit(f"找不到图片目录：{d}\n先跑 python src/stage2_voc2yolo.py")
        imgs += [(split, p) for p in sorted(d.glob("*.png"))]

    print("=" * 70)
    print(f"批量推理：{len(imgs)} 张")
    print(f"  权重  {weights}")
    print(f"  conf  {args.conf}")
    print("=" * 70, flush=True)

    model = YOLO(str(weights))
    out = {}
    for i, (split, p) in enumerate(imgs, 1):
        r = model.predict(str(p), conf=args.conf, imgsz=640,
                          device=0, verbose=False)[0]
        names = r.names
        dets = []
        if r.boxes is not None and len(r.boxes):
            for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                    r.boxes.cls.cpu().numpy().astype(int),
                                    r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = (float(v) for v in box)
                dets.append({"class": names[int(cls)],
                             "conf": round(float(cf), 4),
                             "bbox": [round(x1, 1), round(y1, 1),
                                      round(x2, 1), round(y2, 1)]})
        out[p.stem] = {"split": split, "detections": dets}
        if i % 200 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    n_box = sum(len(v["detections"]) for v in out.values())
    print(f"\n完成：{len(out)} 张图，共 {n_box} 个框，"
          f"平均 {n_box / len(out):.1f} 个/图")
    print(f"缓存已存：{CACHE}  （{CACHE.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
