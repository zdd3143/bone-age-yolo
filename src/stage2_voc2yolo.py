"""阶段 2：把 VOC(XML) 标注转成 YOLO(txt) 格式，并划分 train/val。

输出结构（ultralytics 标准）：
    data/bone_age/
        images/train/*.png      ← 用【硬链接】指回原图，不占额外磁盘
        images/val/*.png
        labels/train/*.txt
        labels/val/*.txt
    configs/bone_age.yaml       ← 训练配置
"""
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# ---------------------------------------------------------------- 路径配置
# 数据根目录（VOC2007 的【上一级】），可用环境变量覆盖：
#     $env:BONE_DATA_ROOT = "D:\path\to\your_dataset"
# 期望的目录结构：
#     <BONE_DATA_ROOT>\VOC2007\Annotations\*.xml
#     <BONE_DATA_ROOT>\VOC2007\JPEGImages\*.png
DATA_ROOT = Path(os.environ.get("BONE_DATA_ROOT", "data/raw"))
VOC_ROOT = DATA_ROOT / "VOC2007"
ANN_DIR = VOC_ROOT / "Annotations"
IMG_DIR = VOC_ROOT / "JPEGImages"

PROJECT = Path(__file__).resolve().parent.parent
DS = PROJECT / "data" / "bone_age"
CFG = PROJECT / "configs" / "bone_age.yaml"
REPORT = PROJECT / "outputs" / "data_report"

VAL_RATIO = 0.2
SEED = 42

# ★ 类别索引必须【固定且有序】—— txt 里的 class_id 就是这里的下标，
#   而 yaml 里的 names 必须按完全相同的顺序写，否则类别会整体错位。
CLASSES = sorted([
    "DistalPhalanx", "MCP", "MCPFirst", "MiddlePhalanx",
    "ProximalPhalanx", "Radius", "Ulna",
])
CLS2ID = {c: i for i, c in enumerate(CLASSES)}


def link_or_copy(src: Path, dst: Path) -> str:
    """优先建硬链接（同盘不占额外空间），失败则复制。
    """
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)          # 硬链接：同卷、同文件、不同路径
        return "link"
    except OSError:
        shutil.copy2(src, dst)     # 跨卷或文件系统不支持时退回复制
        return "copy"


def main():
    print("=" * 70)
    print("阶段 2：VOC → YOLO 标签转换")
    print("=" * 70)

    # ---------- 先检查数据目录（否则会看到一堆难懂的报错）----------
    if not ANN_DIR.exists() or not IMG_DIR.exists():
        raise SystemExit(
            "\n找不到数据目录：\n"
            f"  标注  {ANN_DIR}\n"
            f"  图片  {IMG_DIR}\n\n"
            "请设置环境变量 BONE_DATA_ROOT 指向 VOC2007 的【上一级】目录：\n"
            '   $env:BONE_DATA_ROOT = "D:\\your_dataset"\n'
            "期望的目录结构：\n"
            "   <BONE_DATA_ROOT>\\VOC2007\\Annotations\\*.xml\n"
            "   <BONE_DATA_ROOT>\\VOC2007\\JPEGImages\\*.png\n"
        )

    # ---------- 清空旧结果，保证可重复 ----------
    if DS.exists():
        shutil.rmtree(DS)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DS / sub).mkdir(parents=True, exist_ok=True)
    CFG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    # ---------- 收集所有样本 ----------
    samples = []
    missing_img, no_obj = [], []
    degenerate = []                  # 宽或高 <= 0 的脏标注

    for xml_path in sorted(ANN_DIR.glob("*.xml")):
        stem = xml_path.stem
        img_path = IMG_DIR / f"{stem}.png"
        if not img_path.exists():
            missing_img.append(stem)
            continue

        root = ET.parse(xml_path).getroot()

        # ★★ 关键：尺寸必须从 XML 里读，不能用默认值 ★★
        W = int(root.findtext("size/width"))
        H = int(root.findtext("size/height"))

        boxes = []
        for o in root.findall("object"):
            name = (o.findtext("name") or "").strip()
            if name not in CLS2ID:
                continue
            b = o.find("bndbox")
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))

            # ★ 过滤退化框：宽或高 <= 0 的标注是错的。
            #   实测数据里有 2 个这样的框（比如 w=5 h=0），
            #   不过滤的话会生成 h=0 的标签 —— 既不合法，也让 loss 没法算。
            #   这种"个别脏标注"是真实数据集的常态，必须有防御。
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                degenerate.append((stem, name, x2 - x1, y2 - y1))
                continue

            boxes.append((CLS2ID[name], x1, y1, x2, y2, name))

        if not boxes:
            no_obj.append(stem)
            continue

        samples.append({"stem": stem, "W": W, "H": H, "boxes": boxes})

    print(f"\n【收集】")
    print(f"  可用样本        {len(samples)}")
    print(f"  缺图片的标注    {len(missing_img)}")
    print(f"  没有目标的标注  {len(no_obj)}")
    if degenerate:
        print(f"  ⚠️ 退化框被丢弃  {len(degenerate)} 个（宽或高 <= 0 的脏标注）")
        for stem, name, w, h in degenerate[:5]:
            print(f"     {stem}.xml  {name}  w={w:g} h={h:g}")

    # ---------- 划分 train/val ----------
    random.seed(SEED)
    random.shuffle(samples)
    n_val = int(len(samples) * VAL_RATIO)
    val_set = samples[:n_val]
    train_set = samples[n_val:]

    print(f"\n【划分】（比例 {1 - VAL_RATIO:.0%} / {VAL_RATIO:.0%}，seed={SEED}）")
    print(f"  train  {len(train_set):>4} 张")
    print(f"  val    {len(val_set):>4} 张")

    # ---------- 转换 + 写文件 ----------
    stats = {"link": 0, "copy": 0, "exists": 0}
    n_boxes = {"train": 0, "val": 0}

    for split, data in (("train", train_set), ("val", val_set)):
        for s in data:
            src_img = IMG_DIR / f"{s['stem']}.png"
            dst_img = DS / "images" / split / f"{s['stem']}.png"
            stats[link_or_copy(src_img, dst_img)] += 1

            lines = []
            for cid, x1, y1, x2, y2, _name in s["boxes"]:
                W, H = s["W"], s["H"]
                cx = (x1 + x2) / 2.0 / W
                cy = (y1 + y2) / 2.0 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                # 裁剪到 [0,1]，防止标注越界导致 YOLO 报错
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (DS / "labels" / split / f"{s['stem']}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            n_boxes[split] += len(lines)

    print(f"\n【图片处理】硬链接 {stats['link']}，复制 {stats['copy']}，"
          f"已存在 {stats['exists']}")
    if stats["link"]:
        print("  ✓ 用硬链接，没有额外占用 1.4 GB 磁盘")

    # ---------- 写 yaml ----------
    cfg = {
        "path": str(DS).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": {i: c for i, c in enumerate(CLASSES)},
    }
    CFG.write_text(
        "# 骨龄关节检测 · YOLO 数据集配置\n"
        "# 类别索引必须和 labels/*.txt 里的 class_id 一一对应\n"
        + yaml.dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    print(f"\n【类别索引】")
    for i, c in enumerate(CLASSES):
        print(f"  {i}  {c}")

    # 把生成的 YOLO 标签反算回绝对像素坐标，和原 XML 比对。
    print(f"\n{'=' * 70}")
    print("★ 往返自检：把 YOLO 标签反算回绝对坐标，和原 XML 比对")
    print("=" * 70)

    bad = []
    checked = 0
    max_err = 0.0

    for split, data in (("train", train_set), ("val", val_set)):
        for s in data[:80]:                       # 抽查 80 张，够发现系统性问题
            label_file = DS / "labels" / split / f"{s['stem']}.txt"
            txt = label_file.read_text(encoding="utf-8").strip().splitlines()

            if len(txt) != len(s["boxes"]):
                bad.append((s["stem"], f"框数不符 {len(txt)} vs {len(s['boxes'])}"))
                continue

            W, H = s["W"], s["H"]
            for line, (cid, x1, y1, x2, y2, name) in zip(txt, s["boxes"]):
                p = line.split()
                if int(p[0]) != cid:
                    bad.append((s["stem"], f"类别不符 {p[0]} vs {cid}"))
                    continue
                cx, cy, bw, bh = map(float, p[1:])

                # 反算
                rx1 = (cx - bw / 2) * W
                ry1 = (cy - bh / 2) * H
                rx2 = (cx + bw / 2) * W
                ry2 = (cy + bh / 2) * H

                err = max(abs(rx1 - x1), abs(ry1 - y1),
                          abs(rx2 - x2), abs(ry2 - y2))
                max_err = max(max_err, err)
                checked += 1
                if err > 0.01:                    # 亚像素级误差是正常的（6 位小数）
                    bad.append((s["stem"], f"{name} 偏差 {err:.4f} px"))

    print(f"  抽查框数        {checked}")
    print(f"  最大像素误差    {max_err:.6f} px")
    if bad:
        print(f"  ❌ 发现 {len(bad)} 处异常：")
        for stem, msg in bad[:10]:
            print(f"     {stem}: {msg}")
    else:
        print("  ✅ 全部对得上 —— 转换逻辑正确")

    # ---------- 汇总 ----------
    summary = {
        "classes": CLASSES,
        "train_images": len(train_set),
        "val_images": len(val_set),
        "train_boxes": n_boxes["train"],
        "val_boxes": n_boxes["val"],
        "val_ratio": VAL_RATIO,
        "seed": SEED,
        "roundtrip_max_error_px": round(max_err, 6),
        "roundtrip_checked": checked,
        "roundtrip_bad": len(bad),
        "link_stats": stats,
    }
    (REPORT / "stage2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"数据集：{DS}")
    print(f"配置  ：{CFG}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()