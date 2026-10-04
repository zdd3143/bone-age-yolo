"""阶段 1：看懂数据。
    输出：
    outputs/data_report/  —— 统计图 + data_meta.json
    outputs/data_samples/ —— 带标注框的样例图
"""
import collections
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

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

OUT = Path(__file__).resolve().parent.parent / "outputs"
REPORT = OUT / "data_report"
SAMPLES = OUT / "data_samples"

CLASS_COLORS = {
    "Radius": (255, 80, 80),
    "Ulna": (80, 140, 255),
    "MCPFirst": (255, 180, 0),
    "MCP": (0, 190, 120),
    "ProximalPhalanx": (200, 80, 255),
    "MiddlePhalanx": (255, 120, 180),
    "DistalPhalanx": (120, 200, 255),
}


def boxplot_compat(ax, data, order):
    """画横向箱线图，兼容不同 matplotlib 版本。

    这个 API 在三个版本里改过两次，真实项目里很常见：
        < 3.9          labels=...
        3.9 ~ 3.10     labels 改名 tick_labels
        >= 3.10        vert=False 改名 orientation="horizontal"
        >= 3.11        移除 labels；vert 也开始报弃用警告
    所以按「新 API 优先」依次尝试。
    """
    import warnings

    common = dict(patch_artist=True, widths=0.6)
    attempts = [
        dict(orientation="horizontal", tick_labels=order),   # >= 3.10
        dict(vert=False, tick_labels=order),                 # 3.9 ~ 3.10
        dict(vert=False, labels=order),                      # < 3.9
    ]
    for kw in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ax.boxplot(data, **common, **kw)
        except TypeError:
            continue
    raise RuntimeError("当前 matplotlib 版本不支持任何已知的 boxplot 参数组合")


def parse_all():
    records, cls_count, bad = [], collections.Counter(), []

    for x in sorted(ANN_DIR.glob("*.xml")):
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError as e:
            bad.append((x.name, str(e)))
            continue

        size = root.find("size")
        w = int(size.findtext("width"))
        h = int(size.findtext("height"))

        boxes = []
        for o in root.findall("object"):
            name = (o.findtext("name") or "").strip()
            b = o.find("bndbox")
            if b is None:
                continue
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            boxes.append((name, x1, y1, x2, y2))
            cls_count[name] += 1

        records.append({"stem": x.stem, "w": w, "h": h, "boxes": boxes})

    return records, cls_count, bad


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段 1：数据探查")
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

    records, cls_count, bad = parse_all()
    total_boxes = sum(len(r["boxes"]) for r in records)

    print("\n【总览】")
    print(f"  图片数          {len(records)}")
    print(f"  标注框总数      {total_boxes}")
    print(f"  类别数          {len(cls_count)}")
    print(f"  平均每图框数    {total_boxes / len(records):.1f}")
    if bad:
        print(f"  解析失败 XML    {len(bad)} 个 -> {bad[:3]}")
    else:
        print("  解析失败 XML    0 个（全部可解析）")

    print("\n【类别分布】")
    for k, v in cls_count.most_common():
        print(f"  {k:<20} {v:>6} 个  ({v / total_boxes * 100:5.1f}%)")

    per_img = [len(r["boxes"]) for r in records]
    print("\n【每张图的框数】")
    print(f"  最少 {min(per_img)}   最多 {max(per_img)}   "
          f"平均 {sum(per_img) / len(per_img):.2f}")

    sizes = collections.Counter((r["w"], r["h"]) for r in records)
    print(f"\n【图片尺寸】共 {len(sizes)} 种不同尺寸")
    for (w, h), c in sizes.most_common(6):
        print(f"  {w:>5} × {h:<5}   {c:>4} 张")

    print("\n【框的相对大小】框面积占整图的比例")
    rel = collections.defaultdict(list)
    for r in records:
        area = r["w"] * r["h"]
        for name, x1, y1, x2, y2 in r["boxes"]:
            rel[name].append((x2 - x1) * (y2 - y1) / area)

    for k in sorted(rel, key=lambda k: sum(rel[k]) / len(rel[k])):
        vals = rel[k]
        avg = sum(vals) / len(vals)
        small = sum(1 for v in vals if v < 0.005)
        print(f"  {k:<20} 平均 {avg * 100:5.2f}%   "
              f"<0.5% 的有 {small:>5} 个 ({small / len(vals) * 100:3.0f}%)")

    # ================= 画图 =================
    st = dict(edgecolor="#1F3B73", zorder=2)

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=180)
    ks = [k for k, _ in cls_count.most_common()]
    vs = [cls_count[k] for k in ks]
    bars = ax.bar(ks, vs, color="#DCE6F5", **st)
    bars[0].set_color("#2E7D32")
    for b, v in zip(bars, vs):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, str(v),
                ha="center", fontsize=8, fontweight="bold", color="#1F3B73")
    ax.set_ylabel("标注框数量", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=18)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_title(f"类别分布（共 {total_boxes:,} 个框 / {len(records)} 张图）",
                 fontsize=10, fontweight="bold", color="#1F3B73")
    fig.tight_layout()
    fig.savefig(REPORT / "01_class_dist.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=180)
    ax.hist(per_img, bins=range(min(per_img), max(per_img) + 2),
            color="#DCE6F5", edgecolor="#1F3B73", zorder=2)
    ax.set_xlabel("每张图的关节数", fontsize=9)
    ax.set_ylabel("图片数", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_title("每张图的框数分布", fontsize=10,
                 fontweight="bold", color="#1F3B73")
    fig.tight_layout()
    fig.savefig(REPORT / "02_boxes_per_image.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=180)
    ax.scatter([r["w"] for r in records], [r["h"] for r in records],
               s=8, alpha=0.5, color="#1F3B73", zorder=2)
    ax.set_xlabel("宽 (px)", fontsize=9)
    ax.set_ylabel("高 (px)", fontsize=9)
    ax.grid(ls=":", alpha=0.4, zorder=0)
    ax.set_title(f"图片尺寸分布（{len(sizes)} 种）", fontsize=10,
                 fontweight="bold", color="#1F3B73")
    fig.tight_layout()
    fig.savefig(REPORT / "03_image_sizes.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.0), dpi=180)
    order = sorted(rel, key=lambda k: sum(rel[k]) / len(rel[k]))
    data = [[v * 100 for v in rel[k]] for k in order]
    bp = boxplot_compat(ax, data, order)
    for patch in bp["boxes"]:
        patch.set_facecolor("#DCE6F5")
        patch.set_edgecolor("#1F3B73")
    for med in bp["medians"]:
        med.set_color("#D2691E")
        med.set_linewidth(2)
    ax.axvline(0.5, ls="--", color="#D2691E", lw=1.2, zorder=3)
    ax.text(0.52, len(order) + 0.32, "0.5%（小目标参考线）",
            fontsize=7.5, color="#D2691E")
    ax.set_xscale("log")
    ax.set_xlabel("框面积 / 整图面积（%，对数轴）", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)
    ax.set_title("关节框的相对大小 —— 越靠左的类别越容易漏检",
                 fontsize=10, fontweight="bold", color="#1F3B73")
    fig.tight_layout()
    fig.savefig(REPORT / "04_box_scale.png", bbox_inches="tight")
    plt.close(fig)

    # ================= 样例可视化 =================
    random.seed(42)
    picks = random.sample(records, 4)
    for i, rec in enumerate(picks, 1):
        img_path = IMG_DIR / f"{rec['stem']}.png"
        if not img_path.exists():
            print(f"  ⚠️ 找不到图片 {img_path.name}")
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        lw = max(2, int(max(img.size) / 400))

        for name, x1, y1, x2, y2 in rec["boxes"]:
            color = CLASS_COLORS.get(name, (255, 255, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            draw.text((x1 + 4, y1 + 4), name, fill=color)

        img.thumbnail((1000, 1000))
        img.save(SAMPLES / f"sample_{i}_{rec['stem']}.png")
        print(f"  画好样例 {i}: {rec['stem']}.png  "
              f"（{len(rec['boxes'])} 个框，原图 {rec['w']}×{rec['h']}）")

    meta = {
        "num_images": len(records),
        "num_boxes": total_boxes,
        "classes": dict(cls_count.most_common()),
        "boxes_per_image": {"min": min(per_img), "max": max(per_img),
                            "avg": round(total_boxes / len(records), 2)},
        "image_sizes": {f"{w}x{h}": c for (w, h), c in sizes.most_common()},
        "box_relative_area": {k: round(sum(v) / len(v), 6)
                              for k, v in rel.items()},
        "parse_errors": bad,
    }
    (REPORT / "data_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"报告输出：{REPORT}")
    print(f"样例图  ：{SAMPLES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
