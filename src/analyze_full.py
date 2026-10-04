"""分析完整训练的 results.csv：找出最优点、看收敛过程。"""
import csv
from pathlib import Path

PROJECT = Path(r"D:\AAA\bone-age-yolo")
CSV = PROJECT / "outputs" / "runs" / "full" / "results.csv"

rows = []
with CSV.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append({k.strip(): v.strip() for k, v in r.items()})

def g(r, k):
    return float(r[k])

print("=" * 78)
print(f"共 {len(rows)} 轮")
print("=" * 78)

# ---- 找出各指标的最优轮次 ----
best = {
    "mAP50": max(rows, key=lambda r: g(r, "metrics/mAP50(B)")),
    "mAP50-95": max(rows, key=lambda r: g(r, "metrics/mAP50-95(B)")),
    "precision": max(rows, key=lambda r: g(r, "metrics/precision(B)")),
    "recall": max(rows, key=lambda r: g(r, "metrics/recall(B)")),
}
print(f"\n{'指标':<14}{'最优轮':>7}{'值':>10}{'第 5 轮':>10}{'最后一轮':>11}")
print("-" * 55)
for k, r in best.items():
    col = f"metrics/{k}(B)"
    e5 = g(rows[4], col)
    last = g(rows[-1], col)
    print(f"  {k:<12}{int(g(r,'epoch')):>7}{g(r,col):>10.4f}"
          f"{e5:>10.4f}{last:>11.4f}")

# ---- 关键结论：5 轮 vs 83 轮 ----
print("\n" + "=" * 78)
print("★ 5 轮 vs 83 轮：训练这么久到底有没有用？")
print("=" * 78)
for col, name in [("metrics/mAP50(B)", "mAP@0.5"),
                  ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
                  ("train/box_loss", "train box_loss"),
                  ("val/box_loss", "val box_loss")]:
    e5, elast = g(rows[4], col), g(rows[-1], col)
    print(f"  {name:<16} 第5轮 {e5:>9.4f}  →  第83轮 {elast:>9.4f}"
          f"   变化 {elast - e5:+.4f}")

# ---- 找"首次达到最终水平"的轮次 ----
print("\n" + "=" * 78)
print("收敛速度：什么时候就到顶了？")
print("=" * 78)
final50 = g(rows[-1], "metrics/mAP50(B)")
final95 = g(rows[-1], "metrics/mAP50-95(B)")
for col, name, fin in [("metrics/mAP50(B)", "mAP@0.5", final50),
                       ("metrics/mAP50-95(B)", "mAP@0.5:0.95", final95)]:
    hit = next((int(g(r, "epoch")) for r in rows
                if g(r, col) >= fin * 0.99), None)
    print(f"  {name:<14} 首次达到最终值 99% 的轮次：第 {hit} 轮")

# ---- 每 10 轮采样 ----
print("\n" + "=" * 78)
print("训练曲线（每 10 轮采样）")
print("=" * 78)
print(f"  {'轮':>4}{'mAP50':>9}{'mAP50-95':>10}{'P':>9}{'R':>9}"
      f"{'train_box':>11}{'val_box':>10}")
print("  " + "-" * 62)
for i in range(0, len(rows), 10):
    r = rows[i]
    print(f"  {int(g(r,'epoch')):>4}{g(r,'metrics/mAP50(B)'):>9.4f}"
          f"{g(r,'metrics/mAP50-95(B)'):>10.4f}"
          f"{g(r,'metrics/precision(B)'):>9.4f}{g(r,'metrics/recall(B)'):>9.4f}"
          f"{g(r,'train/box_loss'):>11.4f}{g(r,'val/box_loss'):>10.4f}")
r = rows[-1]
print(f"  {int(g(r,'epoch')):>4}{g(r,'metrics/mAP50(B)'):>9.4f}"
      f"{g(r,'metrics/mAP50-95(B)'):>10.4f}"
      f"{g(r,'metrics/precision(B)'):>9.4f}{g(r,'metrics/recall(B)'):>9.4f}"
      f"{g(r,'train/box_loss'):>11.4f}{g(r,'val/box_loss'):>10.4f}")

# ---- 过拟合检查 ----
print("\n" + "=" * 78)
print("过拟合检查：train loss 还在降，val loss 有没有跟着降？")
print("=" * 78)
tb = [g(r, "train/box_loss") for r in rows]
vb = [g(r, "val/box_loss") for r in rows]
print(f"  train/box_loss  最低 {min(tb):.4f}（第 {tb.index(min(tb)) + 1} 轮）"
      f"   最终 {tb[-1]:.4f}")
print(f"  val/box_loss    最低 {min(vb):.4f}（第 {vb.index(min(vb)) + 1} 轮）"
      f"   最终 {vb[-1]:.4f}")
if vb[-1] > min(vb) * 1.02:
    print("  ⚠️ val loss 已从最低点回升 -> 有轻微过拟合迹象")
else:
    print("  ✅ val loss 没有明显回升")

print(f"\n  总训练时长：{g(rows[-1], 'time') / 60:.1f} 分钟"
      f"（含之前那次休眠跳变）")
