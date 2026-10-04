"""给一张手部 X 光片，自动完成骨龄评估的【检测 + 筛选 + 裁剪】。

用法
----
    python src/predict_bone_age.py --image path/to/hand.png
    python src/predict_bone_age.py --image hand.png --out outputs/predict

输出（默认写到 outputs/predict/<图片名>/）
----
    01_detections.jpg     21 个关节的检测结果（原图 + 框）
    02_rus13.jpg          13 个 RUS-CHN 评分骨（按手指编号着色 + RUS 名称）
    crops/                13 个评分骨的裁剪图（128x128）
    report.json           结构化结果（坐标、置信度、类别、手指编号）

⚠️ 关于「骨龄数字」
------------------
    本脚本完成到【检测 21 个关节 → 筛选 13 个评分骨 → 裁剪】。

    要输出一个骨龄数字，还差两步：
        ① 用 arthrosis/ 的数据训一个等级分类器，判定每块骨的发育等级（1~8 级）
        ② 用 CHN-05 官方评分表把 13 个等级换算成骨龄

    这两步本仓库没有实现，所以脚本**不会给一个假的数字**。
    但它会把分类器需要的输入（13 个标准化的裁剪图）全部准备好 ——
    接上分类器就能跑通全流程。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT / "outputs" / "runs" / "full" / "weights" / "best.pt"
DEFAULT_OUT = PROJECT / "outputs" / "predict"

# 13 块 RUS-CHN 评分骨：(编号, RUS 名称, 数据里的类别, 手指编号)
#   手指编号 None = 不分手指（桡骨/尺骨）
RUS13 = [
    ("1",  "Radius",   "Radius",          None),
    ("2",  "Ulna",     "Ulna",            None),
    ("3",  "MC-I",     "MCPFirst",        1),
    ("4",  "MC-III",   "MCP",             3),
    ("5",  "MC-V",     "MCP",             5),
    ("6",  "PP-I",     "ProximalPhalanx", 1),
    ("7",  "PP-III",   "ProximalPhalanx", 3),
    ("8",  "PP-V",     "ProximalPhalanx", 5),
    ("9",  "MP-III",   "MiddlePhalanx",   3),
    ("10", "MP-V",     "MiddlePhalanx",   5),
    ("11", "DP-I",     "DistalPhalanx",   1),
    ("12", "DP-III",   "DistalPhalanx",   3),
    ("13", "DP-V",     "DistalPhalanx",   5),
]

FCOLOR = {1: (220, 40, 40), 2: (240, 150, 0), 3: (0, 160, 90),
          4: (40, 110, 230), 5: (170, 60, 220)}
NONE_COLOR = (215, 45, 45)

# RUS 骨名 -> 等级分类器的关节名
#   arthrosis/ 里按关节分文件夹：DIP / DIPFirst / MIP / PIP / PIPFirst /
#                                MCP / MCPFirst / Radius / Ulna
#   "First" = 第 1 指（拇指）
#
#   映射依据：
#     Radius / Ulna        -> 同名
#     MC（掌骨区）          -> MCP / MCPFirst
#     PP（近节指骨）        -> PIP / PIPFirst   （近节指骨对应的指间关节区）
#     MP（中节指骨）        -> MIP
#     DP（远节指骨）        -> DIP / DIPFirst
CLS_JOINT = {
    "Radius": "Radius", "Ulna": "Ulna",
    "MC-I": "MCPFirst", "MC-III": "MCP", "MC-V": "MCP",
    "PP-I": "PIPFirst", "PP-III": "PIP", "PP-V": "PIP",
    "MP-III": "MIP", "MP-V": "MIP",
    "DP-I": "DIPFirst", "DP-III": "DIP", "DP-V": "DIP",
}

DEFAULT_CLS_DIR = PROJECT / "outputs" / "cls"

CROP_SIZE = 128          # 分类模型的输入尺寸
CROP_PAD = 0.12          # 裁剪时向外扩一点，保证包含关节全部


# ============================================================
# 手指分组（和 stage5_select13.py 里的逻辑一致）
# ============================================================
def _ctr(d):
    x1, y1, x2, y2 = d["bbox"]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def group_fingers(dets):
    """把检测框分配到 5 根手指。

    关键：用【按垂直坐标排序后顺序配对】，而不是"离哪个锚点最近"。
    因为手指往指尖方向会扇形展开，"谁离谁近"会失效，"谁在谁左边"不会。
    """
    from collections import defaultdict
    by = defaultdict(list)
    for d in dets:
        by[d["class"]].append(d)

    mcps = by.get("MCP", [])
    mcp_first = by.get("MCPFirst", [])
    pps, mps, dps = (by.get("ProximalPhalanx", []),
                     by.get("MiddlePhalanx", []),
                     by.get("DistalPhalanx", []))
    wrist_pts = [_ctr(d) for d in (by.get("Radius", []) + by.get("Ulna", []))]

    if not (mcp_first and len(mcps) >= 3 and pps and dps and wrist_pts):
        return None

    wrist = np.mean(wrist_pts, axis=0)
    palm = np.mean([_ctr(d) for d in mcps + mcp_first], axis=0)
    axis = palm - wrist
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    perp = np.array([-axis[1], axis[0]])
    p_of = lambda d: float(np.dot(_ctr(d) - wrist, perp))

    thumb = mcp_first[0]
    flip = p_of(thumb) > np.mean([p_of(m) for m in mcps])
    order = lambda lst, k: sorted(lst, key=p_of, reverse=flip)[:k]

    F = {1: [thumb]}
    for fid, m in zip((2, 3, 4, 5), order(mcps, 4)):
        F[fid] = [m]
    pp = order(pps, None)
    if pp:
        F[1].append(pp[0])
        for fid, x in zip((2, 3, 4, 5), pp[1:5]):
            F[fid].append(x)
    for fid, x in zip((2, 3, 4, 5), order(mps, None)):
        F[fid].append(x)
    dp = order(dps, None)
    if dp:
        F[1].append(dp[0])
        for fid, x in zip((2, 3, 4, 5), dp[1:5]):
            F[fid].append(x)
    return F


def select_13(fingers, dets):
    """按 RUS-CHN 取 13 块骨。"""
    out = []
    for num, rus_name, cls, fid in RUS13:
        cands = ([d for d in fingers.get(fid, []) if d["class"] == cls]
                 if fid is not None
                 else [d for d in dets if d["class"] == cls])
        best = max(cands, key=lambda d: d["conf"]) if cands else None
        out.append((num, rus_name, cls, fid, best))
    return out


# ============================================================
# 画图
# ============================================================
def _font(size):
    for name in ("msyhbd.ttc", "msyh.ttc", "simhei.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_detections(img, dets):
    im = img.copy()
    dr = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(3, int(max(W, H) / 350))
    f = _font(max(20, W // 60))
    for i, d in enumerate(sorted(dets, key=lambda x: -x["conf"]), 1):
        x1, y1, x2, y2 = d["bbox"]
        dr.rectangle([x1, y1, x2, y2], outline=(60, 140, 255), width=lw)
        dr.text((x1 + 5, y1 + 5), f"{d['class'][:8]} {d['conf']:.2f}",
                fill=(60, 140, 255), font=f)
    dr.text((10, 10), f"检测到 {len(dets)} 个关节", fill=(255, 60, 60), font=f)
    return im


def draw_rus13(img, fingers, picked):
    im = img.copy()
    dr = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(4, int(max(W, H) / 320))
    f = _font(max(22, W // 55))

    # 每根手指连一条线，让归属关系一目了然
    for fid, ds in fingers.items():
        pts = [tuple(_ctr(d)) for d in ds]
        pts.sort(key=lambda q: -q[1])
        if len(pts) > 1:
            dr.line(pts, fill=FCOLOR[fid], width=max(2, lw // 3))

    for num, rus_name, cls, fid, det in picked:
        if det is None:
            continue
        x1, y1, x2, y2 = det["bbox"]
        col = FCOLOR.get(fid, NONE_COLOR)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        tag = f"{num}.{rus_name} {det['conf']:.2f}"
        tb = dr.textbbox((0, 0), tag, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 10 if y1 - th - 10 > 0 else y2 + 5
        dr.rectangle([x1, ty, x1 + tw + 12, ty + th + 8], fill=col)
        dr.text((x1 + 6, ty + 4), tag, fill=(255, 255, 255), font=f)

    legend = ["13 个 RUS-CHN 评分骨",
              "红 拇指(F1)  橙 食指(F2)",
              "绿 中指(F3)  蓝 无名指(F4)",
              "紫 小指(F5)"]
    lh = len(legend) * (f.size + 10) + 12
    dr.rectangle([8, 8, 8 + int(W * 0.30), lh], fill=(255, 255, 255))
    for i, t in enumerate(legend):
        dr.text((18, 14 + i * (f.size + 10)), t,
                fill=(20, 20, 20), font=f if i == 0 else _font(max(16, W // 75)))
    return im


def crop_bone(img, det, size=CROP_SIZE, pad=CROP_PAD):
    """按框裁剪 + 外扩 + 缩放到固定尺寸（给分类模型用）。"""
    W, H = img.size
    x1, y1, x2, y2 = det["bbox"]
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * pad); y1 = max(0, y1 - h * pad)
    x2 = min(W, x2 + w * pad); y2 = min(H, y2 + h * pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((int(x1), int(y1), int(x2), int(y2))).resize(
        (size, size), Image.LANCZOS)


# ============================================================
# 等级分类器
# ============================================================
class GradeClassifier:
    """按需加载 9 个关节的等级分类器（第一次用到才载入，避免全载）。"""

    def __init__(self, cls_dir, device="cpu"):
        self.dir = Path(cls_dir)
        self.device = device
        self.cache = {}
        self.warned = False

    def available(self):
        return self.dir.exists() and any(self.dir.glob("*.pt"))

    def _load(self, joint):
        if joint in self.cache:
            return self.cache[joint]
        p = self.dir / f"{joint}.pt"
        if not p.exists():
            self.cache[joint] = None
            return None
        from stage6_train_grade_cls import SmallCNN
        ck = torch.load(p, map_location=self.device, weights_only=False)
        m = SmallCNN(ck["n_classes"]).to(self.device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        self.cache[joint] = (m, ck)
        return self.cache[joint]

    def predict(self, joint, crop_img):
        """返回 (等级名, 置信度, 全部等级概率, 该关节的等级数) 或 None。"""
        got = self._load(joint)
        if got is None:
            return None
        model, ck = got
        size = ck["img_size"]
        x = np.asarray(crop_img.convert("L").resize((size, size),
                                                    Image.BILINEAR),
                       dtype=np.float32) / 255.0
        # ★ 归一化必须用训练时存下来的 mean/std
        x = (x - ck["norm_mean"]) / ck["norm_std"]
        t = torch.from_numpy(x)[None, None].to(self.device)
        with torch.no_grad():
            prob = model(t).softmax(1)[0].cpu().numpy()
        i = int(prob.argmax())
        grades = ck["grades"]
        return (grades[i], float(prob[i]),
                {grades[k]: float(prob[k]) for k in range(len(grades))},
                ck["n_classes"])


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="输入的手部 X 光片路径")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--conf", type=float, default=0.25,
                    help="检测置信度阈值（默认 0.25）")
    ap.add_argument("--crop-size", type=int, default=CROP_SIZE)
    ap.add_argument("--cls-dir", default=str(DEFAULT_CLS_DIR),
                    help="等级分类器目录（stage6 训练的 <关节>.pt）")
    ap.add_argument("--no-cls", action="store_true",
                    help="跳过等级分类，只做检测 + 筛选 + 裁剪")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"找不到图片：{img_path}")
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"找不到检测权重：{weights}\n"
            "先训练：python src/stage3_train.py --epochs 100 --patience 30 --name full"
        )

    out_dir = Path(args.out) / img_path.stem
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"输入图片：{img_path}")
    print(f"检测权重：{weights.name}")
    print(f"输出目录：{out_dir}")
    print("=" * 70, flush=True)

    img = Image.open(img_path).convert("RGB")
    print(f"\n图片尺寸：{img.size[0]} x {img.size[1]}")

    # ---------- ① 检测 ----------
    model = YOLO(str(weights))
    r = model.predict(str(img_path), conf=args.conf, imgsz=640,
                      device=0, verbose=False)[0]

    names = r.names
    dets = []
    if r.boxes is not None and len(r.boxes):
        for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append({"class": names[int(cls)], "conf": float(cf),
                         "bbox": [x1, y1, x2, y2]})

    print(f"\n① 检测到 {len(dets)} 个关节")
    from collections import Counter
    for k, v in Counter(d["class"] for d in dets).most_common():
        print(f"     {k:<18} {v}")

    draw_detections(img, dets).save(out_dir / "01_detections.jpg",
                                    quality=92)

    if not dets:
        raise SystemExit("\n没有检测到任何关节，无法继续。")

    # ---------- ② 手指分组 + 筛选 13 ----------
    fingers = group_fingers(dets)
    if fingers is None:
        raise SystemExit(
            "\n几何信息不足，无法分组手指。\n"
            "需要至少检测到：桡骨或尺骨、MCPFirst、3 个以上 MCP、"
            "以及近节/远节指骨。"
        )
    picked = select_13(fingers, dets)
    n_ok = sum(1 for *_x, d in picked if d is not None)

    print(f"\n② 筛选 13 个 RUS-CHN 评分骨：找到 {n_ok}/13")

    # ---------- ③ 等级分类 ----------
    clf = None
    if not args.no_cls:
        clf = GradeClassifier(args.cls_dir,
                              "cuda" if torch.cuda.is_available() else "cpu")
        if clf.available():
            print(f"\n③ 发育等级分类（模型：{Path(args.cls_dir)}）")
        else:
            print(f"\n③ 跳过等级分类 —— 在 {args.cls_dir} 里没找到模型文件")
            print("     训练：python src/stage6_train_grade_cls.py")
            clf = None

    print(f"\n     {'#':>3}  {'RUS 名称':<9}{'数据类别':<18}{'手指':>5}"
          f"{'检测':>8}{'等级':>6}{'等级置信度':>12}   {'裁剪图'}")
    print("     " + "-" * 74)

    rows = []
    for num, rus_name, cls, fid, det in picked:
        f_txt = f"F{fid}" if fid else "—"
        row = {"num": num, "rus_name": rus_name, "class": cls,
               "finger": fid, "det_conf": None, "bbox": None,
               "crop": None, "grade": None, "grade_conf": None,
               "grade_probs": None}
        if det is None:
            print(f"     {num:>3}  {rus_name:<9}{cls:<18}{f_txt:>5}"
                  f"{'—':>8}{'—':>6}{'—':>12}   ✗ 未检出")
            rows.append(row)
            continue

        crop = crop_bone(img, det, args.crop_size)
        fn = f"{int(num):02d}_{rus_name}.png"
        if crop is not None:
            crop.save(out_dir / "crops" / fn)

        row["det_conf"] = round(det["conf"], 4)
        row["bbox"] = [round(v, 1) for v in det["bbox"]]
        row["crop"] = f"crops/{fn}" if crop else None

        g_txt, gc_txt = "—", ""
        if clf is not None and crop is not None:
            joint = CLS_JOINT.get(rus_name)
            res = clf.predict(joint, crop) if joint else None
            if res is not None:
                grade, gconf, gprobs, n_cls = res
                row["grade"] = grade
                row["grade_conf"] = round(gconf, 4)
                row["grade_n_classes"] = n_cls
                row["grade_probs"] = {k: round(v, 4)
                                      for k, v in gprobs.items()}
                g_txt, gc_txt = f"L{grade}", f"{gconf:.3f}"

        print(f"     {num:>3}  {rus_name:<9}{cls:<18}{f_txt:>5}"
              f"{det['conf']:>8.3f}{g_txt:>6}{gc_txt:>12}   {fn}")
        rows.append(row)

    # ---------- 成熟度汇总 ----------
    scored = [r for r in rows if r["grade"] is not None]
    maturity = None
    if scored:
        # 归一化：每块骨的等级除以【该关节的最高等级】，再取平均。
        # 这样不同关节（10~14 级）的等级可以放在一起比较。
        vals = [int(r["grade"]) / max(r["grade_n_classes"], 1)
                for r in scored]
        maturity = float(np.mean(vals))
        detail = "  ".join(f"{r['rus_name']}=L{r['grade']}" for r in scored)
        print(f"\n③ 成熟度")
        print(f"     已判定 {len(scored)}/13 块骨")
        print(f"     等级：{detail}")
        print(f"     归一化成熟度指数：{maturity:.3f}"
              f"   （0=最不成熟，1=完全成熟；只是横向比较用，不是骨龄）")

    draw_rus13(img, fingers, picked).save(out_dir / "02_rus13.jpg",
                                          quality=92)

    # ---------- 报告 ----------
    report = {
        "image": str(img_path),
        "image_size": list(img.size),
        "n_detections": len(dets),
        "n_rus13_found": n_ok,
        "n_grades_predicted": len(scored),
        "maturity_index": (round(maturity, 4) if maturity is not None
                           else None),
        "rus13": rows,
        "detections": [{"class": d["class"], "conf": round(d["conf"], 4),
                        "bbox": [round(v, 1) for v in d["bbox"]]}
                       for d in dets],
        "bone_age": None,
        "bone_age_note": (
            "骨龄数字需要 CHN-05 官方评分表（把 13 个等级加权求和后查表）。"
            "本仓库没有这份表，所以不给出数字 —— 编一个假的比不给更糟。"
            "已给出的 maturity_index 是 13 块骨等级除以各自最高等级的均值，"
            "只能用来横向比较，不是骨龄。"
        ),
        "caveat": (
            "等级分类器训练用的 arthrosis/ 切图来自【另一批 X 光片】，"
            "与检测用的 881 张不是同一批，且裁剪方式不同 —— "
            "存在域偏移，等级预测仅供参考。"
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"完成！产物都在：{out_dir}")
    print(f"  01_detections.jpg   检测结果")
    print(f"  02_rus13.jpg        13 个评分骨（按手指着色）")
    print(f"  crops/              13 个裁剪图（{args.crop_size}x{args.crop_size}）")
    print(f"  report.json         结构化结果")
    print()
    if scored:
        print(f"  ✅ 已判定 {len(scored)}/13 块骨的发育等级，"
              f"归一化成熟度指数 {maturity:.3f}")
        print()
        print("  ⚠️⚠️ 等级预测存在【域偏移】，请务必看这一条：")
        print("     分类器是用 arthrosis/ 的切图训练的，那些切图【紧贴关节】")
        print("     （只有骨骺 + 干骺端）；而这里的裁剪来自检测框，是【整根骨】。")
        print("     两者构图不同 → 模型倾向于把整根成熟的骨判成高等级。")
        print("     实测：13 块骨几乎全被判成接近最高级。")
        print("     ⇒ 这个等级只能当演示，不能当结论。")
    print("  ⚠️ 没有输出骨龄数字 —— 还差【CHN-05 官方评分表】这一步。")
    print("=" * 70)


if __name__ == "__main__":
    main()
