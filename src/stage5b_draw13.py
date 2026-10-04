"""画「13 个关节筛选结果」标注图（带图例、编号 1-13）。

产出：只画选中的 13 个框，按手指编号着色，标注 RUS-CHN 名称和置信度。
这样一眼能看出食指（F2）和无名指（F4）只有连线、没有框 ——
因为 RUS-CHN 不评分这两根手指。

说明：数据里没有单独的「掌骨」类别，只有掌指关节（MCP / MCPFirst）。
     RUS 评分掌骨时看的是掌骨头部的骨骺，而掌指关节的框正好覆盖这个区域，
     所以用 MCP 近似 MC（第 1/3/5 掌骨）是合理的 —— 但这是近似。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 13 块骨：(RUS-CHN 名称, 数据里的类别, 手指编号)
RUS13 = [
    ("1 Radius",   "Radius",          None),
    ("2 Ulna",     "Ulna",            None),
    ("3 MC-I",     "MCPFirst",        1),
    ("4 MC-III",   "MCP",             3),
    ("5 MC-V",     "MCP",             5),
    ("6 PP-I",     "ProximalPhalanx", 1),
    ("7 PP-III",   "ProximalPhalanx", 3),
    ("8 PP-V",     "ProximalPhalanx", 5),
    ("9 MP-III",   "MiddlePhalanx",   3),
    ("10 MP-V",    "MiddlePhalanx",   5),
    ("11 DP-I",    "DistalPhalanx",   1),
    ("12 DP-III",  "DistalPhalanx",   3),
    ("13 DP-V",    "DistalPhalanx",   5),
]

FCOLOR = {1: (220, 40, 40), 2: (240, 150, 0), 3: (0, 160, 90),
          4: (40, 110, 230), 5: (170, 60, 220)}


def ctr(d):
    x1, y1, x2, y2 = d["bbox"]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def build_fingers(dets):
    """和 stage5_select13.py 里的分组逻辑一致（按顺序配对）。"""
    from collections import defaultdict
    by = defaultdict(list)
    for d in dets:
        by[d["class"]].append(d)

    mcps, mcp_first = by.get("MCP", []), by.get("MCPFirst", [])
    pps, mps = by.get("ProximalPhalanx", []), by.get("MiddlePhalanx", [])
    dps = by.get("DistalPhalanx", [])
    wrist_pts = [ctr(d) for d in (by.get("Radius", []) + by.get("Ulna", []))]
    if not (mcp_first and len(mcps) >= 3 and pps and dps and wrist_pts):
        return None

    wrist = np.mean(wrist_pts, axis=0)
    palm = np.mean([ctr(d) for d in mcps + mcp_first], axis=0)
    axis = palm - wrist
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    perp = np.array([-axis[1], axis[0]])
    p_of = lambda d: float(np.dot(ctr(d) - wrist, perp))

    thumb = mcp_first[0]
    flip = p_of(thumb) > np.mean([p_of(m) for m in mcps])
    order = lambda lst, k: sorted(lst, key=p_of, reverse=flip)[:k]

    F = {1: [thumb]}
    for fid, m in zip((2, 3, 4, 5), order(mcps, 4)):
        F[fid] = [m]
    pp = order(pps, None)
    F[1].append(pp[0])
    for fid, x in zip((2, 3, 4, 5), pp[1:5]):
        F[fid].append(x)
    for fid, x in zip((2, 3, 4, 5), order(mps, None)):
        F[fid].append(x)
    dp = order(dps, None)
    F[1].append(dp[0])
    for fid, x in zip((2, 3, 4, 5), dp[1:5]):
        F[fid].append(x)
    return F


def pick13(F, dets):
    out = []
    for _name, cls, fid in RUS13:
        cands = ([d for d in F.get(fid, []) if d["class"] == cls]
                 if fid else [d for d in dets if d["class"] == cls])
        out.append(max(cands, key=lambda d: d["conf"]) if cands else None)
    return out


def draw_one(stem, split, dets, save_to):
    F = build_fingers(dets)
    if F is None:
        return False
    picked = pick13(F, dets)
    if any(p is None for p in picked):
        return False

    p = Path(r"D:\AAA\bone-age-yolo") / "data" / "bone_age" / "images" / split / f"{stem}.png"
    img = Image.open(p).convert("RGB")
    W, H = img.size
    dr = ImageDraw.Draw(img)
    lw = max(4, int(max(W, H) / 320))

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", max(22, W // 55))
        fbig = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", max(26, W // 45))
    except Exception:
        font = fbig = ImageFont.load_default()

    # 先画一根手指的整体连线（让归属关系一目了然）
    for fid, ds in F.items():
        pts = [tuple(ctr(d)) for d in ds]
        pts.sort(key=lambda q: -q[1])          # 从上到下
        if len(pts) > 1:
            dr.line(pts, fill=FCOLOR[fid], width=max(2, lw // 3))

    # 只画选中的 13 个
    for i, (det, (name, _cls, fid)) in enumerate(zip(picked, RUS13), 1):
        x1, y1, x2, y2 = det["bbox"]
        col = FCOLOR.get(fid, (220, 40, 40))
        dr.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        tag = f"{name} {det['conf']:.2f}"
        tb = dr.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 8 if y1 - th - 8 > 0 else y2 + 4
        dr.rectangle([x1, ty, x1 + tw + 10, ty + th + 8], fill=col)
        dr.text((x1 + 5, ty + 4), tag, fill=(255, 255, 255), font=font)

    # 左上角图例
    legend = ["13 个 RUS-CHN 评分骨", "颜色 = 手指编号",
              "F1 拇指  F2 食指  F3 中指", "F4 无名指  F5 小指"]
    lh = (len(legend) + 1) * (font.size + 10)
    dr.rectangle([6, 6, 6 + int(W * 0.36), 12 + lh], fill=(255, 255, 255))
    for i, t in enumerate(legend):
        dr.text((16, 12 + i * (font.size + 10)), t,
                fill=(20, 20, 20) if i == 0 else (60, 60, 60),
                font=fbig if i == 0 else font)

    img.thumbnail((1200, 1200))
    img.save(save_to)
    return True


if __name__ == "__main__":
    import json
    OUT = Path(r"D:\AAA\bone-age-yolo\outputs\joint_filter")
    data = json.loads((Path(r"D:\AAA\bone-age-yolo")
                       / "outputs/detections/all_detections.json")
                      .read_text(encoding="utf-8"))
    # 挑几张：1 张之前分组失败的 + 3 张普通的
    want = ["14714", "14757", "1659", "1837"]
    for stem in want:
        if stem not in data:
            continue
        ok = draw_one(stem, data[stem]["split"], data[stem]["detections"],
                      OUT / f"rus13_{stem}.png")
        print(f"  {stem}: {'OK' if ok else '失败'} -> rus13_{stem}.png")
