"""阶段 5：从 21 个检测框里筛出 13 个 RUS-CHN 评分骨。

RUS-CHN（CHN-05 法）评分的是这 13 块骨
--------------------------------------------
    桡骨 Radius                      × 1
    尺骨 Ulna                        × 1
    第 1 / 3 / 5 掌骨  （MC I, III, V）    × 3
    第 1 / 3 / 5 近节指骨（PP I, III, V）  × 3
    第 3 / 5 中节指骨  （MP III, V）       × 2
    第 1 / 3 / 5 远节指骨（DP I, III, V）  × 3
                                    --------
                                      13

注意：只取第 1、3、5 指（拇指 / 中指 / 小指），第 2、4 指不评分。

这一步真正的难点在哪
--------------------
模型给的是「7 个类别、21 个实例」，
而 RUS-CHN 要的是「**哪根手指**的哪类关节」——
中间缺了「手指编号」这个信息。

如果只按类别名去查，会踩一个很隐蔽的坑：
    MCP 有 4 个实例、ProximalPhalanx 有 5 个实例。
    如果按「类别名」逐个取，每次都会命中【同一个框】，
    于是 13 个位置里只有 6 个是不同的关节，其余 7 个是重复的。
    **而且程序不报错、数量也确实等于 13、后面也能拼起来跑** ——
    是个典型的静默错误，最后表现为骨龄分数把同一个关节算了好几遍。

正确做法：先做几何分组，把 21 个框分配到 5 根手指，再按手指编号取。
--------------------------------------------------------------------------
    ① 手轴 = 腕部中心(桡骨+尺骨) → 掌部中心(所有掌指关节)
    ② 垂直轴 = 手轴旋转 90°
    ③ 拇指（手指 1）由 MCPFirst 直接标识
    ④ 另外 4 个 MCP 是第 2/3/4/5 指的【锚点】
    ⑤ 每块指骨（近/中/远）按垂直坐标排序后，**按顺序**配对给各手指
       （不能用"离哪个锚点最近" —— 见 group_fingers() 的说明）
    ⑥ 自检：每根手指的构成必须恰好是
         手指 1（拇指）：MCPFirst + PP + DP          （拇指没有中节）
         手指 2~5      ：MCP + PP + MP + DP
       任何一张图不满足 —— 就说明分组错了，要报出来

这个自检是关键：它把"静默错误"变成"会被打印出来的错误"。
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "outputs" / "detections" / "all_detections.json"
IMG_DIRS = {
    "train": PROJECT / "data" / "bone_age" / "images" / "train",
    "val": PROJECT / "data" / "bone_age" / "images" / "val",
}
OUT = PROJECT / "outputs" / "joint_filter"
REPORT = PROJECT / "outputs" / "data_report"

# RUS-CHN 的 13 块骨：(类别, 手指编号)
#   手指编号：1=拇指, 2=食指, 3=中指, 4=无名指, 5=小指
RUS_CHN = [
    ("Radius", None),
    ("Ulna", None),
    ("MCPFirst", 1),        # 第 1 掌骨区
    ("MCP", 3),             # 第 3 掌骨
    ("MCP", 5),             # 第 5 掌骨
    ("ProximalPhalanx", 1),
    ("ProximalPhalanx", 3),
    ("ProximalPhalanx", 5),
    ("MiddlePhalanx", 3),
    ("MiddlePhalanx", 5),
    ("DistalPhalanx", 1),
    ("DistalPhalanx", 3),
    ("DistalPhalanx", 5),
]

# 每根手指应有的构成（自检用）
EXPECTED = {
    1: ["MCPFirst", "ProximalPhalanx", "DistalPhalanx"],
    2: ["MCP", "ProximalPhalanx", "MiddlePhalanx", "DistalPhalanx"],
    3: ["MCP", "ProximalPhalanx", "MiddlePhalanx", "DistalPhalanx"],
    4: ["MCP", "ProximalPhalanx", "MiddlePhalanx", "DistalPhalanx"],
    5: ["MCP", "ProximalPhalanx", "MiddlePhalanx", "DistalPhalanx"],
}


def ctr(d):
    x1, y1, x2, y2 = d["bbox"]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def group_fingers(dets):
    """把检测框分配到 5 根手指。

    为什么不用「离哪个 MCP 最近」（第一版就是这么做，自检通过率只有 1.4%）
    -----------------------------------------------------------------
    手指往指尖方向会【扇形展开】—— 中指远节的垂直坐标可能比无名指的
    MCP 更"靠边"，于是被分到错误的手指上。

    正确思路：用【顺序】配对，而不是用【距离】
    ----------------------------------------
    虽然手指会展开，但「沿垂直轴的先后顺序」在每一排（掌指 / 近节 /
    中节 / 远节）都是保持一致的：
        掌指排： F2 < F3 < F4 < F5   （沿拇指对侧方向）
        远节排： F2 < F3 < F4 < F5   （顺序不变，只是间距变大了）

    所以：把每一排的框按垂直坐标排序，再按顺序配对给 F2~F5 即可。
    拇指的那一块（MCPFirst 以及对应的近节/远节）在最极端的一端。
    """
    by_cls = defaultdict(list)
    for d in dets:
        by_cls[d["class"]].append(d)

    radius, ulna = by_cls.get("Radius", []), by_cls.get("Ulna", [])
    mcps, mcp_first = by_cls.get("MCP", []), by_cls.get("MCPFirst", [])
    pps = by_cls.get("ProximalPhalanx", [])
    mps = by_cls.get("MiddlePhalanx", [])
    dps = by_cls.get("DistalPhalanx", [])

    if not mcp_first or len(mcps) < 3 or not pps or not dps:
        return None
    wrist_pts = [ctr(d) for d in (radius + ulna)]
    if not wrist_pts:
        return None

    wrist = np.mean(wrist_pts, axis=0)
    palm = np.mean([ctr(d) for d in (mcps + mcp_first)], axis=0)
    axis = palm - wrist
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return None
    axis /= n
    perp = np.array([-axis[1], axis[0]])

    def p_of(d):
        return float(np.dot(ctr(d) - wrist, perp))

    thumb = mcp_first[0]
    tp = p_of(thumb)
    mcp_mean = float(np.mean([p_of(m) for m in mcps]))

    # 拇指在高 p 端还是低 p 端 —— 决定 F2..F5 的排序方向
    flip = tp > mcp_mean

    def ordered(lst, keep):
        """按垂直坐标排序（方向由 flip 决定），取前 keep 个。"""
        s = sorted(lst, key=p_of, reverse=flip)
        return s[:keep] if keep is not None else s

    fingers = {1: [thumb]}

    # 掌指排：4 个 MCP -> F2..F5
    for fid, m in zip((2, 3, 4, 5), ordered(mcps, 4)):
        fingers[fid] = [m]

    # 近节排：5 个（含拇指）-> 极端那个给 F1，其余按序给 F2..F5
    pp_sorted = ordered(pps, None)
    fingers[1].append(pp_sorted[0])
    for fid, pp in zip((2, 3, 4, 5), pp_sorted[1:5]):
        fingers[fid].append(pp)

    # 中节排：4 个（拇指没有中节）-> 按序给 F2..F5
    for fid, mp in zip((2, 3, 4, 5), ordered(mps, None)):
        fingers[fid].append(mp)

    # 远节排：5 个（含拇指）-> 同近节
    dp_sorted = ordered(dps, None)
    fingers[1].append(dp_sorted[0])
    for fid, dp in zip((2, 3, 4, 5), dp_sorted[1:5]):
        fingers[fid].append(dp)

    return fingers


def check_fingers(fingers):
    """自检：返回问题列表（空表示全部正常）。"""
    problems = []
    for fid, ds in fingers.items():
        got = sorted(d["class"] for d in ds)
        want = sorted(EXPECTED[fid])
        if got != want:
            problems.append(f"手指{fid}: 期望 {want}，实际 {got}")
    if len(fingers) != 5:
        problems.append(f"只分出了 {len(fingers)} 根手指")
    return problems


def select_13(fingers, dets):
    """按 RUS-CHN 取 13 块骨。"""
    picked = []
    for cls, fid in RUS_CHN:
        if fid is None:
            cands = [d for d in dets if d["class"] == cls]
        else:
            cands = [d for d in fingers.get(fid, []) if d["class"] == cls]
        if not cands:
            picked.append(None)
            continue
        picked.append(max(cands, key=lambda d: d["conf"]))
    return picked


def main():
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    problems_all = []
    per_image = {}

    for stem, item in data.items():
        dets = item["detections"]
        fingers = group_fingers(dets)
        if fingers is None:
            stats["几何信息不足"] += 1
            continue

        probs = check_fingers(fingers)
        if probs:
            stats["手指分组异常"] += 1
            problems_all.append((stem, probs))
        else:
            stats["分组正确"] += 1

        picked = select_13(fingers, dets)
        n_ok = sum(1 for p in picked if p is not None)
        stats[f"选中{n_ok}个"] += 1
        per_image[stem] = {
            "split": item["split"],
            "finger_check": "ok" if not probs else probs,
            "selected": [None if p is None else p["class"] for p in picked],
            "n_selected": n_ok,
        }

    total = len(data)
    print("=" * 76)
    print(f"阶段 5：13 关节筛选（全部 {total} 张）")
    print("=" * 76)

    print(f"\n【手指分组自检】")
    print(f"  分组正确        {stats['分组正确']:>5}  "
          f"({stats['分组正确'] / total * 100:.1f}%)")
    print(f"  分组异常        {stats['手指分组异常']:>5}  "
          f"({stats['手指分组异常'] / total * 100:.1f}%)")
    print(f"  几何信息不足    {stats['几何信息不足']:>5}")

    print(f"\n【选出的关节数】")
    for k in sorted(k for k in stats if k.startswith("选中")):
        print(f"  {k:<12} {stats[k]:>5}")

    print(f"\n【分组异常的例子（前 8 个）】")
    for stem, probs in problems_all[:8]:
        print(f"  {stem}:")
        for p in probs:
            print(f"     {p}")

    # ---------- 保存逐图结果 ----------
    (REPORT / "select13_report.json").write_text(
        json.dumps({"rus_chn": RUS_CHN, "stats": dict(stats),
                    "per_image": per_image},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 可视化：4 张样例，标出 13 个关节和手指编号 ----------
    FCOLOR = {1: (255, 60, 60), 2: (255, 170, 0), 3: (0, 190, 120),
              4: (60, 140, 255), 5: (200, 80, 255)}
    good = [s for s, v in per_image.items() if v["finger_check"] == "ok"
            and v["n_selected"] == 13][:4]
    print(f"\n【可视化】选了 {len(good)} 张分组正确的图")
    for stem in good:
        item = data[stem]
        fingers = group_fingers(item["detections"])
        picked = select_13(fingers, item["detections"])
        picked_ids = {id(p) for p in picked if p}

        p = IMG_DIRS[item["split"]] / f"{stem}.png"
        img = Image.open(p).convert("RGB")
        dr = ImageDraw.Draw(img)
        lw = max(3, int(max(img.size) / 350))

        for fid, ds in fingers.items():
            col = FCOLOR[fid]
            for d in ds:
                x1, y1, x2, y2 = d["bbox"]
                dr.rectangle([x1, y1, x2, y2], outline=col, width=lw)
                tag = f"F{fid} {d['class'][:6]}"
                dr.text((x1 + 5, y1 + 5), tag, fill=col)

        img.thumbnail((1100, 1100))
        sp = OUT / f"select13_{stem}.png"
        img.save(sp)
        print(f"  {stem} [{item['split']}] -> {sp.name}"
              f"（选中 {sum(1 for x in picked if x)}/13）")

    print(f"\n  报告已存：{REPORT / 'select13_report.json'}")


if __name__ == "__main__":
    main()
