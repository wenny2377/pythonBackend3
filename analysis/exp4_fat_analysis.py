import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

from exp_config import (
    MONGO_URI, DB_BASELINE, ADL_LABELS, USERS,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

FAT        = 5
FAT_VALUES = [3, 5, 10, 15]
MAX_DAY    = 21
NO_RECORD  = {"Walking", "Standing", "StandUp", "PickingUp", "PuttingDown"}


def load_episodes(db_name):
    db = MongoClient(MONGO_URI)[db_name]
    return list(db.eval_logs.find(
        {"spatial_action": {"$exists": True, "$nin": list(NO_RECORD)},
         "zone_label":     {"$exists": True, "$ne": ""},
         "virtual_day":    {"$lte": MAX_DAY, "$exists": True}},
        {"user": 1, "spatial_action": 1, "ground_truth": 1,
         "virtual_day": 1, "virtual_hour": 1}
    ).sort([("virtual_day", 1), ("virtual_hour", 1)]))


def compute_habits_for_fat(episodes, uid, fat):
    weight    = defaultdict(float)
    first_day = {}
    for ep in [e for e in episodes if e.get("user") == uid]:
        day  = ep.get("virtual_day", 0)
        gt   = ep.get("ground_truth", "")
        pred = ep.get("spatial_action") or ep.get("vlm_output", "")
        if gt and gt == pred and gt in ADL_LABELS:
            weight[gt] += 1
            if weight[gt] >= fat and gt not in first_day:
                first_day[gt] = day
    return first_day


def plot_fat_sensitivity_table(episodes, save_path):
    rows = []
    for fat in FAT_VALUES:
        mom      = compute_habits_for_fat(episodes, "User_Mom", fat)
        dad      = compute_habits_for_fat(episodes, "User_Dad", fat)
        mom_last = max(mom.values()) if mom else MAX_DAY
        dad_last = max(dad.values()) if dad else MAX_DAY
        not_mom  = [a for a in ADL_LABELS
                    if a not in mom
                    and any(e.get("user") == "User_Mom" and
                            e.get("ground_truth") == a
                            for e in episodes)]
        rows.append([
            f"FAT = {fat}",
            str(len(mom)),
            str(len(dad)),
            f"Day {mom_last}",
            f"Day {dad_last}",
            ", ".join(not_mom) if not_mom else "—",
        ])

    col_headers = ["FAT", "Mom\nHabits", "Dad\nHabits",
                   "Mom Last\nConfirmed", "Dad Last\nConfirmed",
                   "Not Reached (Mom)"]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    col_widths = [0.10, 0.10, 0.10, 0.14, 0.14, 0.42]
    table = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.6)

    header_color = "#2C3E50"
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    highlight_row = FAT_VALUES.index(FAT)
    row_colors    = ["#EBF5FB", "#FDFEFE", "#EBF5FB", "#FDFEFE"]
    highlight     = "#D5F5E3"

    for i in range(len(FAT_VALUES)):
        for j in range(len(col_headers)):
            cell = table[i + 1, j]
            if i == highlight_row:
                cell.set_facecolor(highlight)
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(row_colors[i % 2])

    ax.set_title(
        f"Table 5.2  FAT Sensitivity Analysis  (Day 1–{MAX_DAY})\n"
        f"★ FAT = {FAT} selected — balances habit confirmation speed "
        f"and filtering of low-frequency behaviours\n"
        f"Note: FAT = 15 reduces Mom habits from 9 to 8 "
        f"because PhoneUse (12 correct recognitions) falls below the threshold",
        fontsize=FONT_TITLE - 2, fontweight="bold", pad=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def plot_personalization(episodes, save_path):
    weight = defaultdict(float)
    for ep in episodes:
        u    = ep.get("user", "")
        gt   = ep.get("ground_truth", "")
        pred = ep.get("spatial_action") or ep.get("vlm_output", "")
        if u and gt and gt == pred and gt in ADL_LABELS:
            weight[(u, gt)] += 1

    mom_habits = {a: w for (u, a), w in weight.items()
                  if u == "User_Mom" and w >= FAT}
    dad_habits = {a: w for (u, a), w in weight.items()
                  if u == "User_Dad" and w >= FAT}

    if not mom_habits and not dad_habits:
        print("[exp4] No personalization data")
        return

    mom_only = sorted(set(mom_habits) - set(dad_habits))
    shared   = sorted(set(mom_habits) & set(dad_habits))
    dad_only = sorted(set(dad_habits) - set(mom_habits))
    ordered  = mom_only + shared + dad_only

    n  = len(ordered)
    x  = np.arange(n)
    bw = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 5))

    mom_vals = [mom_habits.get(a, 0) for a in ordered]
    dad_vals = [dad_habits.get(a, 0) for a in ordered]

    bars_m = ax.bar(x - bw/2, mom_vals, bw, color=C["mom"],
                    alpha=0.85, label="User Mom", edgecolor="white")
    bars_d = ax.bar(x + bw/2, dad_vals, bw, color=C["dad"],
                    alpha=0.85, label="User Dad", edgecolor="white")

    for bar, v in list(zip(bars_m, mom_vals)) + list(zip(bars_d, dad_vals)):
        if v >= FAT:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.0f}", ha="center",
                    fontsize=FONT_ANNOT,
                    color=bar.get_facecolor())

    ax.axhline(FAT, color=C["threshold"], linestyle="--",
               lw=1.2, alpha=0.6, label=f"FAT = {FAT}")

    ymax = max(mom_vals + dad_vals) + 10
    ax.set_ylim(0, ymax)

    if mom_only:
        ax.axvspan(-0.5, len(mom_only) - 0.5, alpha=0.06, color=C["mom"])
        ax.text(len(mom_only) / 2 - 0.5, ymax * 0.92,
                "Mom only", ha="center", fontsize=FONT_ANNOT,
                color=C["mom"], fontweight="bold")

    if shared:
        s = len(mom_only) - 0.5
        e = s + len(shared)
        ax.axvspan(s, e, alpha=0.06, color="#888")
        ax.text((s + e) / 2, ymax * 0.92,
                "Shared", ha="center", fontsize=FONT_ANNOT,
                color="#555", fontweight="bold")

    if dad_only:
        s = len(mom_only) + len(shared) - 0.5
        e = n - 0.5
        ax.axvspan(s, e, alpha=0.06, color=C["dad"])
        ax.text((s + e) / 2, ymax * 0.92,
                "Dad only", ha="center", fontsize=FONT_ANNOT,
                color=C["dad"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Correctly Recognised Count", fontsize=FONT_AXIS)
    ax.set_title(
        f"Figure 5.6  Personalization — Stable Habit Profiles per User  (FAT = {FAT})\n"
        "Only correctly recognised episodes contribute to habit weight  "
        "(spatial_action == ground_truth)",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def save_summary(episodes, save_path):
    weight = defaultdict(float)
    for ep in episodes:
        u    = ep.get("user", "")
        gt   = ep.get("ground_truth", "")
        pred = ep.get("spatial_action") or ep.get("vlm_output", "")
        if u and gt and gt == pred and gt in ADL_LABELS:
            weight[(u, gt)] += 1

    lines = [
        "Experiment 4: FAT Sensitivity Analysis & Personalization",
        f"DB: {DB_BASELINE}  |  FAT = {FAT}  |  Day 1–{MAX_DAY}",
        "",
        "FAT Sensitivity:",
        f"{'FAT':<6} {'Mom Habits':>12} {'Dad Habits':>12}",
        "-" * 32,
    ]
    for fat in FAT_VALUES:
        mom    = compute_habits_for_fat(episodes, "User_Mom", fat)
        dad    = compute_habits_for_fat(episodes, "User_Dad", fat)
        marker = " ★" if fat == FAT else ""
        lines.append(f"{fat:<6} {len(mom):>12} {len(dad):>12}{marker}")

    lines += ["", f"Stable Habits at FAT={FAT}:"]
    for uid in USERS:
        habits = {a: w for (u, a), w in weight.items()
                  if u == uid and w >= FAT}
        lines += [f"  {uid}  ({len(habits)} habits):"]
        for a, w in sorted(habits.items(), key=lambda x: -x[1]):
            lines.append(f"    {a:<16} count={w:.0f}")

    mom = {a for (u, a), w in weight.items() if u == "User_Mom" and w >= FAT}
    dad = {a for (u, a), w in weight.items() if u == "User_Dad" and w >= FAT}
    lines += [
        "",
        f"  Shared   : {sorted(mom & dad)}",
        f"  Mom-only : {sorted(mom - dad)}",
        f"  Dad-only : {sorted(dad - mom)}",
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp4] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    episodes = load_episodes(DB_BASELINE)
    if not episodes:
        print(f"[exp4] No data in {DB_BASELINE}")
        return
    print(f"[exp4] {len(episodes)} episodes loaded")

    plot_fat_sensitivity_table(
        episodes,
        os.path.join(RESULTS_DIR, "exp4_fat_sensitivity.png"))

    plot_personalization(
        episodes,
        os.path.join(RESULTS_DIR, "exp4_personalization.png"))

    save_summary(
        episodes,
        os.path.join(RESULTS_DIR, "exp4_summary.txt"))


if __name__ == "__main__":
    main()