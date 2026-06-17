import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

from exp_config import (
    MONGO_URI, DB_BASELINE, ADL_LABELS, USERS,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    LINE_WIDTH, FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

FAT     = 5
MAX_DAY = 21
NO_RECORD = {"Walking", "Standing", "StandUp", "PickingUp", "PuttingDown"}

USER_STYLES = {
    "User_Mom": {"color": C["mom"], "marker": "o", "label": "User Mom"},
    "User_Dad": {"color": C["dad"], "marker": "s", "label": "User Dad"},
}


def load_episodes(db_name):
    db = MongoClient(MONGO_URI)[db_name]
    return list(db.eval_logs.find(
        {"spatial_action": {"$exists": True, "$nin": list(NO_RECORD)},
         "zone_label":     {"$exists": True, "$ne": ""},
         "virtual_day":    {"$lte": MAX_DAY, "$exists": True}},
        {"user": 1, "spatial_action": 1, "ground_truth": 1,
         "virtual_day": 1, "virtual_hour": 1}
    ).sort([("virtual_day", 1), ("virtual_hour", 1)]))


def compute_first_fat_day(episodes, uid, fat):
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


def plot_days_to_fat(episodes, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, uid, style in zip(
        axes,
        ["User_Mom", "User_Dad"],
        [USER_STYLES["User_Mom"], USER_STYLES["User_Dad"]]
    ):
        first_day   = compute_first_fat_day(episodes, uid, FAT)
        reached     = {a: d for a, d in first_day.items() if a in ADL_LABELS}
        not_reached = [a for a in ADL_LABELS
                       if a not in reached
                       and any(e.get("user") == uid and
                               e.get("ground_truth") == a
                               for e in episodes)]

        all_actions = sorted(reached, key=lambda a: reached[a])
        days_vals   = [reached[a] for a in all_actions]

        colors = []
        for d in days_vals:
            if d <= 7:
                colors.append("#27AE60")
            elif d <= 14:
                colors.append("#F5A623")
            else:
                colors.append("#E74C3C")

        y    = list(range(len(all_actions)))
        bars = ax.barh(y, days_vals, color=colors,
                       alpha=0.85, height=0.55, edgecolor="white")

        for bar, day in zip(bars, days_vals):
            ax.text(bar.get_width() + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"Day {day}",
                    va="center", fontsize=FONT_ANNOT, color="#333")

        for i, a in enumerate(not_reached):
            y_pos = len(all_actions) + i
            ax.barh(y_pos, MAX_DAY, color="#DDD",
                    alpha=0.5, height=0.55, edgecolor="white")
            ax.text(MAX_DAY + 0.2, y_pos,
                    "Not reached",
                    va="center", fontsize=FONT_ANNOT,
                    color="#999", style="italic")
            all_actions.append(a)

        ax.set_yticks(range(len(all_actions)))
        ax.set_yticklabels(all_actions, fontsize=FONT_TICK)
        ax.set_xlabel("Virtual Day", fontsize=FONT_AXIS)
        ax.set_xlim(0, MAX_DAY + 6)
        ax.axvline(7,  color="#27AE60", linestyle="--", lw=1, alpha=0.4)
        ax.axvline(14, color="#F5A623", linestyle="--", lw=1, alpha=0.4)
        ax.set_title(f"{style['label']}",
                     fontsize=FONT_TITLE, fontweight="bold",
                     color=style["color"])

    fig.suptitle(
        f"Figure 5.5  Days to Reach FAT = {FAT} per Activity  (Day 1–{MAX_DAY})\n"
        "Green ≤ Day 7   |   Orange Day 8–14   |   Red Day 15+   |   Grey: not reached",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {save_path}")


def save_summary(episodes, save_path):
    lines = [
        "Experiment 3: Days to Reach FAT per Activity",
        f"DB: {DB_BASELINE}  |  FAT = {FAT}  |  Day 1–{MAX_DAY}",
        "",
    ]
    for uid in USERS:
        first_day = compute_first_fat_day(episodes, uid, FAT)
        lines += [f"{uid}  ({len(first_day)} habits reached FAT={FAT}):"]
        for a, d in sorted(first_day.items(), key=lambda x: x[1]):
            lines.append(f"    Day {d:2d}  {a}")
        not_reached = [a for a in ADL_LABELS
                       if a not in first_day
                       and any(e.get("user") == uid and
                               e.get("ground_truth") == a
                               for e in episodes)]
        for a in not_reached:
            lines.append(f"    Day --  {a}  (not reached within {MAX_DAY} days)")
        lines.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp3] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    episodes = load_episodes(DB_BASELINE)
    if not episodes:
        print(f"[exp3] No data in {DB_BASELINE}")
        return
    print(f"[exp3] {len(episodes)} episodes loaded")

    plot_days_to_fat(
        episodes,
        os.path.join(RESULTS_DIR, "exp3_days_to_fat.png"))

    save_summary(
        episodes,
        os.path.join(RESULTS_DIR, "exp3_summary.txt"))


if __name__ == "__main__":
    main()