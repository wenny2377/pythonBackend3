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
    LINE_WIDTH, MARKER_SIZE, FIG_DPI, RESULTS_DIR, apply_style
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


def compute_learning_timeline(episodes, uid, fat):
    weight     = defaultdict(float)
    learned    = set()
    new_habits = defaultdict(list)
    days_list  = list(range(1, MAX_DAY + 1))
    counts     = []

    eps_u = [e for e in episodes if e.get("user") == uid]

    for day in days_list:
        for ep in eps_u:
            if ep.get("virtual_day") == day:
                gt   = ep.get("ground_truth", "")
                pred = ep.get("spatial_action") or ep.get("vlm_output", "")
                if gt and gt == pred and gt in ADL_LABELS:
                    weight[gt] += 1
                    if weight[gt] >= fat and gt not in learned:
                        learned.add(gt)
                        new_habits[day].append(gt)
        counts.append(len(learned))

    return days_list, counts, new_habits


def plot_learning_speed(episodes, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    label_offsets = {}

    for uid, style in USER_STYLES.items():
        days, counts, new_habits = compute_learning_timeline(
            episodes, uid, FAT)

        ax.step(days, counts, where="post",
                color=style["color"], lw=LINE_WIDTH + 0.5,
                label=style["label"])

        for day, actions in new_habits.items():
            y_val = counts[days.index(day)]

            ax.plot(day, y_val,
                    style["marker"],
                    color=style["color"],
                    ms=MARKER_SIZE + 1,
                    mfc=style["color"], mew=1.5, zorder=5)

            key    = (day, y_val)
            offset = label_offsets.get(key, 0)
            label_offsets[key] = offset + 0.6

            for i, action in enumerate(actions):
                ax.annotate(
                    action,
                    xy=(day, y_val),
                    xytext=(day + 0.3, y_val - 0.3 - offset - i * 0.55),
                    fontsize=7,
                    color=style["color"],
                    arrowprops=dict(
                        arrowstyle="-",
                        color=style["color"],
                        lw=0.8, alpha=0.5),
                )

    _, counts_m, _ = compute_learning_timeline(episodes, "User_Mom", FAT)
    _, counts_d, _ = compute_learning_timeline(episodes, "User_Dad", FAT)

    stable_m = next((i+1 for i in range(len(counts_m)-2)
                     if counts_m[i] == counts_m[i+1] == counts_m[i+2]), MAX_DAY)
    stable_d = next((i+1 for i in range(len(counts_d)-2)
                     if counts_d[i] == counts_d[i+1] == counts_d[i+2]), MAX_DAY)

    ax.axvline(stable_m, color=C["mom"], linestyle=":",
               lw=1.2, alpha=0.5)
    ax.axvline(stable_d, color=C["dad"], linestyle=":",
               lw=1.2, alpha=0.5)

    ax.text(stable_m + 0.15, 0.3,
            f"Mom stable\nDay {stable_m}",
            fontsize=FONT_ANNOT, color=C["mom"], alpha=0.85)
    ax.text(stable_d + 0.15, 1.2,
            f"Dad stable\nDay {stable_d}",
            fontsize=FONT_ANNOT, color=C["dad"], alpha=0.85)

    ax.set_xlabel("Virtual Day", fontsize=FONT_AXIS)
    ax.set_ylabel("Cumulative Stable Habits Learned", fontsize=FONT_AXIS)
    ax.set_xlim(0, MAX_DAY + 1)
    ax.set_ylim(0, max(max(counts_m), max(counts_d)) + 3)
    ax.set_xticks(range(0, MAX_DAY + 1, 3))
    ax.set_title(
        f"Habit Learning Speed  (FAT = {FAT}, Day 1-{MAX_DAY})\n"
        "Only correctly recognised episodes contribute to habit weight",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def save_summary(episodes, save_path):
    lines = [
        "Experiment 4: Habit Learning Speed",
        f"DB: {DB_BASELINE}  |  FAT = {FAT}  |  Day 1-{MAX_DAY}",
        "Note: Only correctly recognised episodes (spatial_action == ground_truth) count toward habit weight",
        "",
    ]
    for uid in USERS:
        days, counts, new_habits = compute_learning_timeline(
            episodes, uid, FAT)
        stable = next((i+1 for i in range(len(counts)-2)
                       if counts[i] == counts[i+1] == counts[i+2]), MAX_DAY)
        lines += [
            f"{uid}  (stable by Day {stable}, {counts[-1]} habits total):",
        ]
        for day in sorted(new_habits.keys()):
            for action in new_habits[day]:
                lines.append(f"    Day {day:2d} -> learned: {action}")
        lines.append("")

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
    print(f"[exp4] {len(episodes)} episodes")
    plot_learning_speed(
        episodes,
        os.path.join(RESULTS_DIR, "exp4_learning_speed.png"))
    save_summary(
        episodes,
        os.path.join(RESULTS_DIR, "exp4_summary.txt"))


if __name__ == "__main__":
    main()