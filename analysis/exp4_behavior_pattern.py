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
    MONGO_URI, DB_BASELINE,
    USERS, C, COL_SEMANTIC,
    FONT_TITLE, FONT_AXIS, FONT_TICK,
    FIG_DPI, RESULTS_DIR,
    apply_style, load_docs, compute_accuracy,
)

apply_style()

USER_LABELS = {
    "User_Mom": "Mom",
    "User_Dad": "Dad",
}

DESIGN_INTENT = {
    "User_Mom": "All-day household routine (cooking/cleaning/reading spread "
                "across morning to night)",
    "User_Dad": "Office-hours pattern (Typing concentrated 08:00-16:00, "
                "leisure only in the evening)",
}

ALL_SCHEDULED_ACTIONS = {
    "Opening", "Cooking", "Eating", "Cleaning", "Reading",
    "Drinking", "SeatedDrinking", "Laying", "UsingPhone",
    "Watching", "Typing",
}

NEVER_SCHEDULED = {
    "User_Mom": sorted(ALL_SCHEDULED_ACTIONS - {
        "Opening", "Cooking", "Eating", "Cleaning", "Reading",
        "Drinking", "SeatedDrinking", "Laying", "UsingPhone", "Watching",
    }),
    "User_Dad": sorted(ALL_SCHEDULED_ACTIONS - {
        "Eating", "Typing", "Drinking", "SeatedDrinking",
        "UsingPhone", "Laying", "Watching",
    }),
}

SCHEDULED_FOR_USER = {
    "User_Mom": ALL_SCHEDULED_ACTIONS - set(NEVER_SCHEDULED["User_Mom"]),
    "User_Dad": ALL_SCHEDULED_ACTIONS - set(NEVER_SCHEDULED["User_Dad"]),
}

def load_hourly_action_data(db) -> dict:
    docs = load_docs(db, COL_SEMANTIC)
    result = {uid: defaultdict(lambda: defaultdict(int)) for uid in USERS}
    for d in docs:
        uid = d.get("user", "")
        if uid not in USERS:
            continue
        action = d.get("_pred", "")
        hour   = d.get("virtual_hour")
        if not action or hour is None:
            continue
        bucket = int(float(hour) // 2) * 2
        result[uid][action][bucket] += 1
    return result


def plot_shared_vs_exclusive_actions(hourly: dict, save_path: str):
    shared = sorted(SCHEDULED_FOR_USER["User_Mom"] & SCHEDULED_FOR_USER["User_Dad"])
    mom_only = sorted(SCHEDULED_FOR_USER["User_Mom"] - SCHEDULED_FOR_USER["User_Dad"])
    dad_only = sorted(SCHEDULED_FOR_USER["User_Dad"] - SCHEDULED_FOR_USER["User_Mom"])
    action_order = shared + mom_only + dad_only

    if not action_order:
        print("[exp4] Skipping shared-vs-exclusive chart: no scheduled actions found")
        return

    def _total(uid, action):
        return sum(hourly[uid].get(action, {}).values())

    mom_vals = [_total("User_Mom", a) for a in action_order]
    dad_vals = [_total("User_Dad", a) for a in action_order]

    x = np.arange(len(action_order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(action_order) * 1.1), 5.5))
    bars_m = ax.bar(x - w/2, mom_vals, w, label="Mom (observed count)",
                     color=C["mom"], alpha=0.85, edgecolor="white")
    bars_d = ax.bar(x + w/2, dad_vals, w, label="Dad (observed count)",
                     color=C["dad"], alpha=0.85, edgecolor="white")

    for bar, val, uid, action in zip(bars_m, mom_vals, ["User_Mom"] * len(action_order), action_order):
        if val > 0:
            unexpected = action not in SCHEDULED_FOR_USER["User_Mom"]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=FONT_TICK - 1,
                    color=C["highlight"] if unexpected else "#333",
                    fontweight="bold" if unexpected else "normal")
    for bar, val, uid, action in zip(bars_d, dad_vals, ["User_Dad"] * len(action_order), action_order):
        if val > 0:
            unexpected = action not in SCHEDULED_FOR_USER["User_Dad"]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=FONT_TICK - 1,
                    color=C["highlight"] if unexpected else "#333",
                    fontweight="bold" if unexpected else "normal")

    n_shared, n_mom, n_dad = len(shared), len(mom_only), len(dad_only)
    for boundary, label in [(n_shared - 0.5, ""), (n_shared + n_mom - 0.5, "")]:
        if 0 < boundary < len(action_order) - 1:
            ax.axvline(boundary, color="#999", linestyle="--", lw=1.2, alpha=0.6)

    ymax = max(max(mom_vals, default=1), max(dad_vals, default=1))

    group_centers = []
    group_labels = []
    if n_shared:
        group_centers.append((n_shared - 1) / 2)
        group_labels.append(f"Shared by design (n={n_shared})")
    if n_mom:
        group_centers.append(n_shared + (n_mom - 1) / 2)
        group_labels.append(f"Mom-only by design (n={n_mom})")
    if n_dad:
        group_centers.append(n_shared + n_mom + (n_dad - 1) / 2)
        group_labels.append(f"Dad-only by design (n={n_dad})")

    for cx, label in zip(group_centers, group_labels):
        ax.text(cx, ymax * 1.20, label, ha="center", va="bottom",
                fontsize=FONT_TICK - 1, fontweight="bold", color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(action_order, rotation=30, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Observed count (System A, 189 episodes)", fontsize=FONT_AXIS)
    ax.set_ylim(0, ymax * 1.40)
    ax.legend(fontsize=FONT_TICK, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0)
    ax.set_title("Shared vs Exclusive Actions: Designed Grouping vs Observed Counts\n"
                 "(highlighted bar = observed for a user it was never designed for)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    caption = (
        f"Grouping source: ExperimentRunner.cs schedule (MomWeekday/MomWeekend, "
        f"DadWeekday/DadWeekend + DayDiff)\n"
        f"Shared by design: {', '.join(shared)}\n"
        f"Mom-only by design: {', '.join(mom_only) if mom_only else '(none)'}   |   "
        f"Dad-only by design: {', '.join(dad_only) if dad_only else '(none)'}"
    )
    fig.text(0.02, -0.02, caption, ha="left", va="top",
              fontsize=FONT_TICK - 2, color="#555", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def save_summary(hourly: dict, save_path: str, exp1_accuracy: float = None):
    lines = [
        "Experiment 4: Behavior Pattern Differentiation",
        f"DB: {DB_BASELINE}",
        "",
        "Note: data comes from the system's own HAR predictions (spatial_action),",
        "not ground truth. This mirrors real deployment, where no ground truth",
        "is available.",
    ]
    if exp1_accuracy is not None:
        lines.append(
            f"Exp 1 measured {exp1_accuracy:.1%} HAR accuracy on this same data; "
            f"the question here is whether the learned pattern still matches "
            f"the designed routine despite that error rate.")
    lines.append("")

    for uid in USERS:
        lines.append(f"{USER_LABELS[uid]} — design intent: {DESIGN_INTENT[uid]}")
        never = NEVER_SCHEDULED.get(uid, [])
        if never:
            lines.append(
                f"  Never scheduled for this user (by design): {', '.join(never)}")
        data = hourly[uid]
        observed_actions = sorted(data.keys(), key=lambda a: -sum(data[a].values()))
        unexpected = [a for a in observed_actions if a in never]
        if unexpected:
            lines.append(
                f"  WARNING — observed but never scheduled (likely HAR "
                f"misclassification, not a real occurrence): "
                f"{', '.join(unexpected)}")
        for action in observed_actions:
            buckets = data[action]
            peak_hour = max(buckets.items(), key=lambda x: x[1])[0]
            total = sum(buckets.values())
            flag = "  [unexpected]" if action in never else ""
            lines.append(f"  {action:16} total={total:3}  peak={peak_hour:02d}:00{flag}")
        lines.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp4] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db = MongoClient(MONGO_URI)[DB_BASELINE]

    hourly = load_hourly_action_data(db)
    total = sum(sum(sum(b.values()) for b in a.values()) for a in hourly.values())

    if total == 0:
        print(f"[exp4] No data in {DB_BASELINE}.{COL_SEMANTIC}")
        return

    print(f"[exp4] {total} episodes loaded")

    plot_shared_vs_exclusive_actions(
        hourly, os.path.join(RESULTS_DIR, "exp4_shared_vs_exclusive.png"))

    exp1_docs = load_docs(db, COL_SEMANTIC)
    exp1_acc  = compute_accuracy(exp1_docs)[0] if exp1_docs else None

    save_summary(hourly, os.path.join(RESULTS_DIR, "exp4_summary.txt"),
                 exp1_accuracy=exp1_acc)

    print("\n[exp4] Done.")


if __name__ == "__main__":
    main()
