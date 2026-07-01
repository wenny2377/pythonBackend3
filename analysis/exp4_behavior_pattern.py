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
    "Drinking", "Laying", "UsingPhone", "Watching", "Typing",
}

NEVER_SCHEDULED = {
    "User_Mom": sorted(ALL_SCHEDULED_ACTIONS - {
        "Opening", "Cooking", "Eating", "Cleaning", "Reading",
        "Drinking", "Laying", "UsingPhone", "Watching",
    }),
    "User_Dad": sorted(ALL_SCHEDULED_ACTIONS - {
        "Eating", "Typing", "Drinking",
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
    shared   = sorted(SCHEDULED_FOR_USER["User_Mom"] & SCHEDULED_FOR_USER["User_Dad"])
    mom_only = sorted(SCHEDULED_FOR_USER["User_Mom"] - SCHEDULED_FOR_USER["User_Dad"])
    dad_only = sorted(SCHEDULED_FOR_USER["User_Dad"] - SCHEDULED_FOR_USER["User_Mom"])
    action_order = mom_only + shared + dad_only

    if not action_order:
        print("[exp4] Skipping shared-vs-exclusive chart: no scheduled actions found")
        return

    def _total(uid, action):
        return sum(hourly[uid].get(action, {}).values())

    mom_vals = [_total("User_Mom", a) for a in action_order]
    dad_vals = [_total("User_Dad", a) for a in action_order]

    x = np.arange(len(action_order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(action_order) * 1.5), 6))

    bars_m = ax.bar(x - w / 2, mom_vals, w, label="Mom (observed count)",
                    color=C["mom"], alpha=0.85, edgecolor="white", zorder=3)
    bars_d = ax.bar(x + w / 2, dad_vals, w, label="Dad (observed count)",
                    color=C["dad"], alpha=0.85, edgecolor="white", zorder=3)

    ymax = max(max(mom_vals, default=1), max(dad_vals, default=1))

    # value labels
    for bar, val, action in zip(bars_m, mom_vals, action_order):
        if val > 0:
            unexpected = action not in SCHEDULED_FOR_USER["User_Mom"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                str(val),
                ha="center", va="bottom",
                fontsize=FONT_TICK - 1,
                color=C["highlight"] if unexpected else "#333",
                fontweight="bold" if unexpected else "normal",
                zorder=4,
            )

    for bar, val, action in zip(bars_d, dad_vals, action_order):
        if val > 0:
            unexpected = action not in SCHEDULED_FOR_USER["User_Dad"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                str(val),
                ha="center", va="bottom",
                fontsize=FONT_TICK - 1,
                color=C["highlight"] if unexpected else "#333",
                fontweight="bold" if unexpected else "normal",
                zorder=4,
            )

    # group dividers
    n_shared = len(shared)
    n_mom    = len(mom_only)
    n_dad    = len(dad_only)

    # order: mom_only | shared | dad_only
    boundaries = []
    if n_mom and (n_shared or n_dad):
        boundaries.append(n_mom - 0.5)
    if n_dad and (n_shared or n_mom):
        boundaries.append(n_mom + n_shared - 0.5)

    for b in boundaries:
        ax.axvline(b, color="#999", linestyle="--", lw=1.2, alpha=0.7, zorder=2)

    # group background shading — order: mom_only | shared | dad_only
    shade_alpha = 0.06
    group_ranges = []
    if n_mom:
        group_ranges.append((0, n_mom, C["mom"], f"Mom-only (n={n_mom})"))
    if n_shared:
        group_ranges.append((n_mom, n_mom + n_shared, "#888888", f"Shared (n={n_shared})"))
    if n_dad:
        group_ranges.append((n_mom + n_shared, n_mom + n_shared + n_dad, C["dad"], f"Dad-only (n={n_dad})"))

    for start, end, color, label in group_ranges:
        ax.axvspan(start - 0.5, end - 0.5, alpha=shade_alpha, color=color, zorder=1)

    # group labels — centred over each shaded region
    for start, end, color, label in group_ranges:
        cx = (start + end - 1) / 2
        # single-bar group: shift right so label sits over the bar
        if end - start == 1:
            cx += 0.3
        ax.text(
            cx, ymax * 1.10,
            label,
            ha="center", va="bottom",
            fontsize=FONT_TICK - 1,
            fontweight="bold",
            color="#444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(action_order, rotation=30, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Observed count (System A, 189 episodes)", fontsize=FONT_AXIS)
    ax.set_ylim(0, ymax * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # legend inside plot
    ax.legend(
        fontsize=FONT_TICK,
        loc="upper right",
        framealpha=0.85,
    )

    ax.set_title(
        "BPA Behavioral Differentiation: Observed Action Counts per User",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=10,
    )

    # caption as x-axis label (stays inside bbox)
    caption_lines = [
        f"Grouping source: ExperimentRunner.cs  |  "
        f"Shared: {', '.join(shared)}",
        f"Mom-only: {', '.join(mom_only) if mom_only else '(none)'}   "
        f"|   Dad-only: {', '.join(dad_only) if dad_only else '(none)'}",
    ]
    ax.set_xlabel(
        "\n".join(caption_lines),
        fontsize=FONT_TICK - 2,
        color="#666",
        labelpad=12,
    )

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
            f"the designed routine despite that error rate."
        )
    lines.append("")

    for uid in USERS:
        lines.append(f"{USER_LABELS[uid]} — design intent: {DESIGN_INTENT[uid]}")
        never = NEVER_SCHEDULED.get(uid, [])
        if never:
            lines.append(
                f"  Never scheduled for this user (by design): {', '.join(never)}"
            )
        data = hourly[uid]
        observed_actions = sorted(data.keys(), key=lambda a: -sum(data[a].values()))
        unexpected = [a for a in observed_actions if a in never]
        if unexpected:
            lines.append(
                f"  WARNING — observed but never scheduled (likely HAR "
                f"misclassification, not a real occurrence): "
                f"{', '.join(unexpected)}"
            )
        for action in observed_actions:
            buckets  = data[action]
            peak_hour = max(buckets.items(), key=lambda x: x[1])[0]
            total    = sum(buckets.values())
            flag     = "  [unexpected]" if action in never else ""
            lines.append(
                f"  {action:16} total={total:3}  peak={peak_hour:02d}:00{flag}"
            )
        lines.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp4] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db = MongoClient(MONGO_URI)[DB_BASELINE]

    hourly = load_hourly_action_data(db)
    total  = sum(
        sum(sum(b.values()) for b in a.values())
        for a in hourly.values()
    )

    if total == 0:
        print(f"[exp4] No data in {DB_BASELINE}.{COL_SEMANTIC}")
        return

    print(f"[exp4] {total} episodes loaded")

    plot_shared_vs_exclusive_actions(
        hourly,
        os.path.join(RESULTS_DIR, "exp4_shared_vs_exclusive.png"),
    )

    exp1_docs = load_docs(db, COL_SEMANTIC)
    exp1_acc  = compute_accuracy(exp1_docs)[0] if exp1_docs else None

    save_summary(
        hourly,
        os.path.join(RESULTS_DIR, "exp4_summary.txt"),
        exp1_accuracy=exp1_acc,
    )

    print("\n[exp4] Done.")


if __name__ == "__main__":
    main()