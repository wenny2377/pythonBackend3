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

# Full schedule data transcribed directly from ExperimentRunner.cs, including
# the base templates AND the per-day diffs (Remove/Replace), so the designed
# reference reflects all 7 days the same way the system's 189-episode
# observation does — not just a single un-perturbed day. This corrects an
# earlier version that only used the Day0 base template and assumed diffs
# never change the dominant action per 2-hour slot; that assumption was
# wrong (verified: Mom's 10:00 and 16:00 slots both have a different
# dominant action on several diffed days vs. the base template).

MOM_WEEKDAY_BASE = [
    (7.0, "Opening"), (7.3, "Cooking"), (7.5, "Eating"), (8.0, "Cleaning"),
    (10.0, "Reading"), (10.5, "Drinking"), (12.0, "Eating"), (13.0, "Laying"),
    (15.0, "Cleaning"), (16.0, "UsingPhone"), (18.0, "Cooking"),
    (18.5, "Eating"), (19.5, "Watching"), (20.0, "Drinking"),
    (21.5, "Reading"), (23.0, "Laying"),
]
MOM_WEEKDAY_DIFFS = {
    1: [("Remove", "UsingPhone", None)],
    2: [("Replace", "Drinking", "SeatedDrinking")],
    3: [("Remove", "Reading", None)],
    4: [("Replace", "UsingPhone", "Reading")],
}
MOM_WEEKEND_BASE = [
    (8.5, "Eating"), (9.0, "Opening"), (10.0, "Cooking"), (10.5, "Eating"),
    (11.5, "Reading"), (12.5, "UsingPhone"), (13.0, "Eating"),
    (14.0, "Laying"), (16.0, "Watching"), (17.0, "Drinking"),
    (18.5, "Cooking"), (19.0, "Eating"), (20.0, "Watching"),
    (22.0, "Reading"), (23.5, "Laying"),
]
MOM_WEEKEND_DIFFS = {
    0: [("Remove", "UsingPhone", None)],
    1: [("Replace", "Drinking", "SeatedDrinking")],
}
DAD_WEEKDAY_BASE = [
    (7.0, "Eating"), (8.0, "Typing"), (9.0, "Drinking"), (9.5, "Typing"),
    (10.5, "UsingPhone"), (12.0, "Eating"), (13.0, "Laying"),
    (14.0, "Typing"), (16.0, "UsingPhone"), (18.5, "Eating"),
    (19.5, "Watching"), (21.0, "UsingPhone"), (23.0, "Laying"),
]
DAD_WEEKDAY_DIFFS = {
    1: [("Remove", "Drinking", None)],
    2: [("Replace", "UsingPhone", "Watching")],
    3: [("Remove", "UsingPhone", None)],
    4: [("Replace", "Typing", "Drinking")],
}
DAD_WEEKEND_BASE = [
    (9.0, "Laying"), (10.0, "Eating"), (11.0, "Watching"),
    (12.0, "UsingPhone"), (13.0, "Eating"), (14.5, "Laying"),
    (16.0, "Watching"), (17.5, "Drinking"), (19.0, "Eating"),
    (20.0, "Watching"), (21.5, "UsingPhone"), (23.5, "Laying"),
]
DAD_WEEKEND_DIFFS = {
    0: [("Remove", "UsingPhone", None)],
    1: [("Replace", "Watching", "Drinking")],
}


def _apply_day_diff(base, diffs_for_day):
    result = []
    for hour, action in base:
        removed = False
        for diff_type, target, replacement in diffs_for_day:
            if action != target:
                continue
            if diff_type == "Remove":
                removed = True
                break
            elif diff_type == "Replace":
                action = replacement
        if not removed:
            result.append((hour, action))
    return result


def _expand_7day_schedule(weekday_base, weekday_diffs, weekend_base, weekend_diffs):
    """Returns the full list of (hour, action) across all 7 days (5 weekday + 2 weekend),
    with day-specific diffs applied exactly as ApplyDayDiff() does in ExperimentRunner.cs."""
    all_events = []
    for day in range(5):
        all_events.extend(_apply_day_diff(weekday_base, weekday_diffs.get(day, [])))
    for day in range(2):
        all_events.extend(_apply_day_diff(weekend_base, weekend_diffs.get(day, [])))
    return all_events


def _build_designed_schedule(events):
    """Buckets the 7-day expanded events into 2-hour slots and picks the
    dominant action per slot, matching how the observed-side chart is built."""
    bucket_counts = defaultdict(lambda: defaultdict(int))
    for hour, action in events:
        bucket = int(hour // 2) * 2
        bucket_counts[bucket][action] += 1
    return {
        bucket: max(counts, key=counts.get)
        for bucket, counts in bucket_counts.items()
    }


DESIGNED_SCHEDULE = {
    "User_Mom": _build_designed_schedule(_expand_7day_schedule(
        MOM_WEEKDAY_BASE, MOM_WEEKDAY_DIFFS, MOM_WEEKEND_BASE, MOM_WEEKEND_DIFFS)),
    "User_Dad": _build_designed_schedule(_expand_7day_schedule(
        DAD_WEEKDAY_BASE, DAD_WEEKDAY_DIFFS, DAD_WEEKEND_BASE, DAD_WEEKEND_DIFFS)),
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


GREY      = "#E4E4E4"
DARK_GREY = "#4A4A4A"
MATCH     = "#2D9CDB"
MISMATCH  = "#EB5757"
DESIGN_COLOR = "#9AA5B1"


def plot_daily_routine_strip(hourly: dict, save_path: str):
    hour_buckets = list(range(0, 24, 2))

    fig, axes = plt.subplots(4, 1, figsize=(14, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [0.6, 1, 0.6, 1],
                                           "hspace": 0.15})

    for row_offset, uid in enumerate(USERS):
        ax_design = axes[row_offset * 2]
        ax_obs    = axes[row_offset * 2 + 1]
        data      = hourly[uid]
        designed  = DESIGNED_SCHEDULE.get(uid, {})

        for h in hour_buckets:
            d_action = designed.get(h)
            color = DESIGN_COLOR if d_action else GREY
            ax_design.barh(0, 2, left=h, height=0.8, color=color,
                            edgecolor="white", linewidth=1.5)
            if d_action:
                ax_design.text(h + 1, 0, d_action, ha="center", va="center",
                                fontsize=FONT_TICK - 2, color="white", fontweight="bold")

        ax_design.set_xlim(0, 24)
        ax_design.set_ylim(-0.6, 0.6)
        ax_design.set_yticks([0])
        ax_design.set_yticklabels(["Designed"], fontsize=FONT_TICK - 1)
        ax_design.set_xticks([])
        for spine in ax_design.spines.values():
            spine.set_visible(False)

        for h in hour_buckets:
            counts_here = {a: data[a].get(h, 0) for a in data if data[a].get(h, 0) > 0}
            d_action = designed.get(h)
            if not counts_here:
                color = GREY
                label = ""
            else:
                dominant = max(counts_here, key=counts_here.get)
                total = sum(counts_here.values())
                if d_action is None:
                    color = DARK_GREY
                elif dominant == d_action:
                    color = MATCH
                else:
                    color = MISMATCH
                label = f"{dominant} ({total})"

            ax_obs.barh(0, 2, left=h, height=0.8, color=color,
                        edgecolor="white", linewidth=1.5)
            if label:
                ax_obs.text(h + 1, 0, label, ha="center", va="center",
                            fontsize=FONT_TICK - 2, color="white", fontweight="bold")

        ax_obs.set_xlim(0, 24)
        ax_obs.set_ylim(-0.6, 0.6)
        ax_obs.set_yticks([0])
        ax_obs.set_yticklabels(["Observed"], fontsize=FONT_TICK - 1)
        if row_offset == len(USERS) - 1:
            ax_obs.set_xticks(hour_buckets)
            ax_obs.set_xticklabels([f"{h:02d}:00" for h in hour_buckets], fontsize=FONT_TICK)
        else:
            ax_obs.set_xticks([])
        for spine in ax_obs.spines.values():
            spine.set_visible(False)

        ax_design.set_title(f"{USER_LABELS[uid]} — {DESIGN_INTENT[uid]}",
                             fontsize=FONT_TITLE - 1, fontweight="bold", loc="left", pad=6)

    from matplotlib.patches import Patch
    handles = [
        Patch(color=DESIGN_COLOR, label="Designed slot"),
        Patch(color=MATCH, label="Observed: matches design"),
        Patch(color=MISMATCH, label="Observed: differs from design"),
        Patch(color=DARK_GREY, label="Observed: no design reference for this slot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=FONT_TICK - 1, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Daily Routine: Designed vs Observed (Mom / Dad)\n"
                 "(grey = designed schedule, blue = system observation matches "
                 "design, red = differs)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


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
        ax.text(cx, max(max(mom_vals, default=0), max(dad_vals, default=0)) * 1.12,
                label, ha="center", va="bottom", fontsize=FONT_TICK - 1,
                fontweight="bold", color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(action_order, rotation=30, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Observed count (System A, 189 episodes)", fontsize=FONT_AXIS)
    ax.set_ylim(0, max(max(mom_vals, default=1), max(dad_vals, default=1)) * 1.25)
    ax.legend(fontsize=FONT_TICK, loc="upper right")
    ax.set_title("Shared vs Exclusive Actions: Designed Grouping vs Observed Counts\n"
                 "(highlighted bar = observed for a user it was never designed for)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

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

        designed = DESIGNED_SCHEDULE.get(uid, {})
        slots_with_design = [h for h in designed.keys()]
        matched = 0
        for h in slots_with_design:
            counts_here = {a: data[a].get(h, 0) for a in data if data[a].get(h, 0) > 0}
            if not counts_here:
                continue
            dominant = max(counts_here, key=counts_here.get)
            if dominant == designed[h]:
                matched += 1
        if slots_with_design:
            rate = matched / len(slots_with_design)
            lines.append(f"  Designed-vs-Observed match rate: {matched}/{len(slots_with_design)} "
                         f"slots ({rate:.1%})")
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

    plot_daily_routine_strip(
        hourly, os.path.join(RESULTS_DIR, "exp4_daily_routine_strip.png"))

    plot_shared_vs_exclusive_actions(
        hourly, os.path.join(RESULTS_DIR, "exp4_shared_vs_exclusive.png"))

    exp1_docs = load_docs(db, COL_SEMANTIC)
    exp1_acc  = compute_accuracy(exp1_docs)[0] if exp1_docs else None

    save_summary(hourly, os.path.join(RESULTS_DIR, "exp4_summary.txt"),
                 exp1_accuracy=exp1_acc)

    print("\n[exp4] Done.")


if __name__ == "__main__":
    main()