
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
    LINE_WIDTH, MARKER_SIZE, FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

FAT      = 5
MAX_DAY  = 21
NO_RECORD = {"Walking", "Standing", "StandUp", "PickingUp", "PuttingDown"}


def load_episodes(db_name):
    db = MongoClient(MONGO_URI)[db_name]
    return list(db.eval_logs.find(
        {"spatial_action": {"$exists": True, "$nin": list(NO_RECORD)},
         "zone_label":     {"$exists": True, "$ne": ""},
         "virtual_day":    {"$lte": MAX_DAY, "$exists": True}},
        {"user": 1, "spatial_action": 1, "virtual_day": 1, "virtual_hour": 1}
    ).sort([("virtual_day", 1), ("virtual_hour", 1)]))


def habits_at_day(episodes, fat, target_day):
    weight = defaultdict(float)
    for ep in episodes:
        if ep.get("virtual_day", 0) > target_day:
            break
        u = ep.get("user", "")
        a = ep.get("spatial_action", "")
        if u and a:
            weight[(u, a)] += 1
    return {k for k, w in weight.items() if w >= fat}


# ── Plot 1: Convergence (baseline only, per user) ─────────────────────────────

def plot_convergence(episodes, save_path):
    days = list(range(1, MAX_DAY + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    for uid, color, marker in zip(
        USERS,
        [C["mom"], C["dad"]],
        ["o", "s"]
    ):
        eps_u = [e for e in episodes if e.get("user") == uid]
        ys    = [len({a for (u,a) in habits_at_day(eps_u, FAT, d) if u == uid})
                 for d in days]

        # fix: count only this user's habits
        ys_fixed = []
        for d in days:
            h = habits_at_day(eps_u, FAT, d)
            ys_fixed.append(len(h))

        ax.plot(days, ys_fixed,
                marker + "-", color=color,
                lw=LINE_WIDTH, ms=MARKER_SIZE,
                mfc="white", mew=2,
                label=uid.replace("_", " "))

        # Convergence marker
        stable_day = None
        for i in range(len(ys_fixed) - 2):
            if ys_fixed[i] == ys_fixed[i+1] == ys_fixed[i+2]:
                stable_day = days[i]
                break

        if stable_day:
            ax.axvline(stable_day, color=color,
                       linestyle=":", lw=1.2, alpha=0.6)
            ax.text(stable_day + 0.2,
                    ys_fixed[-1] * 0.15,
                    f"Stable\nDay {stable_day}",
                    fontsize=FONT_ANNOT - 1, color=color, alpha=0.8)

        # Final count label
        ax.text(days[-1] + 0.3, ys_fixed[-1],
                f"{ys_fixed[-1]} habits",
                fontsize=FONT_ANNOT, color=color, va="center")

    ax.set_xlabel("Virtual Day", fontsize=FONT_AXIS)
    ax.set_ylabel("Stable Habits Learned", fontsize=FONT_AXIS)
    ax.set_xlim(0, MAX_DAY + 3)
    ax.set_ylim(0, 12)
    ax.set_xticks(range(0, MAX_DAY + 1, 3))
    ax.set_title(
        f"Habit Learning Convergence  (FAT = {FAT}, Day 1–{MAX_DAY})\n"
        "System progressively learns stable habits from passive observation",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {save_path}")


# ── Plot 2: Personalization ───────────────────────────────────────────────────

def plot_personalization(episodes, save_path):
    weight = defaultdict(float)
    for ep in episodes:
        u = ep.get("user", "")
        a = ep.get("spatial_action", "")
        if u and a and a in ADL_LABELS:
            weight[(u, a)] += 1

    actions_present = sorted({
        a for (u, a), w in weight.items()
        if w >= FAT and a in ADL_LABELS
    })

    if not actions_present:
        print("[exp3] No personalization data"); return

    n = len(actions_present)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    mom_vals = [weight.get(("User_Mom", a), 0) for a in actions_present]
    dad_vals = [weight.get(("User_Dad", a), 0) for a in actions_present]

    bars_m = ax.bar(x - w/2, mom_vals, w, color=C["mom"],
                    alpha=0.85, label="User Mom", edgecolor="white")
    bars_d = ax.bar(x + w/2, dad_vals, w, color=C["dad"],
                    alpha=0.85, label="User Dad", edgecolor="white")

    for bar, v in list(zip(bars_m, mom_vals)) + list(zip(bars_d, dad_vals)):
        if v >= FAT:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}", ha="center",
                    fontsize=FONT_ANNOT,
                    color=bar.get_facecolor())

    ax.axhline(FAT, color=C["threshold"], linestyle="--",
               lw=1.2, alpha=0.6, label=f"FAT = {FAT}")

    # Annotate unique habits
    for i, a in enumerate(actions_present):
        mom_v = weight.get(("User_Mom", a), 0)
        dad_v = weight.get(("User_Dad", a), 0)
        if mom_v >= FAT and dad_v < FAT:
            ax.text(x[i], -4, "Mom\nonly",
                    ha="center", fontsize=7,
                    color=C["mom"], fontweight="bold")
        elif dad_v >= FAT and mom_v < FAT:
            ax.text(x[i], -4, "Dad\nonly",
                    ha="center", fontsize=7,
                    color=C["dad"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(actions_present, rotation=30,
                       ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Observation Count", fontsize=FONT_AXIS)
    ax.set_ylim(-6, max(mom_vals + dad_vals) + 8)
    ax.set_title(
        "Personalization — Learned Habit Profiles per User\n"
        "Users develop distinct behaviour patterns through passive observation",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {save_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(episodes, save_path):
    weight = defaultdict(float)
    for ep in episodes:
        weight[(ep.get("user",""), ep.get("spatial_action",""))] += 1

    lines = [
        "Experiment 3: Habit Learning Convergence & Personalization",
        f"DB: {DB_BASELINE}  |  FAT = {FAT}  |  Day 1–{MAX_DAY}",
        "",
    ]
    for uid in USERS:
        habits = {(u,a): w for (u,a),w in weight.items()
                  if u == uid and w >= FAT}
        lines += [f"{uid}  ({len(habits)} stable habits):"]
        for (_, a), w in sorted(habits.items(), key=lambda x: -x[1]):
            lines.append(f"    {a:<16} weight={w:.0f}")
        lines.append("")

    mom = {a for (u,a),w in weight.items() if u=="User_Mom" and w>=FAT}
    dad = {a for (u,a),w in weight.items() if u=="User_Dad" and w>=FAT}
    lines += [
        "Personalization:",
        f"  Shared   : {sorted(mom & dad)}",
        f"  Mom-only : {sorted(mom - dad)}",
        f"  Dad-only : {sorted(dad - mom)}",
    ]
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp3] Saved: {save_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    episodes = load_episodes(DB_BASELINE)
    if not episodes:
        print(f"[exp3] No data in {DB_BASELINE}"); return
    print(f"[exp3] {len(episodes)} episodes")
    plot_convergence(episodes,
                     os.path.join(RESULTS_DIR, "exp3_habit_convergence.png"))
    plot_personalization(episodes,
                         os.path.join(RESULTS_DIR, "exp3_personalization.png"))
    save_summary(episodes,
                 os.path.join(RESULTS_DIR, "exp3_summary.txt"))


if __name__ == "__main__":
    main()