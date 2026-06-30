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
    USERS, ADL_LABELS, C,
    FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR,
    apply_style,
)

apply_style()

NO_WEIGHT_ACTIONS = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}

TIME_SLOTS = ["Morning", "Noon", "Afternoon", "Evening", "Night"]

KEY_TRANSITIONS = [
    ("Watching",  "Drinking"),
    ("Eating",    "Watching"),
    ("Typing",    "Drinking"),
    ("Eating",    "Typing"),
    ("Watching",  "UsingPhone"),
    ("Eating",    "Laying"),
]

USER_LABELS = {
    "User_Mom": "Mom",
    "User_Dad": "Dad",
}


def load_transition_counts(db) -> dict:
    result = {uid: defaultdict(lambda: defaultdict(int)) for uid in USERS}
    for doc in db.transition_counts.find({}):
        uid  = doc.get("user_id", "")
        frm  = doc.get("from_action", "")
        to   = doc.get("to_action", "")
        slot = doc.get("time_slot", "All")
        cnt  = doc.get("count", 0)
        if uid in USERS:
            result[uid][frm][to] += cnt
    return result


def load_observation_logs(db) -> dict:
    result = {uid: [] for uid in USERS}
    for doc in db.observation_logs.find({}):
        uid = doc.get("user", "")
        if uid in USERS:
            result[uid].append(doc)
    return result


def plot_transition_comparison(trans: dict, save_path: str):
    mom_vals, dad_vals, labels = [], [], []
    for frm, to in KEY_TRANSITIONS:
        mom_cnt = trans.get("User_Mom", {}).get(frm, {}).get(to, 0)
        dad_cnt = trans.get("User_Dad", {}).get(frm, {}).get(to, 0)
        mom_vals.append(mom_cnt)
        dad_vals.append(dad_cnt)
        labels.append(f"{frm}\n→ {to}")

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_m = ax.bar(x - w/2, mom_vals, w, label="Mom",
                    color=C["mom"], alpha=0.85, edgecolor="white")
    bars_d = ax.bar(x + w/2, dad_vals, w, label="Dad",
                    color=C["dad"], alpha=0.85, edgecolor="white")

    for bar, val in zip(list(bars_m) + list(bars_d),
                        mom_vals + dad_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1, str(val),
                    ha="center", va="bottom", fontsize=FONT_ANNOT)

    ax.axhline(3, color="#999", linestyle="--", lw=1.0,
               alpha=0.6, label="Min threshold (3)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Transition Count (7 days)", fontsize=FONT_AXIS)
    ax.set_title("BPA Transition Counts — Mom vs Dad\n"
                 "(Behavioral Pattern Accumulation, 7-day observation)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def plot_top_habits(obs: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, uid in zip(axes, USERS):
        docs = [d for d in obs[uid] if d.get("action") not in NO_WEIGHT_ACTIONS]
        agg  = defaultdict(float)
        for d in docs:
            key = d.get("action", "")
            agg[key] += d.get("weight", 1)
        if not agg:
            ax.set_title(f"{USER_LABELS[uid]}\n(no data)")
            continue
        sorted_items = sorted(agg.items(), key=lambda x: -x[1])[:8]
        actions = [i[0] for i in sorted_items]
        weights = [i[1] for i in sorted_items]
        color   = C["mom"] if uid == "User_Mom" else C["dad"]

        bars = ax.barh(range(len(actions)), weights,
                       color=color, alpha=0.82, height=0.55, edgecolor="white")
        for bar, w in zip(bars, weights):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.0f}", va="center", fontsize=FONT_ANNOT)

        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=FONT_TICK)
        ax.set_xlabel("Accumulated Weight", fontsize=FONT_AXIS)
        ax.set_title(f"{USER_LABELS[uid]} — Top Activities\n"
                     f"({len(docs)} observations over 7 days)",
                     fontsize=FONT_TITLE, fontweight="bold")

    plt.suptitle("Individual Behavioral Profiles (BPA) — 7-day Accumulation",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def plot_timeslot_heatmap(obs: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, uid in zip(axes, USERS):
        docs = [d for d in obs[uid] if d.get("action") not in NO_WEIGHT_ACTIONS]
        matrix = defaultdict(lambda: defaultdict(float))
        for d in docs:
            action = d.get("action", "")
            slot   = d.get("time_slot", "")
            if action in ADL_LABELS and slot in TIME_SLOTS:
                matrix[action][slot] += d.get("weight", 1)

        present_actions = [a for a in ADL_LABELS
                           if any(matrix[a][s] > 0 for s in TIME_SLOTS)]
        if not present_actions:
            ax.set_title(f"{USER_LABELS[uid]}\n(no data)")
            continue

        data = np.array([[matrix[a][s] for s in TIME_SLOTS]
                         for a in present_actions])
        row_max = data.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        data_norm = data / row_max

        color = "RdPu" if uid == "User_Mom" else "Blues"
        im = ax.imshow(data_norm, cmap=color, vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Relative Weight")

        for i in range(len(present_actions)):
            for j in range(len(TIME_SLOTS)):
                if data[i, j] > 0:
                    ax.text(j, i, f"{data[i,j]:.0f}",
                            ha="center", va="center", fontsize=7.5,
                            color="white" if data_norm[i, j] > 0.6 else "black")

        ax.set_xticks(range(len(TIME_SLOTS)))
        ax.set_xticklabels(TIME_SLOTS, fontsize=FONT_TICK)
        ax.set_yticks(range(len(present_actions)))
        ax.set_yticklabels(present_actions, fontsize=FONT_TICK)
        ax.set_title(f"{USER_LABELS[uid]} — Activity × Time Slot",
                     fontsize=FONT_TITLE, fontweight="bold")

    plt.suptitle("Temporal Behavioral Pattern Distribution (BPA, 7-day)",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def plot_object_preference(db, save_path: str):
    mom_objs = defaultdict(int)
    dad_objs = defaultdict(int)
    for doc in db.object_events.find({}):
        uid = doc.get("user", "")
        obj = doc.get("object", "").lower().strip()
        if not obj or obj in ("none", ""):
            continue
        if uid == "User_Mom":
            mom_objs[obj] += 1
        elif uid == "User_Dad":
            dad_objs[obj] += 1

    if not mom_objs and not dad_objs:
        print("[exp4] No object_events data")
        return

    all_objs = sorted(
        set(list(mom_objs.keys()) + list(dad_objs.keys())),
        key=lambda o: -(mom_objs.get(o, 0) + dad_objs.get(o, 0))
    )[:12]

    x = np.arange(len(all_objs))
    w = 0.35
    mom_vals = [mom_objs.get(o, 0) for o in all_objs]
    dad_vals = [dad_objs.get(o, 0) for o in all_objs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, mom_vals, w, label="Mom", color=C["mom"], alpha=0.85)
    ax.bar(x + w/2, dad_vals, w, label="Dad", color=C["dad"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_objs, rotation=30, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Pickup Count", fontsize=FONT_AXIS)
    ax.set_title("Object Interaction Preference (BPA) — Mom vs Dad",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def plot_watching_to_drinking(trans: dict, save_path: str):
    slots    = TIME_SLOTS
    mom_vals = [trans.get("User_Mom", {}).get("Watching", {}).get("Drinking", 0)
                for _ in slots]
    dad_vals = [trans.get("User_Dad", {}).get("Watching", {}).get("Drinking", 0)
                for _ in slots]

    # use per-slot data from DB
    client = MongoClient(MONGO_URI)
    db     = client[DB_BASELINE]
    mom_slot = defaultdict(int)
    dad_slot = defaultdict(int)
    for doc in db.transition_counts.find(
            {"from_action": "Watching", "to_action": "Drinking"}):
        uid  = doc.get("user_id", "")
        slot = doc.get("time_slot", "")
        cnt  = doc.get("count", 0)
        if uid == "User_Mom": mom_slot[slot] += cnt
        if uid == "User_Dad": dad_slot[slot] += cnt

    mom_vals = [mom_slot.get(s, 0) for s in slots]
    dad_vals = [dad_slot.get(s, 0) for s in slots]

    x = np.arange(len(slots))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, mom_vals, w, label="Mom", color=C["mom"], alpha=0.85)
    ax.bar(x + w/2, dad_vals, w, label="Dad", color=C["dad"], alpha=0.85)
    ax.axhline(3, color="#999", linestyle="--", lw=1.2, alpha=0.6,
               label="Proactive trigger threshold (3)")
    ax.set_xticks(x)
    ax.set_xticklabels(slots, fontsize=FONT_TICK)
    ax.set_ylabel("Transition Count", fontsize=FONT_AXIS)
    ax.set_title("Watching → Drinking Transition by Time Slot\n"
                 "(BPA-learned pattern: basis for proactive service trigger)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved: {save_path}")


def save_summary(trans: dict, obs: dict, save_path: str):
    lines = [
        "Experiment 4: Behavior Pattern Learning",
        f"DB: {DB_BASELINE}",
        "",
    ]
    for uid in USERS:
        lines.append(f"{USER_LABELS[uid]}:")
        lines.append(f"  Observations: {len(obs[uid])}")
        top_trans = sorted(
            [(frm, to, cnt)
             for frm, tos in trans.get(uid, {}).items()
             for to, cnt in tos.items()],
            key=lambda x: -x[2]
        )[:5]
        lines.append("  Top transitions:")
        for frm, to, cnt in top_trans:
            lines.append(f"    {frm:16} → {to:16} count={cnt}")

        watch_drink = trans.get(uid, {}).get("Watching", {}).get("Drinking", 0)
        lines.append(f"  Watching → Drinking: {watch_drink} times")
        lines.append(f"  Proactive trigger ready: {'YES' if watch_drink >= 3 else 'NO'}")
        lines.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp4] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db    = MongoClient(MONGO_URI)[DB_BASELINE]
    trans = load_transition_counts(db)
    obs   = load_observation_logs(db)

    total_obs = sum(len(v) for v in obs.values())
    total_trans = sum(
        sum(sum(tos.values()) for tos in frms.values())
        for frms in trans.values())

    if total_obs == 0:
        print(f"[exp4] No observation_logs in {DB_BASELINE}")
        return

    print(f"[exp4] {total_obs} observations, {total_trans} transitions loaded")

    plot_transition_comparison(
        trans, os.path.join(RESULTS_DIR, "exp4_transition_comparison.png"))

    plot_top_habits(
        obs, os.path.join(RESULTS_DIR, "exp4_top_habits.png"))

    plot_timeslot_heatmap(
        obs, os.path.join(RESULTS_DIR, "exp4_timeslot_heatmap.png"))

    plot_watching_to_drinking(
        trans, os.path.join(RESULTS_DIR, "exp4_watching_drinking.png"))

    plot_object_preference(
        db, os.path.join(RESULTS_DIR, "exp4_object_preference.png"))

    save_summary(trans, obs, os.path.join(RESULTS_DIR, "exp4_summary.txt"))

    print("\n[exp4] Done.")


if __name__ == "__main__":
    main()