import os, sys, json, requests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

from exp_config import (
    MONGO_URI, DB_BASELINE, BACKEND_URL, ADL_LABELS, USERS,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    LINE_WIDTH, FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

# ── Ablation: re-run LLM with masked modality ─────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL  = "llama3.1:8b"

BEHAVIOR_LABELS = ADL_LABELS + [
    "Opening", "StandUp", "Standing", "Walking", "PickingUp", "PuttingDown"
]

BODY_IMPOSSIBLE = {
    "lying":    {"Typing", "Cooking", "Cleaning", "Eating", "Reading",
                 "PhoneUse", "Watching", "Drinking", "SittingDrink", "Opening"},
    "standing": {"Laying"},
}

ROOM_IMPOSSIBLE = {
    "DadRoom":    {"Cooking", "Cleaning"},
    "Kitchen":    {"Laying", "Typing"},
    "LivingRoom": {"Typing", "Cooking"},
}


def _prune_candidates(body_pos: str, room: str) -> list:
    candidates = set(ADL_LABELS)
    bp = (body_pos or "").lower()
    for key, blocked in BODY_IMPOSSIBLE.items():
        if key in bp:
            candidates -= blocked
    candidates -= ROOM_IMPOSSIBLE.get(room or "", set())
    return list(candidates) if candidates else ADL_LABELS


def _build_scene_text(doc: dict, mask: str) -> str:
    """
    Build LLM scene prompt from eval_log fields.
    mask: "skeleton" | "object" | "spatial" | None
    """
    parts = []

    # Always include: room, time slot
    room = doc.get("room_name", "Unknown")
    slot = doc.get("time_slot", "Unknown")
    parts.append(f"Room: {room}. Time: {slot}.")

    # Body posture (from VLM)
    body = doc.get("body_position", "unknown")
    orient = doc.get("body_orientation", "unknown")
    parts.append(f"Body position: {body}, orientation: {orient}.")

    # Skeleton features — masked in "skeleton" ablation
    if mask != "skeleton":
        skel_parts = []
        hp = doc.get("head_pitch")
        if hp is not None:
            if hp > 45:
                skel_parts.append("head bent far forward")
            elif hp > 20:
                skel_parts.append("head slightly forward")
        h2h = doc.get("hand_to_head")
        if h2h is not None:
            if h2h < 0.35:
                skel_parts.append("hand very close to face")
            elif h2h < 0.55:
                skel_parts.append("hand near face")
        wh = doc.get("wrist_height")
        if wh is not None:
            if wh > 0.3:
                skel_parts.append("wrist raised high")
            elif wh < -0.1:
                skel_parts.append("wrist low")
        skel_body = doc.get("skel_body", "")
        if skel_body:
            skel_parts.append(f"skeleton posture: {skel_body}")
        if skel_parts:
            parts.append("Skeleton cues: " + ", ".join(skel_parts) + ".")

    # Object / held event — masked in "object" ablation
    if mask != "object":
        held = doc.get("held_event", "none")
        if held and held not in ("none", "", "null"):
            parts.append(f"Object event: {held}.")
        items = doc.get("interacting_items", [])
        if items:
            parts.append(f"Nearby objects: {', '.join(items)}.")

    # Spatial context (zone, room) — masked in "spatial" ablation
    if mask != "spatial":
        zone = doc.get("zone_label", "")
        if zone and "Unknown" not in zone:
            parts.append(f"Nearest furniture zone: {zone}.")

    return " ".join(parts)


def _llm_predict(scene_text: str, body_pos: str, room: str) -> str:
    candidates = _prune_candidates(body_pos, room)
    prompt = (
        f"You are an activity recognition system.\n"
        f"Scene: {scene_text}\n"
        f"Choose ONE activity from: {', '.join(candidates)}\n"
        f"Reply with ONLY the activity name, nothing else."
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        for label in candidates:
            if label.lower() in raw.lower():
                return label
        return raw.split()[0] if raw else "Unknown"
    except Exception as e:
        print(f"  [LLM error] {e}")
        return "Unknown"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_accuracy(docs, predictions=None):
    total = correct = 0
    for i, d in enumerate(docs):
        gt   = d.get("ground_truth", "")
        pred = predictions[i] if predictions else (
            d.get("spatial_action") or d.get("vlm_output", ""))
        if gt in ADL_LABELS:
            total   += 1
            correct += int(gt == pred)
    return correct / total if total else 0.0, correct, total


def per_class_accuracy(docs, predictions=None):
    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for i, d in enumerate(docs):
        gt   = d.get("ground_truth", "")
        pred = predictions[i] if predictions else (
            d.get("spatial_action") or d.get("vlm_output", ""))
        if gt in ADL_LABELS:
            by_class[gt]["total"] += 1
            if gt == pred:
                by_class[gt]["tp"] += 1
    return by_class


# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(docs, save_path):
    present = [l for l in ADL_LABELS
               if any(d.get("ground_truth") == l for d in docs)]
    n = len(present)
    matrix = np.zeros((n, n), dtype=int)
    for d in docs:
        gt   = d.get("ground_truth", "")
        pred = d.get("spatial_action") or d.get("vlm_output", "")
        if gt in present and pred in present:
            matrix[present.index(gt)][present.index(pred)] += 1

    total   = int(matrix.sum())
    correct = int(np.trace(matrix))
    acc     = correct / total if total else 0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall Rate", fontsize=FONT_AXIS)

    for i in range(n):
        for j in range(n):
            v = norm[i][j]
            if matrix[i][j] > 0:
                ax.text(j, i,
                        f"{v:.2f}\n({matrix[i][j]})",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=FONT_TICK)
    ax.set_yticklabels(present, fontsize=FONT_TICK)
    ax.set_xlabel("Predicted", fontsize=FONT_AXIS)
    ax.set_ylabel("Ground Truth", fontsize=FONT_AXIS)
    ax.set_title(
        f"HAR Confusion Matrix — Baseline\n"
        f"Overall Accuracy: {acc:.1%}  ({correct}/{total} episodes)",
        fontsize=FONT_TITLE, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")
    return acc, correct, total


# ── Plot 2: Ablation ──────────────────────────────────────────────────────────

def run_ablation(docs):
    """
    True ablation: re-run LLM inference with each modality masked.
    Uses stored eval_log fields — no Unity re-run needed.
    """
    conditions = {
        "Full System":          None,
        "w/o Skeleton":         "skeleton",
        "w/o Object Events":    "object",
        "w/o Spatial Context":  "spatial",
    }

    results = {}
    baseline_preds = [d.get("spatial_action") or d.get("vlm_output","") for d in docs]
    full_acc, _, _ = compute_accuracy(docs, baseline_preds)
    results["Full System"] = full_acc

    for name, mask in conditions.items():
        if mask is None:
            continue
        print(f"  [Ablation] Running: {name} ...")
        preds = []
        for d in docs:
            scene = _build_scene_text(d, mask=mask)
            body  = d.get("body_position", "")
            room  = d.get("room_name", "")
            preds.append(_llm_predict(scene, body, room))
        acc, _, _ = compute_accuracy(docs, preds)
        results[name] = acc
        print(f"    → {acc:.1%}")

    return results


def plot_ablation(ablation_results, full_acc, save_path):
    names  = list(ablation_results.keys())
    accs   = [ablation_results[n] * 100 for n in names]
    deltas = [full_acc * 100 - a for a in accs]

    colors = [C["baseline"] if n == "Full System" else C["ablation"] for n in names]

    # Sort by accuracy ascending (worst first)
    order  = sorted(range(len(names)), key=lambda i: accs[i])
    names  = [names[i] for i in order]
    accs   = [accs[i] for i in order]
    deltas = [deltas[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(names)), accs, color=colors,
                   alpha=0.88, height=0.55, edgecolor="white")

    for i, (bar, acc, delta) in enumerate(zip(bars, accs, deltas)):
        label = f"{acc:.1f}%" if delta == 0 else f"{acc:.1f}%  (−{delta:.1f}%)"
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=FONT_ANNOT,
                color=C["highlight"] if delta > 5 else "#333",
                fontweight="bold" if delta > 5 else "normal")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=FONT_TICK)
    ax.set_xlabel("Recognition Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_xlim(0, 115)
    ax.set_title(
        "Modality Ablation Study\n"
        "(LLM re-inference with each modality masked)",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(docs, acc, correct, total, ablation_results, save_path):
    by_class = per_class_accuracy(docs)
    lines = [
        "Experiment 1: HAR Baseline + Ablation",
        f"DB: {DB_BASELINE}",
        f"Episodes: {total}  Correct: {correct}  Overall: {acc:.1%}",
        "",
        "Per-class Accuracy:",
        f"{'Action':<16} {'Acc':>6} {'TP':>5} {'Total':>7}",
        "-" * 38,
    ]
    for label in ADL_LABELS:
        info = by_class.get(label, {"tp": 0, "total": 0})
        if info["total"] == 0:
            continue
        a = info["tp"] / info["total"]
        lines.append(f"{label:<16} {a:>5.1%} {info['tp']:>5} {info['total']:>7}")

    lines += ["", "Ablation Results:"]
    for name, a in ablation_results.items():
        lines.append(f"  {name:<26} {a:.1%}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp1] Saved: {save_path}")
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_per_class_bar(docs, save_path):
    by_class = per_class_accuracy(docs)
    present  = [(l, by_class[l]) for l in ADL_LABELS
                if by_class[l]["total"] > 0]
    present.sort(key=lambda x: x[1]["tp"] / x[1]["total"])

    labels = [p[0] for p in present]
    accs   = [p[1]["tp"] / p[1]["total"] * 100 for p in present]
    totals = [p[1]["total"] for p in present]
    n      = len(labels)

    colors = []
    for a in accs:
        if a >= 80:
            colors.append(C["baseline"])
        elif a >= 60:
            colors.append("#F5A623")
        else:
            colors.append(C["corruption"])

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.65)))
    bars = ax.barh(range(n), accs, color=colors,
                   alpha=0.88, height=0.55, edgecolor="white")

    for i, (bar, acc, tot) in enumerate(zip(bars, accs, totals)):
        ax.text(min(bar.get_width() + 0.8, 101),
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%  ({int(acc/100*tot)}/{tot})",
                va="center", fontsize=FONT_ANNOT,
                color="#333", fontweight="bold" if acc < 80 else "normal")

    ax.axvline(80, color="#999", linestyle="--", lw=1.2, alpha=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_xlim(0, 120)
    ax.set_title("Per-class Recognition Accuracy — Baseline",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C["baseline"],    label="≥ 80%"),
        Patch(color="#F5A623",        label="60–79%"),
        Patch(color=C["corruption"],  label="< 60%"),
    ], fontsize=FONT_TICK, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")


def plot_ablation_table(ablation_results, full_acc, save_path):
    order  = ["Full System", "w/o Skeleton", "w/o Spatial Context", "w/o Object Events"]
    accs   = {k: ablation_results.get(k, full_acc) for k in order}
    deltas = {k: (full_acc - accs[k]) * 100 for k in order}

    col_headers = ["Method", "Skeleton", "Object\nEvents", "Spatial\nContext", "Acc (%)", "Δ"]
    modality_map = {
        "Full System":          (True,  True,  True),
        "w/o Skeleton":         (False, True,  True),
        "w/o Spatial Context":  (True,  True,  False),
        "w/o Object Events":    (True,  False, True),
    }

    rows = []
    for name in order:
        skel, obj, spa = modality_map[name]
        acc_val = accs[name] * 100
        delta_val = deltas[name]
        delta_str = "—" if name == "Full System" else f"−{delta_val:.1f}%"
        rows.append([
            name,
            "✓" if skel else "✗",
            "✓" if obj  else "✗",
            "✓" if spa  else "✗",
            f"{acc_val:.1f}%",
            delta_str,
        ])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    col_widths = [0.32, 0.1, 0.12, 0.13, 0.12, 0.11]
    table = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    header_color = "#2C3E50"
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    row_colors = ["#EBF5FB", "#FDFEFE", "#EBF5FB", "#FDFEFE"]
    highlight  = "#FDEDEC"
    for i, name in enumerate(order):
        for j in range(len(col_headers)):
            cell = table[i + 1, j]
            if name == "w/o Object Events":
                cell.set_facecolor(highlight)
            else:
                cell.set_facecolor(row_colors[i])
            if j in (1, 2, 3):
                txt = rows[i][j]
                cell.set_text_props(
                    color="#27AE60" if txt == "✓" else "#E74C3C",
                    fontweight="bold", fontsize=13)
            if j == 5 and name != "Full System":
                cell.set_text_props(
                    color="#E74C3C", fontweight="bold")

    ax.set_title("Modality Ablation Study — LLM Re-inference",
                 fontsize=FONT_TITLE, fontweight="bold", pad=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db   = MongoClient(MONGO_URI)[DB_BASELINE]
    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
    ))

    if not docs:
        print(f"[exp1] No eval_logs in {DB_BASELINE}")
        return

    print(f"[exp1] {len(docs)} episodes loaded")

    acc, correct, total = plot_confusion_matrix(
        docs, os.path.join(RESULTS_DIR, "exp1_confusion_matrix.png"))

    plot_per_class_bar(
        docs, os.path.join(RESULTS_DIR, "exp1_per_class_bar.png"))

    print(f"[exp1] Running ablation (this may take a few minutes)...")
    ablation = run_ablation(docs)
    ablation["Full System"] = acc
    plot_ablation(ablation, acc,
                  os.path.join(RESULTS_DIR, "exp1_ablation.png"))

    plot_ablation_table(ablation, acc,
                        os.path.join(RESULTS_DIR, "exp1_ablation_table.png"))

    save_summary(docs, acc, correct, total, ablation,
                 os.path.join(RESULTS_DIR, "exp1_summary.txt"))


if __name__ == "__main__":
    main()