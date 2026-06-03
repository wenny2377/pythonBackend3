"""
analysis/simulate_ideal.py
Simulated ideal data — target charts for thesis
Shows what results SHOULD look like after full experiment
Outputs to results_ideal/ (separate from real results)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "results_ideal")
os.makedirs(OUT, exist_ok=True)

BEHAVIORS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]
HIGH   = {"Cooking", "Opening", "Laying", "Cleaning", "PhoneUse", "Typing"}
MEDIUM = {"Eating", "Drinking"}
LOW    = {"SittingDrink", "Sitting", "Reading", "Watching"}

def group_color(b):
    if b in HIGH:   return "#F44336"
    if b in MEDIUM: return "#FF9800"
    return "#2196F3"


def plot_fig1():
    print("ideal Fig1: Confusion Matrix...")
    n = len(BEHAVIORS)
    diag = {
        "Drinking":    0.80, "SittingDrink": 0.62, "Sitting":  1.00,
        "Eating":      0.58, "Cooking":      0.83, "Opening":  0.75,
        "Laying":      1.00, "Watching":     0.65, "Reading":  0.54,
        "Cleaning":    0.73, "PhoneUse":     1.00, "Typing":   1.00,
    }
    confusions = {
        "SittingDrink": {"Sitting": 0.25, "Drinking": 0.13},
        "Eating":       {"PhoneUse": 0.28, "Sitting": 0.14},
        "Opening":      {"Drinking": 0.12, "PhoneUse": 0.13},
        "Watching":     {"Sitting": 0.22, "SittingDrink": 0.13},
        "Reading":      {"Sitting": 0.30, "Cleaning": 0.16},
        "Cleaning":     {"Sitting": 0.09, "Reading": 0.18},
        "Drinking":     {"Standing": 0.20},
    }
    matrix = np.zeros((n, n))
    for i, b in enumerate(BEHAVIORS):
        matrix[i][i] = diag.get(b, 0.90)
        for j, b2 in enumerate(BEHAVIORS):
            if b in confusions and b2 in confusions[b]:
                matrix[i][j] = confusions[b][b2]

    overall    = np.mean(list(diag.values()))
    high_acc   = np.mean([diag[b] for b in BEHAVIORS if b in HIGH])
    medium_acc = np.mean([diag[b] for b in BEHAVIORS if b in MEDIUM])
    low_acc    = np.mean([diag[b] for b in BEHAVIORS if b in LOW])
    total_n    = 480

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall Rate")
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(BEHAVIORS, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(BEHAVIORS, fontsize=9)
    for tick, b in zip(ax.get_xticklabels(), BEHAVIORS):
        tick.set_color(group_color(b))
    for tick, b in zip(ax.get_yticklabels(), BEHAVIORS):
        tick.set_color(group_color(b))
    ax.set_title(
        f"Fig1  Behaviour Recognition Confusion Matrix  [IDEAL]\n"
        f"Overall = {overall:.1%} ({int(overall*total_n)}/{total_n})  |  "
        f"High: {high_acc:.1%}  Medium: {medium_acc:.1%}  Low: {low_acc:.1%}\n"
        f"[Red=High-specificity  Orange=Medium  Blue=Low-specificity]",
        fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT, "Fig1_confusion.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fig2():
    print("ideal Fig2: Ablation Study...")
    configs = [
        ("VLM Only\n(Baseline)",            0.219, "#BDBDBD"),
        ("+ Skeleton\n(hip+head)",           0.656, "#2196F3"),
        ("+ Geometry\n(affinity+ray)",       0.695, "#4CAF50"),
        ("+ Object Context\n(held+nearby)",  0.725, "#FF9800"),
        ("Full System\n(+temporal)",         0.752, "#F44336"),
    ]
    total_n = 480
    accs    = [c[1] for c in configs]
    labels  = [c[0] for c in configs]
    colors  = [c[2] for c in configs]
    counts  = [int(a * total_n) for a in accs]
    deltas  = [0] + [accs[i] - accs[i-1] for i in range(1, len(accs))]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(configs)), [a * 100 for a in accs],
                  color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for i, (bar, acc, cnt, delta) in enumerate(zip(bars, accs, counts, deltas)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.1%}\n(n={cnt})",
                ha="center", fontsize=10, fontweight="bold")
        if i > 0:
            sign = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{sign}{delta:.1%}",
                    ha="center", fontsize=9, color="white", fontweight="bold")
    for i in range(1, len(accs)):
        x1 = i - 1 + 0.32
        x2 = i     - 0.32
        y  = max(accs[i-1], accs[i]) * 100 + 8
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#616161", lw=1.5))
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Fig2  Ablation Study — Incremental Layer Contribution  [IDEAL]\n"
        f"Total = {total_n}  |  VLM-only = 21.9%  →  Full system = 75.2%\n"
        f"Skeleton layer: largest single gain (+43.7%)",
        fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUT, "Fig2_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fig3():
    print("ideal Fig3: FAT Sensitivity...")
    thrs = [2, 3, 5, 8, 10]
    x    = np.arange(len(thrs))

    recall_mom    = [1.00, 0.92, 0.82, 0.58, 0.40]
    precision_mom = [0.58, 0.74, 0.91, 0.96, 1.00]
    f1_mom = [2*p*r/(p+r) if p+r > 0 else 0
              for p, r in zip(precision_mom, recall_mom)]

    recall_dad    = [1.00, 0.90, 0.78, 0.52, 0.35]
    precision_dad = [0.55, 0.70, 0.88, 0.94, 1.00]
    f1_dad = [2*p*r/(p+r) if p+r > 0 else 0
              for p, r in zip(precision_dad, recall_dad)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle(
        "Fig3  FAT Threshold Sensitivity  [IDEAL]\n"
        "Precision / Recall / F1; FAT=5 selected as optimal",
        fontsize=12, fontweight="bold")

    for ax, (title, recall, precision, f1) in zip(axes, [
        ("User Mom", recall_mom, precision_mom, f1_mom),
        ("User Dad", recall_dad, precision_dad, f1_dad),
    ]):
        ax.axvline(x=thrs.index(5), color="#E53935", linewidth=1.8,
                   linestyle="--", alpha=0.7, label="FAT=5 (selected)")
        ax.plot(x, recall,    "o-", color="#2196F3", linewidth=2.2,
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                label="Recall")
        ax.plot(x, precision, "s-", color="#FF9800", linewidth=2.2,
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                label="Precision")
        ax.plot(x, f1,        "^-", color="#4CAF50", linewidth=2.5,
                markersize=9, markerfacecolor="white", markeredgewidth=2.5,
                label="F1")
        for i, (r, p) in enumerate(zip(recall, precision)):
            ax.text(i, r + 0.02, f"{r:.2f}", ha="center",
                    fontsize=8, color="#1565C0")
            ax.text(i, p - 0.06, f"{p:.2f}", ha="center",
                    fontsize=8, color="#E65100")
        ax.set_xticks(x)
        ax.set_xticklabels([f"FAT={v}" for v in thrs], fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Fast Adaptation Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig3_fat_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fig4():
    print("ideal Fig4: Correction Rate...")
    rounds   = [1, 2, 3, 4, 5]
    err_mom  = [100, 0, 0, 0, 0]
    err_dad  = [100, 0, 0, 0, 0]
    baseline = [100, 100, 100, 100, 100]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(rounds, baseline, "x--", color="#BDBDBD", linewidth=1.5,
            markersize=8, label="No Learning (Baseline)")
    ax.plot(rounds, err_mom, "o-", color="#2196F3", linewidth=2.5,
            markersize=10, markerfacecolor="white", markeredgewidth=2.5,
            label="User Mom (rejected juice)")
    ax.plot(rounds, err_dad, "s-", color="#4CAF50", linewidth=2.5,
            markersize=10, markerfacecolor="white", markeredgewidth=2.5,
            label="User Dad (rejected cola)")
    ax.axvline(x=1.5, color="#E53935", linewidth=1.5,
               linestyle=":", alpha=0.7, label="Rejection event")
    ax.annotate("Rejection\nrecorded",
                xy=(1, 100), xytext=(1.6, 80), fontsize=9, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5))
    ax.annotate("SKILL.md updated\n→ 0% error rate",
                xy=(2, 0), xytext=(2.5, 30), fontsize=9, color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=10)
    ax.set_ylim(-10, 120)
    ax.set_xlabel("Dialogue Round", fontsize=12)
    ax.set_ylabel("Wrong Recommendation Rate (%)", fontsize=12)
    ax.set_title(
        "Fig4  Correction Rate — Feedback Learning Effectiveness  [IDEAL]\n"
        "One rejection → immediate correction → 0% error from Round 2",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUT, "Fig4_correction_rate.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fig5():
    print("ideal Fig5: Personalization Table...")
    rows = [
        ["Morning 07:00",
         "Opening → Cooking → Eating\n(kitchen-oriented)",
         "Opening → Eating → Typing\n(work-oriented)"],
        ["Noon 12:00",
         "Sitting → Reading → Laying\n(reading nap)",
         "Laying → Watching\n(TV nap)"],
        ["Afternoon 15:00",
         "Cleaning → Reading\n(chores + reading)",
         "Typing → PhoneUse\n(work + digital)"],
        ["Evening 19:00",
         "Cooking → Eating → Watching\n(kitchen → living room)",
         "Eating → PhoneUse → SittingDrink\n(digital-oriented)"],
        ["Night 23:00",
         "Reading → Laying\n(reading to sleep)",
         "PhoneUse → Watching → Laying\n(TV to sleep)"],
        ["Drink Preference",
         "Juice\n(learned from observation)",
         "Cola\n(learned from observation)"],
        ["Service Proposal",
         "\"Would you like some juice\nwhile watching TV?\"",
         "\"Would you like some cola\nwith your phone?\""],
    ]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Time / Category", "User Mom", "User Dad"],
        cellLoc="center", loc="center",
        colWidths=[0.18, 0.41, 0.41],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 3.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        elif c == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#FFF8E1")
        elif c == 2:
            cell.set_facecolor("#F3E5F5")
        if r == len(rows) and c > 0:
            cell.set_facecolor("#E8F5E9" if c == 1 else "#FCE4EC")
    ax.set_title(
        "Fig5  Mom vs Dad Personalised Habit Profile  [IDEAL]\n"
        "Learned passively from observation — no explicit user configuration required",
        fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(OUT, "Fig5_personalization.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 55)
    print("Simulated Ideal Charts → results_ideal/")
    print("=" * 55)
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    print()
    print("Done. All ideal charts saved to results_ideal/")
    print()
    print("To generate real results (after experiment):")
    print("  python3 analysis/layer1_recognition.py")
    print("  python3 analysis/layer2_habit.py")
    print("  python3 analysis/layer3_correction.py")
    print("  python3 analysis/table_compare.py")