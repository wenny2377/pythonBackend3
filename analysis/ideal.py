"""
generate_target_charts.py
根據實際系統設計生成目標圖（假數據版本）
讓你知道實驗理想上要長什麼樣子

保留的圖：
  Fig.A  混淆矩陣（行為辨識）
  Fig.C  Ablation Study（各層貢獻）
  Fig.F  FAT Sensitivity（習慣學習門檻）
  Table  Mom vs Dad 個人化對比
  Fig.CR Correction Rate（回饋修正）

刪掉的圖：
  Fig.D  習慣學習曲線（3天數據，沒資訊）
  Fig.G  Affinity 收斂（不直觀）
  Fig.E  Zone Discrimination（不是核心貢獻）
  Exp.B  Entropy Heatmap（沒有實驗數據支撐）
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.titlesize": 13,
    "figure.dpi":       150,
})

OUT = "target_charts"
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


# ══════════════════════════════════════════════════════════════
# Fig.A  混淆矩陣
# 理想：高特異性行為對角線 1.00，低特異性行為有合理誤判
# ══════════════════════════════════════════════════════════════
def plot_fig_a():
    print("Fig.A: Confusion Matrix...")
    n = len(BEHAVIORS)

    # 設計理想的混淆矩陣
    # 對角線 = recall rate
    # 非對角線 = 合理的誤判模式
    diag = {
        "Drinking":    0.80,
        "SittingDrink":0.62,
        "Sitting":     1.00,
        "Eating":      0.58,
        "Cooking":     0.83,
        "Opening":     0.75,
        "Laying":      1.00,
        "Watching":    0.50,
        "Reading":     0.54,
        "Cleaning":    0.73,
        "PhoneUse":    1.00,
        "Typing":      1.00,
    }

    # 誤判模式（哪些行為容易被誤判成哪些）
    confusions = {
        "SittingDrink": {"Sitting": 0.25, "Drinking": 0.13},
        "Eating":       {"PhoneUse": 0.28, "Sitting": 0.14},
        "Opening":      {"Drinking": 0.12, "PhoneUse": 0.13},
        "Watching":     {"Sitting": 0.33, "SittingDrink": 0.17},
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

    correct = sum(diag[b] for b in BEHAVIORS) / n
    high_acc   = np.mean([diag[b] for b in BEHAVIORS if b in HIGH])
    medium_acc = np.mean([diag[b] for b in BEHAVIORS if b in MEDIUM])
    low_acc    = np.mean([diag[b] for b in BEHAVIORS if b in LOW])

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

    total_n = 480
    correct_n = int(correct * total_n)
    ax.set_title(
        f"Fig.A  Behaviour Recognition Confusion Matrix\n"
        f"Overall Acc = {correct:.1%} ({correct_n}/{total_n})  |  "
        f"High: {high_acc:.1%}  Medium: {medium_acc:.1%}  Low: {low_acc:.1%}\n"
        f"[Red=High-specificity  Orange=Medium  Blue=Low-specificity]",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUT, "FigA_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Overall: {correct:.1%}  High: {high_acc:.1%}  Low: {low_acc:.1%}")


# ══════════════════════════════════════════════════════════════
# Fig.C  Ablation Study
# 理想：每層都有正向貢獻，骨架層貢獻最大
# ══════════════════════════════════════════════════════════════
def plot_fig_c():
    print("Fig.C: Ablation Study...")

    configs = [
        ("VLM Only\n(Baseline)",           0.219, "#BDBDBD"),
        ("+ Skeleton\n(hip+head)",          0.656, "#2196F3"),
        ("+ Geometry\n(affinity+ray)",      0.695, "#4CAF50"),
        ("+ Object Context\n(held+nearby)", 0.725, "#FF9800"),
        ("Full System\n(+temporal)",        0.752, "#F44336"),
    ]

    total_n = 480
    labels  = [c[0] for c in configs]
    accs    = [c[1] for c in configs]
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
                    ha="center", fontsize=9,
                    color="white", fontweight="bold")

    for i in range(1, len(accs)):
        x1 = i - 1 + 0.32
        x2 = i     - 0.32
        y  = max(accs[i-1], accs[i]) * 100 + 8
        ax.annotate("",
                    xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#616161", lw=1.5))

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Fig.C  Ablation Study — Incremental Layer Contribution\n"
        f"Total episodes = {total_n} | VLM-only baseline = 21.9% | Full system = 75.2%\n"
        f"Skeleton layer provides largest single gain (+43.7%)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "FigC_ablation_study.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.F  FAT Sensitivity
# 理想：FAT=5 的 F1 最高，兩邊下降
# Recall 隨 FAT 增大而下降（太保守）
# Precision 隨 FAT 增大而上升（更嚴格）
# F1 在 FAT=5 達到峰值
# ══════════════════════════════════════════════════════════════
def plot_fig_f():
    print("Fig.F: FAT Sensitivity...")

    thrs = [2, 3, 5, 8, 10]
    x    = np.arange(len(thrs))

    # Mom：FAT=5 F1 最高
    recall_mom    = [1.00, 0.92, 0.82, 0.58, 0.40]
    precision_mom = [0.58, 0.74, 0.91, 0.96, 1.00]
    f1_mom = [2*p*r/(p+r) if p+r > 0 else 0
              for p, r in zip(precision_mom, recall_mom)]

    # Dad：FAT=5 F1 最高
    recall_dad    = [1.00, 0.90, 0.78, 0.52, 0.35]
    precision_dad = [0.55, 0.70, 0.88, 0.94, 1.00]
    f1_dad = [2*p*r/(p+r) if p+r > 0 else 0
              for p, r in zip(precision_dad, recall_dad)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle(
        "Fig.F  FAT Threshold Sensitivity\n"
        "Precision / Recall / F1 across FAT values; selected threshold = FAT=5",
        fontsize=12, fontweight="bold"
    )

    datasets = [
        ("User Mom", recall_mom, precision_mom, f1_mom),
        ("User Dad", recall_dad, precision_dad, f1_dad),
    ]

    for ax, (title, recall, precision, f1) in zip(axes, datasets):
        fat5_idx = thrs.index(5)
        ax.axvline(x=fat5_idx, color="#E53935", linewidth=1.8,
                   linestyle="--", alpha=0.7, label="FAT=5 (selected)")

        ax.plot(x, recall,    "o-", color="#2196F3", linewidth=2.2,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, label="Recall")
        ax.plot(x, precision, "s-", color="#FF9800", linewidth=2.2,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, label="Precision")
        ax.plot(x, f1,        "^-", color="#4CAF50", linewidth=2.5,
                markersize=9, markerfacecolor="white",
                markeredgewidth=2.5, label="F1")

        for i, (r, p, f) in enumerate(zip(recall, precision, f1)):
            ax.text(i, r + 0.02, f"{r:.2f}", ha="center",
                    fontsize=8, color="#1565C0")
            ax.text(i, p - 0.06, f"{p:.2f}", ha="center",
                    fontsize=8, color="#E65100")
            ax.text(i, f + 0.02, f"{f:.2f}", ha="center",
                    fontsize=8, color="#2E7D32")

        ax.set_xticks(x)
        ax.set_xticklabels([f"FAT={v}" for v in thrs], fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Fast Adaptation Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUT, "FigF_fat_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Mom FAT=5 F1: {f1_mom[thrs.index(5)]:.3f}")
    print(f"  Dad FAT=5 F1: {f1_dad[thrs.index(5)]:.3f}")


# ══════════════════════════════════════════════════════════════
# Fig.CR  Correction Rate
# 理想：拒絕後，下一輪錯誤推薦率從 100% 降到 0%
# 這個圖最直觀，委員一眼就懂
# ══════════════════════════════════════════════════════════════
def plot_fig_cr():
    print("Fig.CR: Correction Rate...")

    rounds = [1, 2, 3, 4, 5]

    # Mom 拒絕果汁後的錯誤推薦率
    # Round 1：100%（第一次推薦果汁，Mom 拒絕）
    # Round 2：0%（系統記住，不再推薦果汁）
    # Round 3-5：0%（持續正確）
    error_rate_mom = [100, 0, 0, 0, 0]

    # Dad 拒絕可樂後的錯誤推薦率
    error_rate_dad = [100, 0, 0, 0, 0]

    # 無個人化的基線（每次都推薦同樣的東西）
    baseline = [100, 100, 100, 100, 100]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(rounds, baseline,       "x--", color="#BDBDBD", linewidth=1.5,
            markersize=8, label="No Learning (Baseline)")
    ax.plot(rounds, error_rate_mom, "o-",  color="#2196F3", linewidth=2.5,
            markersize=10, markerfacecolor="white",
            markeredgewidth=2.5, label="User Mom (rejected juice)")
    ax.plot(rounds, error_rate_dad, "s-",  color="#4CAF50", linewidth=2.5,
            markersize=10, markerfacecolor="white",
            markeredgewidth=2.5, label="User Dad (rejected cola)")

    ax.axvline(x=1.5, color="#E53935", linewidth=1.5,
               linestyle=":", alpha=0.7, label="Rejection event")

    ax.annotate("Rejection\nrecorded",
                xy=(1, 100), xytext=(1.6, 80),
                fontsize=9, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5))

    ax.annotate("SKILL.md updated\n→ 0% error rate",
                xy=(2, 0), xytext=(2.5, 30),
                fontsize=9, color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=10)
    ax.set_ylim(-10, 120)
    ax.set_xlabel("Dialogue Round", fontsize=12)
    ax.set_ylabel("Wrong Recommendation Rate (%)", fontsize=12)
    ax.set_title(
        "Fig.CR  Correction Rate — Feedback Learning Effectiveness\n"
        "After one rejection, system immediately stops recommending the rejected item\n"
        "SKILL.md 'What NOT to do' section updated after Round 1",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "FigCR_correction_rate.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Table  Mom vs Dad 個人化對比
# 用圖表形式呈現，比純文字更清楚
# ══════════════════════════════════════════════════════════════
def plot_personalization_table():
    print("Table: Mom vs Dad Personalization...")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    columns = ["Time Slot", "Behaviour Pattern",
               "User Mom", "User Dad"]

    rows = [
        ["Morning\n07:00",
         "Kitchen / Workspace",
         "Opening → Cooking → Eating\n(kitchen-oriented)",
         "Opening → Eating → Typing\n(work-oriented)"],
        ["Noon\n12:00",
         "Rest / Lunch",
         "Sitting → Reading → Laying\n(reading nap)",
         "Laying → Watching\n(TV nap)"],
        ["Afternoon\n15:00",
         "Activity",
         "Cleaning → Reading\n(chores + reading)",
         "Typing → PhoneUse\n(work + digital)"],
        ["Evening\n19:00",
         "Dinner / Leisure",
         "Cooking → Eating → Watching\n(kitchen → living room)",
         "Eating → PhoneUse → SittingDrink\n(digital-oriented)"],
        ["Night\n23:00",
         "Sleep Prep",
         "Reading → Laying\n(reading to sleep)",
         "PhoneUse → Watching → Laying\n(TV to sleep)"],
        ["Drink\nPreference",
         "Personalised Service",
         "🧃 Juice\n(learned from observation)",
         "🥤 Cola\n(learned from observation)"],
        ["Service\nProposal",
         "Proactive Recommendation",
         "\"Would you like some juice\nwhile watching TV?\"",
         "\"Would you like some cola\nwith your phone?\""],
    ]

    col_widths = [0.10, 0.18, 0.36, 0.36]
    col_colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5"]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.2)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif r == len(rows):
            cell.set_facecolor("#E53935")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(rows) - 1:
            cell.set_facecolor("#FF9800")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor(col_colors[c] if c < len(col_colors) else "white")
        cell.set_edgecolor("#CCCCCC")

    ax.set_title(
        "Table: Mom vs Dad Personalised Habit Profile\n"
        "Learned passively from observation — no explicit user configuration required",
        fontsize=12, fontweight="bold", pad=20
    )

    plt.tight_layout()
    path = os.path.join(OUT, "Table_mom_vs_dad_personalization.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Target Chart Generator (Simulated Data)")
    print("These show what the ideal results should look like")
    print("=" * 60)

    plot_fig_a()
    plot_fig_c()
    plot_fig_f()
    plot_fig_cr()
    plot_personalization_table()

    print("\n" + "=" * 60)
    print(f"Done. Check {OUT}/")
    print()
    print("Figure index:")
    print("  FigA  Confusion Matrix        → Layer 1 辨識準確率")
    print("  FigC  Ablation Study          → Layer 1 各層貢獻（最重要）")
    print("  FigF  FAT Sensitivity         → Layer 2 習慣學習門檻")
    print("  FigCR Correction Rate         → Layer 3 回饋修正")
    print("  Table Mom vs Dad              → Layer 2+3 個人化對比")
    print()
    print("用真實數據取代模擬數據的步驟：")
    print("  FigA, FigC → 跑完 HabitExp，python3 exp2_habit.py --only A,C")
    print("  FigF       → 跑完 30 天 HabitExp，python3 exp2_habit.py --only F")
    print("  FigCR      → 執行 Correction Rate 實驗腳本")
    print("  Table      → 從 SKILL.md 讀取真實內容")
    print("=" * 60)