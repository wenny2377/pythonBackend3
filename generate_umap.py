# generate_umap.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pymongo import MongoClient
from collections import Counter

try:
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("請先安裝：pip install umap-learn hdbscan scikit-learn matplotlib")
    exit()

# ── 連接 DB ──
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["robot_rag_db"]

docs = list(db.manifold_points.find(
    {"feature_vec": {"$exists": True}},
    {"feature_vec": 1, "action": 1, "user_id": 1,
     "timestamp": 1, "virtual_hour": 1, "prev_action": 1}
))

print(f"總筆數：{len(docs)}")
if len(docs) < 10:
    print("資料不足，請先跑 Exp4")
    exit()

X       = np.array([d["feature_vec"] for d in docs], dtype=np.float32)
actions = [d.get("action", "unknown") for d in docs]
users   = [d.get("user_id", "unknown") for d in docs]
prev_actions = [d.get("prev_action", "unknown") for d in docs]

print(f"特徵維度：{X.shape[1]}")
print(f"行為分布（原始）：")
for k, v in Counter(actions).most_common():
    print(f"  {v:3d}  {k}")

# ── 行為正規化 ──
ACTION_DISPLAY_MAP = {
    "Watching":    "Watching",
    "Standing":    "Standing",
    "Drink":       "Drink",
    "SittingIdle": "SittingIdle",
    "Reading":     "Reading",
    "Typing":      "Typing",
    "Sleeping":    "Sleeping",
    "Eating":      "Eating",
    "Exercising":  "Exercising",
    "Cooking":     "Cooking",
    "Walking":     "Walking",
}
KEYWORD_MAP = [
    ("sitting", "SittingIdle"), ("sit",  "SittingIdle"),
    ("standing","Standing"),    ("stand","Standing"),
    ("drinking","Drink"),       ("drink","Drink"),
    ("typing",  "Typing"),      ("type", "Typing"),
    ("reading", "Reading"),     ("read", "Reading"),
    ("sleeping","Sleeping"),    ("lying","Sleeping"),
    ("eating",  "Eating"),      ("watching","Watching"),
    ("walking", "Walking"),     ("cooking", "Cooking"),
]

def normalize_action(a: str) -> str:
    if a in ACTION_DISPLAY_MAP:
        return a
    a_lower = a.lower()
    for kw, label in KEYWORD_MAP:
        if kw in a_lower:
            return label
    return "Other"

actions_display      = [normalize_action(a) for a in actions]
prev_actions_display = [normalize_action(a) for a in prev_actions]

print(f"\n行為分布（正規化後）：{dict(Counter(actions_display).most_common())}")
print(f"前一個行為分布：{dict(Counter(prev_actions_display).most_common())}")

# ── 時段 ──
def to_slot(vh, ts):
    if vh is not None:
        h = float(vh)
    elif ts is not None:
        h = float(ts.hour)
    else:
        return "Unknown"
    if   h < 10: return "Morning"
    elif h < 13: return "Noon"
    elif h < 18: return "Afternoon"
    else:        return "Evening"

time_slots = [to_slot(d.get("virtual_hour"), d.get("timestamp")) for d in docs]
print(f"時段分布：{dict(Counter(time_slots).most_common())}")

# ── 行為轉換對（prev → current）──
transitions = [f"{p}→{c}" for p, c in zip(prev_actions_display, actions_display)]
print(f"\n最常見的行為轉換：")
for k, v in Counter(transitions).most_common(8):
    print(f"  {v:3d}  {k}")

# ── UMAP ──
print("\n跑 UMAP...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP(
    n_components = 2,
    n_neighbors  = min(15, len(docs) - 1),
    min_dist     = 0.1,
    metric       = "cosine",
    random_state = 42,
)
embedding = reducer.fit_transform(X_scaled)
print(f"UMAP 完成，shape={embedding.shape}")

# ── HDBSCAN ──
print("跑 HDBSCAN...")
clusterer      = hdbscan.HDBSCAN(min_cluster_size=max(3, len(docs) // 10))
cluster_labels = clusterer.fit_predict(embedding)
n_clusters     = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"找到 {n_clusters} 個 cluster")

# ── Silhouette Score ──
valid_mask = cluster_labels != -1
if valid_mask.sum() > 1 and n_clusters > 1:
    s_score = silhouette_score(embedding[valid_mask], cluster_labels[valid_mask])
    print(f"Silhouette Score: {s_score:.4f}")
else:
    s_score = None
    print("⚠️  cluster 數不足，無法計算 Silhouette Score")

# ── 顏色設定 ──
BEHAVIOR_COLORS = {
    'Watching':    '#E24B4A',
    'Standing':    '#8B5CF6',
    'Drink':       '#059669',
    'SittingIdle': '#F59E0B',
    'Reading':     '#0891B2',
    'Typing':      '#D97706',
    'Sleeping':    '#6366F1',
    'Eating':      '#EC4899',
    'Exercising':  '#14B8A6',
    'Walking':     '#84CC16',
    'Other':       '#9CA3AF',
    'unknown':     '#9CA3AF',
}
SLOT_COLORS = {
    'Morning':   '#F0B429',
    'Noon':      '#FF7043',
    'Afternoon': '#2563EB',
    'Evening':   '#059669',
    'Unknown':   '#888888',
}
SLOT_MARKERS = {
    'Morning': 'o', 'Noon': 's',
    'Afternoon': '^', 'Evening': 'D', 'Unknown': 'x'
}

# ── 畫圖：1x3 布局 ──
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
title = f"UMAP Behavioral Manifold — Exp4 (n={len(docs)},  dim={X.shape[1]})"
if s_score:
    title += f"    Silhouette Score = {s_score:.4f}"
fig.suptitle(title, fontsize=13, fontweight='bold')

# ────────────────────────────────────────────────────────────
# 左圖：按行為上色
# ────────────────────────────────────────────────────────────
ax1 = axes[0]
for behavior, color in BEHAVIOR_COLORS.items():
    mask = np.array([a == behavior for a in actions_display])
    if mask.sum() == 0:
        continue
    ax1.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=color, label=f"{behavior} (n={mask.sum()})",
        alpha=0.75, s=55,
        edgecolors='white', linewidths=0.4
    )
ax1.set_title("Color by Behavior", fontsize=12)
ax1.set_xlabel("UMAP dim 1")
ax1.set_ylabel("UMAP dim 2")
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.25)

# ────────────────────────────────────────────────────────────
# 中圖：按時段上色
# ────────────────────────────────────────────────────────────
ax2 = axes[1]
for slot, color in SLOT_COLORS.items():
    mask = np.array([t == slot for t in time_slots])
    if mask.sum() == 0:
        continue
    ax2.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=color, label=f"{slot} (n={mask.sum()})",
        alpha=0.75, s=55,
        marker=SLOT_MARKERS[slot],
        edgecolors='white', linewidths=0.4
    )
ax2.set_title("Color by Time Slot", fontsize=12)
ax2.set_xlabel("UMAP dim 1")
ax2.set_ylabel("UMAP dim 2")
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.25)

# ────────────────────────────────────────────────────────────
# 右圖：行為轉換箭頭（序列上下文視覺化）
# 每個點按「當前行為」上色，並畫箭頭指向「下一個點」
# 箭頭顏色代表轉換類型（同行為=灰，不同行為=紅）
# ────────────────────────────────────────────────────────────
ax3 = axes[2]

# 先畫所有點（按當前行為上色）
for behavior, color in BEHAVIOR_COLORS.items():
    mask = np.array([a == behavior for a in actions_display])
    if mask.sum() == 0:
        continue
    ax3.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=color, alpha=0.6, s=45,
        edgecolors='white', linewidths=0.3,
        zorder=2,
    )

# 畫轉換箭頭（只畫前 N 條，避免太亂）
MAX_ARROWS = 60
step = max(1, len(docs) // MAX_ARROWS)
for i in range(0, len(docs) - 1, step):
    x0, y0 = embedding[i, 0], embedding[i, 1]
    x1, y1 = embedding[i + 1, 0], embedding[i + 1, 1]
    same_behavior = (actions_display[i] == actions_display[i + 1])
    arrow_color = "#6B7280" if same_behavior else "#EF4444"
    alpha       = 0.25 if same_behavior else 0.55
    ax3.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=arrow_color,
            alpha=alpha,
            lw=1.2,
        ),
        zorder=3,
    )

# 標記最常見的轉換
top_transitions = Counter(transitions).most_common(3)
transition_text = "Top transitions:\n" + "\n".join(
    f"  {k} ({v})" for k, v in top_transitions
)
ax3.text(
    0.02, 0.98, transition_text,
    transform=ax3.transAxes,
    fontsize=8, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4)
)

# 圖例
legend_patches = [
    mpatches.Patch(color='#6B7280', alpha=0.5, label='Same behavior transition'),
    mpatches.Patch(color='#EF4444', alpha=0.8, label='Behavior change'),
]
ax3.legend(handles=legend_patches, loc='lower right', fontsize=8)
ax3.set_title("Behavior Transition (Sequence Context)", fontsize=12)
ax3.set_xlabel("UMAP dim 1")
ax3.set_ylabel("UMAP dim 2")
ax3.grid(True, alpha=0.25)

plt.tight_layout()
outfile = "umap_exp4.png"
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\n✅ 圖片已存成 {outfile}")
plt.close()

# ── 印出 cluster 組成 ──
print("\n── Cluster 組成 ──")
for cid in sorted(set(cluster_labels)):
    mask    = cluster_labels == cid
    acts    = [actions_display[i]      for i in range(len(docs)) if mask[i]]
    prevs   = [prev_actions_display[i] for i in range(len(docs)) if mask[i]]
    slots   = [time_slots[i]           for i in range(len(docs)) if mask[i]]
    users_c = [users[i]                for i in range(len(docs)) if mask[i]]
    label   = "noise" if cid == -1 else f"Cluster {cid}"
    print(f"  {label} (n={mask.sum()}): "
          f"行為={Counter(acts).most_common(2)}, "
          f"prev={Counter(prevs).most_common(2)}, "
          f"時段={Counter(slots).most_common(2)}, "
          f"用戶={Counter(users_c).most_common(2)}")

# ── 印出序列轉換矩陣 ──
print("\n── 行為轉換矩陣（row=prev, col=current, 格子=次數）──")
all_behaviors = sorted(set(actions_display + prev_actions_display) - {"Other", "unknown"})
header = f"{'':>12}" + "".join(f"{b:>12}" for b in all_behaviors)
print(header)
for prev in all_behaviors:
    row = f"{prev:>12}"
    for curr in all_behaviors:
        count = sum(
            1 for p, c in zip(prev_actions_display, actions_display)
            if p == prev and c == curr
        )
        row += f"{count:>12}" if count > 0 else f"{'·':>12}"
    print(row)