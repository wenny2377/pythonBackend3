"""
umap_dashboard.py  ── 口試 Demo 版（穩定 layout）
=====================================================
關鍵改進：
  - UMAP fit once（資料達到 MIN_FIT_PTS 時固定 layout）
  - 新資料用 transform() 投影，layout 不跳動
  - 軌跡箭頭平滑移動
  - 即時顯示當前行為 + 前一個行為

執行：
  python3 umap_dashboard.py            # live mode
  python3 umap_dashboard.py --replay   # replay mode
  python3 umap_dashboard.py --replay --speed 2
"""

import time
import argparse
import datetime
import threading
import numpy as np
from collections import Counter
from pymongo import MongoClient

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False
    print("請安裝：pip install umap-learn hdbscan dash plotly scikit-learn")

# ── 參數 ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--replay", action="store_true")
parser.add_argument("--speed",  type=float, default=1.0)
parser.add_argument("--port",   type=int,   default=8050)
args, _ = parser.parse_known_args()

MONGO_URI    = "mongodb://127.0.0.1:27017/"
DB_NAME      = "robot_rag_db"
REFRESH_MS   = 1500
TRAIL_LEN    = 12     # 軌跡保留點數（更長更好看）
MIN_PTS      = 10     # 最少幾筆才跑 UMAP
MIN_FIT_PTS  = 30     # 達到這個數量才固定 layout

# ── 顏色 ──────────────────────────────────────────────────────
BG_DARK  = "#0A0E1A"
BG_CARD  = "#111827"
BG_PLOT  = "#0D1526"
BORDER   = "#1F2D45"
TEXT_PRI = "#E2E8F0"
TEXT_SEC = "#64748B"
ACCENT   = "#3B82F6"

BEHAVIOR_COLORS = {
    "Watching":    "#EF4444",
    "Standing":    "#8B5CF6",
    "Drink":       "#10B981",
    "SittingIdle": "#F59E0B",
    "Reading":     "#3B82F6",
    "Typing":      "#F97316",
    "Sleeping":    "#6366F1",
    "Eating":      "#EC4899",
    "Other":       "#6B7280",
    "unknown":     "#6B7280",
}
SLOT_COLORS = {
    "Morning":   "#FBBF24",
    "Noon":      "#FB923C",
    "Afternoon": "#60A5FA",
    "Evening":   "#34D399",
    "Unknown":   "#6B7280",
}
SLOT_SYMBOLS = {
    "Morning":"circle", "Noon":"square",
    "Afternoon":"triangle-up", "Evening":"diamond", "Unknown":"x",
}
USER_COLORS = {"User_Mom": "#F472B6", "User_Dad": "#38BDF8"}

KEYWORD_MAP = [
    ("sitting","SittingIdle"), ("sit","SittingIdle"),
    ("standing","Standing"),   ("stand","Standing"),
    ("drinking","Drink"),      ("drink","Drink"),
    ("typing","Typing"),       ("type","Typing"),
    ("watching","Watching"),   ("reading","Reading"),
    ("sleeping","Sleeping"),   ("lying","Sleeping"),
    ("eating","Eating"),
]

def norm_action(a):
    if not a or a == "unknown": return "unknown"
    if a in BEHAVIOR_COLORS: return a
    al = a.lower()
    for kw, label in KEYWORD_MAP:
        if kw in al: return label
    return "Other"

def to_slot(vh, ts):
    if vh is not None: h = float(vh)
    elif ts: h = float(ts.hour) if hasattr(ts,"hour") else 0
    else: return "Unknown"
    if   h < 10: return "Morning"
    elif h < 13: return "Noon"
    elif h < 18: return "Afternoon"
    else:        return "Evening"

# ── Replay ────────────────────────────────────────────────────
_replay_state = {
    "all_docs":   [],
    "cursor":     0,
    "running":    False,
    "interval_s": 2.0 / max(args.speed, 0.1),
}

def _load_replay_data():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db     = client[DB_NAME]
        docs   = list(db.manifold_points.find(
            {"feature_vec": {"$exists": True}},
            {"feature_vec":1,"action":1,"user_id":1,
             "timestamp":1,"virtual_hour":1,"prev_action":1}
        ).sort("timestamp", 1))
        client.close()
        _replay_state["all_docs"] = docs
        print(f"[Replay] 載入 {len(docs)} 筆")
        return len(docs)
    except Exception as e:
        print(f"[Replay] 失敗: {e}")
        return 0

def _replay_ticker():
    while True:
        time.sleep(_replay_state["interval_s"])
        if _replay_state["running"]:
            if _replay_state["cursor"] < len(_replay_state["all_docs"]):
                _replay_state["cursor"] += 1

if args.replay:
    n = _load_replay_data()
    _replay_state["running"] = True
    _replay_state["cursor"]  = min(10, n)
    threading.Thread(target=_replay_ticker, daemon=True).start()
    print(f"[Replay] 啟動 x{args.speed}")

# ── UMAP 狀態（fit once）─────────────────────────────────────
_umap_state = {
    "fitted":    False,       # 是否已 fit
    "n_fit":     0,           # fit 時的資料量
    "reducer":   None,        # umap.UMAP 模型
    "scaler":    None,        # StandardScaler
    "clusterer": None,        # hdbscan 模型
    "labels":    None,        # cluster labels（fit 時的）
    "s_score":   None,
    # 所有點的 embedding（包含 fit 後 transform 的新點）
    "all_xy":    None,        # shape (N, 2)
    "all_ids":   [],          # 對應的 doc _id
    "lock":      threading.Lock(),
}

def _fit_umap(docs):
    """第一次 fit，固定 layout"""
    X = np.array([d["feature_vec"] for d in docs], dtype=np.float32)
    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(docs)-1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    xy = reducer.fit_transform(X_s)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(3, len(docs)//10),
        prediction_data=True,
    )
    labels = clusterer.fit_predict(xy)

    valid = labels != -1
    s_score = None
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    if valid.sum() > 1 and n_cl > 1:
        s_score = silhouette_score(xy[valid], labels[valid])

    with _umap_state["lock"]:
        _umap_state["fitted"]    = True
        _umap_state["n_fit"]     = len(docs)
        _umap_state["reducer"]   = reducer
        _umap_state["scaler"]    = scaler
        _umap_state["clusterer"] = clusterer
        _umap_state["labels"]    = labels
        _umap_state["s_score"]   = s_score
        _umap_state["all_xy"]    = xy
        _umap_state["all_ids"]   = [str(d["_id"]) for d in docs]

    print(f"[UMAP] fit 完成 n={len(docs)} clusters={n_cl} S={s_score:.4f if s_score else 'N/A'}")

def _transform_new(docs):
    """有新資料時，用 transform() 投影，不重算 layout"""
    new_docs = [d for d in docs
                if str(d["_id"]) not in _umap_state["all_ids"]]
    if not new_docs:
        return

    X_new = np.array([d["feature_vec"] for d in new_docs], dtype=np.float32)
    X_s   = _umap_state["scaler"].transform(X_new)
    xy_new = _umap_state["reducer"].transform(X_s)

    with _umap_state["lock"]:
        _umap_state["all_xy"]  = np.vstack([_umap_state["all_xy"], xy_new])
        _umap_state["all_ids"] += [str(d["_id"]) for d in new_docs]

    print(f"[UMAP] transform +{len(new_docs)} 筆，total={len(_umap_state['all_ids'])}")

def update_umap(docs):
    """主要更新函式：自動決定 fit 或 transform"""
    if not _DEPS_OK or len(docs) < MIN_PTS:
        return False

    if not _umap_state["fitted"]:
        if len(docs) >= MIN_FIT_PTS:
            _fit_umap(docs)
            return True
        else:
            # 資料不夠 30 筆，用臨時 fit（不鎖定）
            try:
                X   = np.array([d["feature_vec"] for d in docs], dtype=np.float32)
                X_s = StandardScaler().fit_transform(X)
                xy  = umap.UMAP(n_components=2, n_neighbors=min(8,len(docs)-1),
                                min_dist=0.1, metric="cosine", random_state=42
                               ).fit_transform(X_s)
                with _umap_state["lock"]:
                    _umap_state["all_xy"]  = xy
                    _umap_state["all_ids"] = [str(d["_id"]) for d in docs]
            except:
                pass
            return True
    else:
        # 已 fit，只 transform 新資料
        _transform_new(docs)
        return True

# ── 讀資料 ────────────────────────────────────────────────────
def fetch_docs():
    if args.replay:
        return _replay_state["all_docs"][:_replay_state["cursor"]]
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db     = client[DB_NAME]
        docs   = list(db.manifold_points.find(
            {"feature_vec": {"$exists": True}},
            {"feature_vec":1,"action":1,"user_id":1,
             "timestamp":1,"virtual_hour":1,"prev_action":1}
        ).sort("timestamp", 1))
        client.close()
        return docs
    except:
        return []

# ── Dash App ──────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Robot Brain — Manifold Demo"

def _card_style():
    return {
        "background":BG_CARD,"borderRadius":"8px","padding":"14px",
        "marginBottom":"8px","border":f"1px solid {BORDER}",
    }

_mode_badge = "🔴 REPLAY" if args.replay else "🟢 LIVE"
_mode_color = "#EF4444"   if args.replay else "#10B981"

def _btn(bg, label, id_):
    return html.Button(label, id=id_, n_clicks=0, style={
        "background":bg,"color":"white","border":"none",
        "padding":"5px 14px","borderRadius":"5px",
        "cursor":"pointer","fontSize":"13px","fontWeight":"bold",
    })

app.layout = html.Div([
    # 標題列
    html.Div([
        html.Div([
            html.Span("🧠", style={"fontSize":"22px","marginRight":"8px"}),
            html.Span("Robot Brain System",
                      style={"color":TEXT_PRI,"fontWeight":"bold","fontSize":"18px"}),
            html.Span(" — Live Manifold",
                      style={"color":TEXT_SEC,"fontSize":"15px"}),
        ]),
        html.Div([
            html.Span(_mode_badge, style={
                "background":_mode_color,"color":"white","fontSize":"11px",
                "padding":"2px 10px","borderRadius":"4px","marginRight":"12px",
                "fontWeight":"bold",
            }),
            html.Span(id="status-text", style={"color":TEXT_SEC,"fontSize":"13px"}),
        ]),
    ], style={
        "background":BG_CARD,"padding":"10px 20px","display":"flex",
        "justifyContent":"space-between","alignItems":"center",
        "borderBottom":f"1px solid {BORDER}",
    }),

    # Replay 控制
    html.Div([
        _btn("#374151","⏸ 暫停","btn-pause"),
        _btn("#1D4ED8","▶ 繼續","btn-resume"),
        _btn("#7C3AED","⏮ 重置","btn-reset"),
        _btn("#065F46","🔄 重新 fit","btn-refit"),
        html.Span(id="replay-info",
                  style={"color":TEXT_SEC,"fontSize":"13px","marginLeft":"16px"}),
    ], style={
        "display":"flex" if args.replay else "none",
        "background":"#1A2235","padding":"8px 20px",
        "alignItems":"center","gap":"8px",
        "borderBottom":f"1px solid {BORDER}",
    }),

    # 主體
    html.Div([
        # 左：兩張圖
        html.Div([
            dcc.Graph(id="fig-behavior", style={"height":"46vh","marginBottom":"8px"}),
            dcc.Graph(id="fig-timeslot", style={"height":"46vh"}),
        ], style={"width":"62%","paddingRight":"10px"}),

        # 右：面板
        html.Div([
            html.Div(id="score-card",   style=_card_style()),
            html.Div(id="latest-card",  style=_card_style()),
            html.Div(id="stats-card",   style=_card_style()),
            html.Div(id="cluster-card", style={**_card_style(), "marginBottom":"0"}),
        ], style={"width":"38%","overflowY":"auto","maxHeight":"92vh"}),

    ], style={
        "display":"flex","padding":"10px 14px","background":BG_DARK,
        "height":"calc(100vh - 56px)","boxSizing":"border-box",
    }),

    dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),
    dcc.Store(id="store-paused", data=False),

], style={"background":BG_DARK,"fontFamily":"'Segoe UI',Arial,sans-serif","margin":"0"})

# ── Replay 控制 ───────────────────────────────────────────────
@app.callback(
    Output("store-paused","data"),
    Input("btn-pause","n_clicks"),
    Input("btn-resume","n_clicks"),
    Input("btn-reset","n_clicks"),
    Input("btn-refit","n_clicks"),
    State("store-paused","data"),
    prevent_initial_call=True,
)
def handle_controls(pause, resume, reset, refit, paused):
    from dash import ctx
    tid = ctx.triggered_id
    if tid == "btn-pause":
        _replay_state["running"] = False
        return True
    elif tid == "btn-resume":
        _replay_state["running"] = True
        return False
    elif tid == "btn-reset":
        _replay_state["cursor"]  = min(10, len(_replay_state["all_docs"]))
        _replay_state["running"] = True
        # 清空 UMAP 狀態，讓它重新 fit
        with _umap_state["lock"]:
            _umap_state["fitted"]  = False
            _umap_state["all_xy"]  = None
            _umap_state["all_ids"] = []
        return False
    elif tid == "btn-refit":
        # 強制重新 fit（不清資料）
        with _umap_state["lock"]:
            _umap_state["fitted"]  = False
            _umap_state["all_xy"]  = None
            _umap_state["all_ids"] = []
        return paused
    return paused


# ── 主更新 ────────────────────────────────────────────────────
@app.callback(
    Output("fig-behavior",  "figure"),
    Output("fig-timeslot",  "figure"),
    Output("score-card",    "children"),
    Output("latest-card",   "children"),
    Output("stats-card",    "children"),
    Output("cluster-card",  "children"),
    Output("status-text",   "children"),
    Output("replay-info",   "children"),
    Input("interval",       "n_intervals"),
)
def update(_n):
    docs = fetch_docs()
    n    = len(docs)
    now  = datetime.datetime.now().strftime("%H:%M:%S")
    r_total = len(_replay_state["all_docs"])
    r_cur   = _replay_state["cursor"]
    status  = f"n={n}  |  {now}"
    r_info  = f"播放 {r_cur} / {r_total} 筆" if args.replay else ""

    if n < MIN_PTS:
        empty = _empty_fig(f"等待資料（{n}/{MIN_PTS} 筆）...")
        return (empty, empty,
                _score_card(None), _latest_card(None,None,None),
                _stats_card([],[],[]), _cluster_card(None,None,None,None),
                status, r_info)

    # 更新 UMAP（fit or transform）
    update_umap(docs)

    with _umap_state["lock"]:
        all_xy  = _umap_state["all_xy"]
        labels  = _umap_state["labels"]
        ss      = _umap_state["s_score"]
        all_ids = list(_umap_state["all_ids"])

    if all_xy is None:
        empty = _empty_fig("UMAP 計算中...")
        return (empty, empty,
                _score_card(None), _latest_card(None,None,None),
                _stats_card([],[],[]), _cluster_card(None,None,None,None),
                status, r_info)

    # 對齊 docs 和 all_xy（all_ids 可能比 docs 多，取交集）
    id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
    doc_idxs  = [id_to_idx[str(d["_id"])] for d in docs if str(d["_id"]) in id_to_idx]

    if not doc_idxs:
        empty = _empty_fig("資料對齊中...")
        return (empty, empty,
                _score_card(None), _latest_card(None,None,None),
                _stats_card([],[],[]), _cluster_card(None,None,None,None),
                status, r_info)

    emb       = all_xy[doc_idxs]
    doc_labels = labels[doc_idxs] if labels is not None and len(labels) == len(all_ids) else None

    actions   = [norm_action(docs[i].get("action","unknown"))      for i in range(len(docs)) if str(docs[i]["_id"]) in id_to_idx]
    prevs     = [norm_action(docs[i].get("prev_action","unknown")) for i in range(len(docs)) if str(docs[i]["_id"]) in id_to_idx]
    timeslots = [to_slot(docs[i].get("virtual_hour"), docs[i].get("timestamp")) for i in range(len(docs)) if str(docs[i]["_id"]) in id_to_idx]
    users     = [docs[i].get("user_id","unknown") for i in range(len(docs)) if str(docs[i]["_id"]) in id_to_idx]

    # 軌跡：最近 TRAIL_LEN 個點
    trail_x = emb[-TRAIL_LEN:, 0].tolist()
    trail_y = emb[-TRAIL_LEN:, 1].tolist()

    fitted_flag = _umap_state["fitted"]
    n_fit       = _umap_state["n_fit"]

    return (
        _make_fig_behavior(emb, actions, users, doc_labels, trail_x, trail_y, n, fitted_flag, n_fit),
        _make_fig_timeslot(emb, timeslots, trail_x, trail_y, n),
        _score_card(ss),
        _latest_card(docs[-1], actions[-1], timeslots[-1], prevs[-1]),
        _stats_card(docs, actions, timeslots),
        _cluster_card(emb, doc_labels, actions, users),
        status, r_info,
    )


# ── 圖1：行為 ─────────────────────────────────────────────────
def _make_fig_behavior(emb, actions, users, labels, trail_x, trail_y, n, fitted, n_fit):
    fig = go.Figure()

    # cluster 背景
    if labels is not None:
        for cid in sorted(set(labels)):
            if cid == -1: continue
            mask = labels == cid
            pts  = emb[mask]
            if len(pts) < 3: continue
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            r = max(pts[:,0].std(), pts[:,1].std()) * 1.8 + 0.6
            theta = np.linspace(0, 2*np.pi, 60)
            fig.add_trace(go.Scatter(
                x=cx + r*np.cos(theta), y=cy + r*np.sin(theta),
                mode="lines", fill="toself",
                fillcolor="rgba(59,130,246,0.05)",
                line=dict(color="rgba(59,130,246,0.2)", width=1),
                hoverinfo="skip", showlegend=False,
            ))

    # 所有歷史點（半透明）
    for beh, color in BEHAVIOR_COLORS.items():
        for user, symbol in [("User_Mom","circle"),("User_Dad","square"),("unknown","diamond")]:
            mask = [i for i,(a,u) in enumerate(zip(actions,users)) if a==beh and u==user]
            if not mask: continue
            # 分舊點和新點（最後 TRAIL_LEN 個是新點）
            old_mask = [i for i in mask if i < len(actions) - TRAIL_LEN]
            new_mask = [i for i in mask if i >= len(actions) - TRAIL_LEN]

            if old_mask:
                fig.add_trace(go.Scatter(
                    x=[emb[i,0] for i in old_mask],
                    y=[emb[i,1] for i in old_mask],
                    mode="markers",
                    name=f"{beh}/{user.replace('User_','')}",
                    legendgroup=beh,
                    showlegend=(user=="User_Mom" and not new_mask),
                    marker=dict(color=color, size=8, symbol=symbol,
                                opacity=0.45,
                                line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
                    hovertemplate=f"<b>{beh}</b> | {user}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
                ))

            if new_mask:
                fig.add_trace(go.Scatter(
                    x=[emb[i,0] for i in new_mask],
                    y=[emb[i,1] for i in new_mask],
                    mode="markers",
                    name=f"{beh}/{user.replace('User_','')}",
                    legendgroup=beh,
                    showlegend=(user=="User_Mom"),
                    marker=dict(color=color, size=11, symbol=symbol,
                                opacity=0.95,
                                line=dict(width=1.2, color="white")),
                    hovertemplate=f"<b>{beh}</b> | {user}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
                ))

    # 軌跡線（漸層）
    if len(trail_x) >= 2:
        for i in range(len(trail_x)-1):
            alpha = 0.2 + 0.8 * (i / (len(trail_x)-1))
            width = 1.5 + 2.0 * (i / (len(trail_x)-1))
            fig.add_trace(go.Scatter(
                x=[trail_x[i], trail_x[i+1]],
                y=[trail_y[i], trail_y[i+1]],
                mode="lines",
                line=dict(color=f"rgba(239,68,68,{alpha:.2f})", width=width),
                hoverinfo="skip", showlegend=False,
            ))
        # 箭頭
        fig.add_annotation(
            x=trail_x[-1], y=trail_y[-1],
            ax=trail_x[-2], ay=trail_y[-2],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=4, arrowsize=2.0,
            arrowwidth=3, arrowcolor="#EF4444",
        )

    # 最新點（發光）
    fig.add_trace(go.Scatter(
        x=[emb[-1,0]], y=[emb[-1,1]], mode="markers",
        marker=dict(size=26, color="rgba(239,68,68,0.12)",
                    line=dict(width=2, color="#EF4444")),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[emb[-1,0]], y=[emb[-1,1]], mode="markers",
        marker=dict(size=12, color="#FFFFFF",
                    line=dict(width=3, color="#EF4444")),
        hovertemplate=f"<b>NOW: {actions[-1]}</b><extra></extra>",
        showlegend=False,
    ))

    # fit 狀態標示
    fit_note = f"Layout fixed (fit on {n_fit} pts)" if fitted else f"Layout updating..."
    fig.add_annotation(
        text=fit_note, xref="paper", yref="paper",
        x=0.01, y=0.01, showarrow=False,
        font=dict(size=10, color="#475569"),
    )

    fig.update_layout(**_fig_layout(f"Color by Behavior  (n={n})"))
    return fig


# ── 圖2：時段 ─────────────────────────────────────────────────
def _make_fig_timeslot(emb, timeslots, trail_x, trail_y, n):
    fig = go.Figure()

    for slot, color in SLOT_COLORS.items():
        mask = [i for i,s in enumerate(timeslots) if s==slot]
        if not mask: continue
        fig.add_trace(go.Scatter(
            x=[emb[i,0] for i in mask],
            y=[emb[i,1] for i in mask],
            mode="markers",
            name=f"{slot} (n={len(mask)})",
            marker=dict(color=color, size=10,
                        symbol=SLOT_SYMBOLS.get(slot,"circle"),
                        opacity=0.85,
                        line=dict(width=0.8, color="rgba(255,255,255,0.4)")),
            hovertemplate=f"<b>{slot}</b><br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
        ))

    if len(trail_x) >= 2:
        fig.add_trace(go.Scatter(
            x=trail_x, y=trail_y, mode="lines",
            line=dict(color="rgba(251,191,36,0.6)", width=2, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=trail_x[-1], y=trail_y[-1],
            ax=trail_x[-2], ay=trail_y[-2],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=4, arrowsize=1.8,
            arrowwidth=3, arrowcolor="#FBBF24",
        )

    fig.add_trace(go.Scatter(
        x=[emb[-1,0]], y=[emb[-1,1]], mode="markers",
        marker=dict(size=20, color="rgba(251,191,36,0.12)",
                    line=dict(width=2, color="#FBBF24")),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(**_fig_layout(f"Color by Time Slot  (n={n})"))
    return fig


def _fig_layout(title):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_PRI, size=13), x=0.01),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT_SEC, size=11),
        xaxis=dict(title="UMAP dim 1", gridcolor="#1A2D45",
                   zerolinecolor="#1A2D45", tickfont=dict(size=10)),
        yaxis=dict(title="UMAP dim 2", gridcolor="#1A2D45",
                   zerolinecolor="#1A2D45", tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
                    font=dict(size=10), x=1.01, y=1),
        margin=dict(l=45, r=10, t=35, b=40),
        hovermode="closest",
        uirevision="stable_layout",   # ← 關鍵：保持縮放和視角不被重置
    )

def _empty_fig(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=TEXT_SEC, size=14))
    fig.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    return fig


# ── 面板 ──────────────────────────────────────────────────────
def _score_card(ss):
    if ss is None:
        return [
            html.P("Silhouette Score",
                   style={"color":TEXT_SEC,"fontSize":"12px","margin":"0"}),
            html.P("—", style={"color":TEXT_SEC,"fontSize":"28px",
                               "margin":"4px 0 0 0","fontWeight":"bold"}),
            html.P(f"Layout {'fixed' if _umap_state['fitted'] else 'updating'}",
                   style={"color":"#475569","fontSize":"11px","margin":"4px 0 0 0"}),
        ]
    color  = "#10B981" if ss >= 0.7 else "#F59E0B" if ss >= 0.5 else "#EF4444"
    status = "Excellent ✓" if ss >= 0.7 else "Good ✓" if ss >= 0.5 else "Weak"
    fitted_txt = f"Layout fixed on {_umap_state['n_fit']} pts" if _umap_state["fitted"] else "Layout updating..."
    return [
        html.P("Silhouette Score",
               style={"color":TEXT_SEC,"fontSize":"12px","margin":"0","letterSpacing":"0.5px"}),
        html.P(f"{ss:.4f}",
               style={"color":color,"fontSize":"36px","margin":"4px 0 0 0",
                      "fontWeight":"bold","fontFamily":"monospace"}),
        html.P(status, style={"color":color,"fontSize":"12px","margin":"0"}),
        html.P("門檻 ≥ 0.5", style={"color":TEXT_SEC,"fontSize":"11px","margin":"2px 0 0 0"}),
        html.P(fitted_txt, style={"color":"#475569","fontSize":"11px","margin":"4px 0 0 0"}),
    ]


def _latest_card(doc, action, slot, prev=None):
    if doc is None:
        return [html.P("等待資料...", style={"color":TEXT_SEC,"fontSize":"13px"})]
    ts     = doc.get("timestamp","")
    ts_str = ts.strftime("%H:%M:%S") if hasattr(ts,"strftime") else str(ts)[:19]
    vh     = doc.get("virtual_hour","?")
    uid    = doc.get("user_id","?").replace("User_","")
    prev   = prev or norm_action(doc.get("prev_action","unknown"))
    uid_color  = USER_COLORS.get(doc.get("user_id",""), TEXT_PRI)
    beh_color  = BEHAVIOR_COLORS.get(action, TEXT_PRI)
    prev_color = BEHAVIOR_COLORS.get(prev, "#94A3B8")
    return [
        html.P("🎯 最新觀測",
               style={"color":TEXT_PRI,"fontSize":"13px","fontWeight":"bold","margin":"0 0 8px 0"}),
        _row("用戶",       uid,    uid_color),
        _row("前一行為",   prev,   prev_color),
        _row("→ 當前行為", action, beh_color),
        _row("時段",       slot,   SLOT_COLORS.get(slot, TEXT_PRI)),
        _row("Virtual Hr", str(vh)),
        _row("時間",       ts_str),
    ]


def _stats_card(docs, actions, timeslots):
    n = len(docs)
    children = [
        html.P("📊 統計",
               style={"color":TEXT_PRI,"fontSize":"13px","fontWeight":"bold","margin":"0 0 8px 0"}),
        _row("總筆數", str(n)),
    ]
    if actions:
        ac = Counter(actions).most_common(3)
        children.append(_row("主要行為", "  ".join(f"{a}({c})" for a,c in ac)))
    if timeslots:
        sc = Counter(timeslots)
        children.append(_row("時段", "  ".join(f"{k}:{v}" for k,v in sorted(sc.items()))))
    if docs:
        uc = Counter(d.get("user_id","?") for d in docs)
        children.append(_row("用戶", "  ".join(f"{k.replace('User_','')}:{v}" for k,v in uc.items())))
    return children


def _cluster_card(emb, labels, actions=None, users=None):
    if emb is None or labels is None:
        return [html.P("Cluster 計算中...", style={"color":TEXT_SEC,"fontSize":"13px"})]
    n_cl  = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels==-1).sum())
    result = [
        html.P("🔵 Cluster",
               style={"color":TEXT_PRI,"fontSize":"13px","fontWeight":"bold","margin":"0 0 8px 0"}),
        _row("數量", str(n_cl),  ACCENT),
        _row("Noise", str(noise), "#6B7280"),
    ]
    for cid in sorted(set(labels)):
        if cid == -1: continue
        mask  = [i for i,l in enumerate(labels) if l==cid]
        acts  = [actions[i] for i in mask] if actions else []
        top_a = Counter(acts).most_common(1)
        top_u = ""
        if users:
            uc    = Counter(users[i] for i in mask)
            top_u = uc.most_common(1)[0][0].replace("User_","") if uc else ""
        lstr  = top_a[0][0] if top_a else "?"
        color = BEHAVIOR_COLORS.get(lstr, TEXT_PRI)
        result.append(_row(f"C{cid} (n={len(mask)})", f"{lstr} {top_u}", color))
    return result


def _row(label, value, value_color=None):
    return html.Div([
        html.Span(label, style={"color":TEXT_SEC,"fontSize":"12px",
                                "minWidth":"100px","display":"inline-block"}),
        html.Span(value, style={"color":value_color or TEXT_PRI,
                                "fontSize":"12px","fontWeight":"600"}),
    ], style={"marginBottom":"5px"})


if __name__ == "__main__":
    mode = "REPLAY" if args.replay else "LIVE"
    print(f"\n{'='*50}")
    print(f"  Robot Brain — Manifold Dashboard  [{mode}]")
    print(f"  http://localhost:{args.port}")
    if args.replay:
        print(f"  資料：{len(_replay_state['all_docs'])} 筆，倍速 x{args.speed}")
    print(f"  Layout 鎖定於 {MIN_FIT_PTS} 筆後（新點 transform 投影）")
    print(f"{'='*50}\n")
    app.run(debug=False, port=args.port)