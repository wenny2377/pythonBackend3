"""
manifold_engine.py  —  L4 Intent Manifold Engine (redesigned)

Architecture:
  Input  : 19-dim context vector [time(2), pos(2), prev_action_onehot(15)]
  Model  : Per-user isolated MLP  19 → 64 → 32 → 15 (softmax)
  Output : 15-class behavior probability distribution
  Trigger: max_prob >= MIN_CONFIDENCE → Service Proposal

Data pipeline:
  1. record_training_sample() — store raw (X, y) to manifold_training_data
  2. train_model(user_id)     — augment + train MLP (call after Exp3)
  3. predict_intent()         — runtime inference (19-dim → 15 probs)
  4. probe_spatiotemporal()   — virtual probe for heatmap generation
  5. probe_zone_behavior()    — virtual probe for topology comparison
"""

import os
import math
import pickle
import datetime
import threading
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("⚠️  [Manifold] PyTorch not installed — training disabled")

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "PickingUp", "PuttingDown", "Standing", "Walking",
]
N_BEHAVIORS   = len(BEHAVIOR_LABELS)   # 15
N_FEATURES    = 2 + 2 + N_BEHAVIORS   # 19

MIN_CONFIDENCE   = 0.60
MIN_TRAIN_SAMPLE = 20       # minimum raw samples before training
AUGMENT_FACTOR   = 100      # Monte Carlo over-sampling multiplier
TIME_NOISE_STD   = 0.5 / 24 # ± 30 min in normalised [0,1] scale
POS_NOISE_STD    = 0.05     # ± ~0.5m (pos is divided by 10)
RETRAIN_EVERY    = 20       # retrain after every N new samples
MODEL_DIR        = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "manifold_models"
)  # absolute path relative to this file


# ── helpers ──────────────────────────────────────────────────────────

def _time_encode(hour: float):
    """Convert 0-24 hour to (sin, cos) pair."""
    rad = 2 * math.pi * float(hour) / 24.0
    return math.sin(rad), math.cos(rad)


def _prev_onehot(action: str):
    vec = np.zeros(N_BEHAVIORS, dtype=np.float32)
    if action in BEHAVIOR_LABELS:
        vec[BEHAVIOR_LABELS.index(action)] = 1.0
    return vec


def build_x(virtual_hour, user_pos, prev_action):
    """
    Build 19-dim feature vector.
    virtual_hour : float (0-24)
    user_pos     : dict {"x": ..., "z": ...} or None
    prev_action  : str  behaviour label or ""
    """
    h = float(virtual_hour) if virtual_hour is not None else 12.0
    sin_t, cos_t = _time_encode(h)
    time_vec = np.array([sin_t, cos_t], dtype=np.float32)

    x = float(user_pos.get("x", 0)) / 10.0 if user_pos else 0.0
    z = float(user_pos.get("z", 0)) / 10.0 if user_pos else 0.0
    pos_vec = np.array([x, z], dtype=np.float32)

    prev_vec = _prev_onehot(prev_action or "Standing")
    return np.concatenate([time_vec, pos_vec, prev_vec])


# ── MLP definition ───────────────────────────────────────────────────

class _MLP(nn.Module if _TORCH_OK else object):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, N_BEHAVIORS),
        )

    def forward(self, x):
        return self.net(x)


# ── augmentation ─────────────────────────────────────────────────────

def _augment(X: np.ndarray, y: np.ndarray, factor: int = AUGMENT_FACTOR):
    """
    Monte Carlo Gaussian oversampling.
    Adds isotropic noise to time_vec (dim 0-1) and pos_vec (dim 2-3).
    One-hot prev_action (dim 4-18) is kept intact.
    """
    rng    = np.random.default_rng(42)
    n      = len(X)
    n_aug  = n * factor
    idx    = rng.integers(0, n, size=n_aug)
    X_aug  = X[idx].copy()
    y_aug  = y[idx].copy()

    # time noise
    X_aug[:, 0] += rng.normal(0, TIME_NOISE_STD, n_aug).astype(np.float32)
    X_aug[:, 1] += rng.normal(0, TIME_NOISE_STD, n_aug).astype(np.float32)
    # pos noise
    X_aug[:, 2] += rng.normal(0, POS_NOISE_STD, n_aug).astype(np.float32)
    X_aug[:, 3] += rng.normal(0, POS_NOISE_STD, n_aug).astype(np.float32)

    # re-normalise time dims to unit circle (approximate)
    norms = np.sqrt(X_aug[:, 0]**2 + X_aug[:, 1]**2 + 1e-8)
    X_aug[:, 0] /= norms
    X_aug[:, 1] /= norms

    X_full = np.vstack([X, X_aug])
    y_full = np.concatenate([y, y_aug])
    return X_full, y_full


# ── main engine ──────────────────────────────────────────────────────

class ManifoldEngine:

    def __init__(self, db, sbert_model=None):
        self.db          = db
        self.sbert       = sbert_model   # kept for API compatibility, not used
        self._lock       = threading.Lock()
        self._models     = {}            # {user_id: _MLP}
        self._sample_cnt = {}            # {user_id: int}
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_all_models()
        print("✅ [ManifoldEngine] Ready (19-dim MLP, per-user isolated)")

    # ── public API ───────────────────────────────────────────────────

    def record_training_sample(self, user_id: str,
                                virtual_hour, user_pos: dict,
                                prev_action: str, current_action: str):
        """
        Store one (X, y) training sample.
        Only called when current_action is a valid non-Unknown action.
        """
        if current_action not in BEHAVIOR_LABELS:
            return
        if current_action in ("Unknown", "Standing", "Walking"):
            return

        X = build_x(virtual_hour, user_pos, prev_action).tolist()
        y = BEHAVIOR_LABELS.index(current_action)
        try:
            self.db.manifold_training_data.insert_one({
                "user_id":      user_id,
                "X":            X,
                "y":            y,
                "action":       current_action,
                "prev_action":  prev_action or "",
                "virtual_hour": float(virtual_hour) if virtual_hour else 12.0,
                "timestamp":    datetime.datetime.utcnow(),
            })
            cnt = self._sample_cnt.get(user_id, 0) + 1
            self._sample_cnt[user_id] = cnt
            print(f"   📝 [Manifold] sample #{cnt} | "
                  f"{user_id} {prev_action}→{current_action}")

            if cnt >= MIN_TRAIN_SAMPLE and cnt % RETRAIN_EVERY == 0:
                t = threading.Thread(
                    target=self.train_model,
                    args=(user_id,), daemon=True)
                t.start()
        except Exception as e:
            print(f"[Manifold] record_training_sample error: {e}")

    def train_model(self, user_id: str):
        """Train (or retrain) the per-user MLP. Thread-safe."""
        if not _TORCH_OK:
            print("[Manifold] PyTorch not available, skipping training")
            return

        docs = list(self.db.manifold_training_data.find(
            {"user_id": user_id}, {"X": 1, "y": 1}))
        if len(docs) < MIN_TRAIN_SAMPLE:
            print(f"[Manifold] {user_id}: only {len(docs)} samples, "
                  f"need {MIN_TRAIN_SAMPLE}")
            return

        X_raw = np.array([d["X"] for d in docs], dtype=np.float32)
        y_raw = np.array([d["y"] for d in docs], dtype=np.int64)

        X_aug, y_aug = _augment(X_raw, y_raw)
        print(f"[Manifold] {user_id}: {len(X_raw)} raw → "
              f"{len(X_aug)} augmented samples, training...")

        dataset    = TensorDataset(
            torch.tensor(X_aug), torch.tensor(y_aug))
        loader     = DataLoader(dataset, batch_size=64, shuffle=True)
        model      = _MLP()
        criterion  = nn.CrossEntropyLoss()
        optimizer  = optim.Adam(model.parameters(),
                                lr=1e-3, weight_decay=1e-4)

        model.train()
        for epoch in range(60):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"   [Manifold] {user_id} epoch {epoch+1} "
                      f"loss={total_loss/len(loader):.4f}")

        model.eval()
        with self._lock:
            self._models[user_id] = model

        path = os.path.join(MODEL_DIR, f"{user_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ [Manifold] {user_id} model saved → {path}")

    def predict_intent(self, user_id: str,
                       virtual_hour=None, user_pos=None,
                       prev_action: str = "Standing"):
        """
        Runtime intent prediction.
        Returns dict with trigger, intent, confidence, probs.
        """
        empty = {
            "trigger": False, "intent": "unknown",
            "confidence": 0.0,
            "probs": {b: 0.0 for b in BEHAVIOR_LABELS},
        }
        model = self._get_model(user_id)
        if model is None:
            return empty

        X   = build_x(virtual_hour, user_pos, prev_action)
        try:
            with torch.no_grad():
                logits = model(torch.tensor(X).unsqueeze(0))
                probs  = torch.softmax(logits, dim=1).numpy()[0]
        except Exception as e:
            print(f"[Manifold] predict error: {e}")
            return empty

        best_i = int(np.argmax(probs))
        best_p = float(probs[best_i])
        intent = BEHAVIOR_LABELS[best_i]

        prob_dict = {b: round(float(p), 4)
                     for b, p in zip(BEHAVIOR_LABELS, probs)}

        trigger = best_p >= MIN_CONFIDENCE
        if trigger:
            print(f"   🧭 [Manifold] intent={intent} "
                  f"conf={best_p:.2f} trigger=True")

        return {
            "trigger":    trigger,
            "intent":     intent,
            "confidence": round(best_p, 3),
            "probs":      prob_dict,
        }

    def probe_spatiotemporal(self, user_id: str,
                              pos: list, prev_action: str = "Standing",
                              n_hours: int = 48):
        """
        Virtual probe: sweep 24h at fixed pos and prev_action.
        Returns (hours, matrix) where matrix shape = (N_BEHAVIORS, n_hours).
        Used for Experiment Figure: time-space entropy heatmap.
        """
        model = self._get_model(user_id)
        hours  = np.linspace(0, 24, n_hours, endpoint=False)
        matrix = np.zeros((N_BEHAVIORS, n_hours), dtype=np.float32)

        if model is None:
            return hours, matrix

        user_pos = {"x": pos[0] * 10.0, "z": pos[1] * 10.0}
        try:
            Xs = np.array([
                build_x(h, user_pos, prev_action) for h in hours
            ], dtype=np.float32)
            with torch.no_grad():
                logits = model(torch.tensor(Xs))
                probs  = torch.softmax(logits, dim=1).numpy()
            matrix = probs.T   # shape (N_BEHAVIORS, n_hours)
        except Exception as e:
            print(f"[Manifold] probe_spatiotemporal error: {e}")

        return hours, matrix

    def probe_zone_behavior(self, user_id: str,
                             zone_centers: dict,
                             virtual_hour: float = 20.0,
                             prev_action: str = "Standing"):
        """
        Virtual probe: sweep zone centers at fixed time and prev_action.
        zone_centers: {"Sofa_Zone": [cx, cz], "Desk_Zone": [...], ...}
        Returns matrix shape (n_zones, N_BEHAVIORS).
        Used for Experiment Figure: Mom vs Dad topology comparison.
        """
        model      = self._get_model(user_id)
        zone_names = list(zone_centers.keys())
        matrix     = np.zeros((len(zone_names), N_BEHAVIORS), dtype=np.float32)

        if model is None:
            return zone_names, matrix

        try:
            Xs = []
            for zn in zone_names:
                cx, cz = zone_centers[zn]
                pos    = {"x": cx * 10.0, "z": cz * 10.0}
                Xs.append(build_x(virtual_hour, pos, prev_action))
            Xs = np.array(Xs, dtype=np.float32)
            with torch.no_grad():
                logits = model(torch.tensor(Xs))
                probs  = torch.softmax(logits, dim=1).numpy()
            matrix = probs   # shape (n_zones, N_BEHAVIORS)
        except Exception as e:
            print(f"[Manifold] probe_zone_behavior error: {e}")

        return zone_names, matrix

    def update_service_result(self, user_id: str,
                               action: str, result: str):
        """Log accepted / rejected for future analysis."""
        try:
            self.db.service_results.insert_one({
                "user_id":   user_id,
                "action":    action,
                "result":    result,
                "timestamp": datetime.datetime.utcnow(),
            })
        except Exception as e:
            print(f"[Manifold] update_service_result error: {e}")

    def get_training_count(self, user_id: str) -> int:
        return self.db.manifold_training_data.count_documents(
            {"user_id": user_id})

    # ── internal ─────────────────────────────────────────────────────

    def _get_model(self, user_id: str):
        with self._lock:
            return self._models.get(user_id)

    def _load_all_models(self):
        """Load pre-trained models from disk at startup."""
        if not _TORCH_OK:
            return
        for fname in os.listdir(MODEL_DIR):
            if not fname.endswith(".pkl"):
                continue
            uid  = fname[:-4]
            path = os.path.join(MODEL_DIR, fname)
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                self._models[uid] = model
                print(f"   [Manifold] Loaded model for {uid} from {path}")
            except Exception as e:
                print(f"   [Manifold] Failed to load {path}: {e}")