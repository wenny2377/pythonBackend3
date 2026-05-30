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
    print("[Manifold] PyTorch not installed — training disabled")

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
]
N_BEHAVIORS   = len(BEHAVIOR_LABELS)
N_FEATURES    = 2 + 2 + N_BEHAVIORS

MIN_CONFIDENCE   = 0.60
MIN_TRAIN_SAMPLE = 20
AUGMENT_FACTOR   = 100
TIME_NOISE_STD   = 0.5 / 24
POS_NOISE_STD    = 0.05
RETRAIN_EVERY    = 20
MODEL_DIR        = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "manifold_models"
)

NO_RECORD_ACTIONS = {"Unknown", "Standing", "Walking", "StandUp",
                      "PickingUp", "PuttingDown"}


def _time_encode(hour: float):
    rad = 2 * math.pi * float(hour) / 24.0
    return math.sin(rad), math.cos(rad)


def _prev_onehot(action: str):
    vec = np.zeros(N_BEHAVIORS, dtype=np.float32)
    if action in BEHAVIOR_LABELS:
        vec[BEHAVIOR_LABELS.index(action)] = 1.0
    return vec


def build_x(virtual_hour, user_pos, prev_action):
    h = float(virtual_hour) if virtual_hour is not None else 12.0
    sin_t, cos_t = _time_encode(h)
    time_vec = np.array([sin_t, cos_t], dtype=np.float32)

    x = float(user_pos.get("x", 0)) / 10.0 if user_pos else 0.0
    z = float(user_pos.get("z", 0)) / 10.0 if user_pos else 0.0
    pos_vec = np.array([x, z], dtype=np.float32)

    prev_vec = _prev_onehot(prev_action or "Standing")
    return np.concatenate([time_vec, pos_vec, prev_vec])


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


def _augment(X: np.ndarray, y: np.ndarray, factor: int = AUGMENT_FACTOR):
    rng    = np.random.default_rng(42)
    n      = len(X)
    n_aug  = n * factor
    idx    = rng.integers(0, n, size=n_aug)
    X_aug  = X[idx].copy()
    y_aug  = y[idx].copy()

    X_aug[:, 0] += rng.normal(0, TIME_NOISE_STD, n_aug).astype(np.float32)
    X_aug[:, 1] += rng.normal(0, TIME_NOISE_STD, n_aug).astype(np.float32)
    X_aug[:, 2] += rng.normal(0, POS_NOISE_STD, n_aug).astype(np.float32)
    X_aug[:, 3] += rng.normal(0, POS_NOISE_STD, n_aug).astype(np.float32)

    norms = np.sqrt(X_aug[:, 0]**2 + X_aug[:, 1]**2 + 1e-8)
    X_aug[:, 0] /= norms
    X_aug[:, 1] /= norms

    X_full = np.vstack([X, X_aug])
    y_full = np.concatenate([y, y_aug])
    return X_full, y_full


class ManifoldEngine:

    def __init__(self, db, sbert_model=None):
        self.db          = db
        self.sbert       = sbert_model
        self._lock       = threading.Lock()
        self._models     = {}
        self._sample_cnt = {}
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_all_models()
        print(f"[ManifoldEngine] Ready ({N_FEATURES}-dim MLP, "
              f"{N_BEHAVIORS} behaviors, per-user isolated)")

    def record_training_sample(self, user_id: str,
                                virtual_hour, user_pos: dict,
                                prev_action: str, current_action: str):
        if current_action not in BEHAVIOR_LABELS:
            return
        if current_action in NO_RECORD_ACTIONS:
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
            print(f"[Manifold] sample #{cnt} | "
                  f"{user_id} {prev_action}→{current_action}")

            if cnt >= MIN_TRAIN_SAMPLE and cnt % RETRAIN_EVERY == 0:
                threading.Thread(
                    target=self.train_model,
                    args=(user_id,), daemon=True).start()
        except Exception as e:
            print(f"[Manifold] record_training_sample error: {e}")

    def train_model(self, user_id: str):
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

        if X_raw.shape[1] != N_FEATURES:
            print(f"[Manifold] Feature dim mismatch: "
                  f"data={X_raw.shape[1]} expected={N_FEATURES}, skip")
            return

        X_aug, y_aug = _augment(X_raw, y_raw)
        print(f"[Manifold] {user_id}: {len(X_raw)} raw → "
              f"{len(X_aug)} augmented, training...")

        dataset   = TensorDataset(
            torch.tensor(X_aug), torch.tensor(y_aug))
        loader    = DataLoader(dataset, batch_size=64, shuffle=True)
        model     = _MLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),
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
                print(f"[Manifold] {user_id} epoch {epoch+1} "
                      f"loss={total_loss/len(loader):.4f}")

        model.eval()
        with self._lock:
            self._models[user_id] = model

        path = os.path.join(MODEL_DIR, f"{user_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[Manifold] {user_id} model saved → {path}")

    def predict_intent(self, user_id: str,
                       virtual_hour=None, user_pos=None,
                       prev_action: str = "Standing"):
        empty = {
            "trigger":    False,
            "intent":     "unknown",
            "confidence": 0.0,
            "probs":      {b: 0.0 for b in BEHAVIOR_LABELS},
        }
        model = self._get_model(user_id)
        if model is None:
            return empty

        X = build_x(virtual_hour, user_pos, prev_action)
        if len(X) != N_FEATURES:
            print(f"[Manifold] predict: feature dim mismatch {len(X)} vs {N_FEATURES}")
            return empty

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
            print(f"[Manifold] intent={intent} conf={best_p:.2f} trigger=True")

        return {
            "trigger":    trigger,
            "intent":     intent,
            "confidence": round(best_p, 3),
            "probs":      prob_dict,
        }

    def probe_spatiotemporal(self, user_id: str,
                              pos: list, prev_action: str = "Standing",
                              n_hours: int = 48):
        model  = self._get_model(user_id)
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
            matrix = probs.T
        except Exception as e:
            print(f"[Manifold] probe_spatiotemporal error: {e}")

        return hours, matrix

    def probe_zone_behavior(self, user_id: str,
                             zone_centers: dict,
                             virtual_hour: float = 20.0,
                             prev_action: str = "Standing"):
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
            matrix = probs
        except Exception as e:
            print(f"[Manifold] probe_zone_behavior error: {e}")

        return zone_names, matrix

    def update_service_result(self, user_id: str,
                               action: str, result: str):
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

    def _get_model(self, user_id: str):
        with self._lock:
            return self._models.get(user_id)

    def _load_all_models(self):
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
                if hasattr(model, 'net'):
                    first_layer = list(model.net.children())[0]
                    if hasattr(first_layer, 'in_features'):
                        if first_layer.in_features != N_FEATURES:
                            print(f"[Manifold] {uid}: model dim mismatch "
                                  f"({first_layer.in_features} vs {N_FEATURES}), skip")
                            continue
                self._models[uid] = model
                print(f"[Manifold] Loaded model for {uid}")
            except Exception as e:
                print(f"[Manifold] Failed to load {path}: {e}")