import math
import os
from collections import Counter

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _load_config(path: str) -> dict:
    if _YAML_OK and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    return {}


_sys_cfg = _load_config("config/system_config.yaml")
_entropy_cfg = _sys_cfg.get("entropy", {})

ENTROPY_HIGH_THRESHOLD  = float(_entropy_cfg.get("high_threshold",   1.2))
ENTROPY_LOW_THRESHOLD   = float(_entropy_cfg.get("low_threshold",    0.4))
VLM_WEIGHT_HIGH_ENTROPY = float(_entropy_cfg.get("vlm_weight_high",  0.10))
VLM_WEIGHT_LOW_ENTROPY  = float(_entropy_cfg.get("vlm_weight_low",   0.30))
VLM_WEIGHT_DEFAULT      = float(_entropy_cfg.get("vlm_weight_default", 0.20))


class EntropyMonitor:

    def __init__(self):
        self._last_entropy: dict = {}

    def compute_entropy(self, votes: list) -> float:
        if not votes:
            return float("inf")
        counts  = Counter(votes)
        total   = len(votes)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def get_dynamic_vlm_weight(self, entropy: float) -> float:
        """
        Returns W_VLM adjusted by VLM voting consistency.

        With 4 images (topN=2 x burst=2):
          entropy = 0.0  → 4:0 unanimous      → trust VLM more
          entropy = 0.4  → 3:1 consistent      → default weight
          entropy = 1.0  → 2:2 split           → reduce VLM weight
          entropy = 1.2  → 2:1:1 three-way     → low VLM weight
          entropy = 2.0  → 1:1:1:1 random      → minimum VLM weight

        This is more reliable than vlm_confidence (self-reported)
        because it measures external consistency across multiple views.
        """
        if entropy >= ENTROPY_HIGH_THRESHOLD:
            return VLM_WEIGHT_HIGH_ENTROPY
        elif entropy <= ENTROPY_LOW_THRESHOLD:
            return VLM_WEIGHT_LOW_ENTROPY
        else:
            ratio = ((entropy - ENTROPY_LOW_THRESHOLD) /
                     (ENTROPY_HIGH_THRESHOLD - ENTROPY_LOW_THRESHOLD))
            w = (VLM_WEIGHT_LOW_ENTROPY +
                 ratio * (VLM_WEIGHT_HIGH_ENTROPY - VLM_WEIGHT_LOW_ENTROPY))
            return round(w, 3)

    def get_weights(self, entropy: float) -> dict:
        vlm_w = self.get_dynamic_vlm_weight(entropy)
        if entropy >= ENTROPY_HIGH_THRESHOLD:
            mode = "high_entropy"
        elif entropy <= ENTROPY_LOW_THRESHOLD:
            mode = "low_entropy"
        else:
            mode = "interpolated"
        return {"vlm": vlm_w, "mode": mode}

    def analyze(self, user_id: str,
                activity_votes: list,
                body_votes: list,
                held_votes: list) -> dict:
        act_entropy  = self.compute_entropy(activity_votes)
        body_entropy = self.compute_entropy(body_votes)
        held_entropy = self.compute_entropy(held_votes)

        overall = (act_entropy  * 0.6 +
                   body_entropy * 0.2 +
                   held_entropy * 0.2)

        self._last_entropy[user_id] = overall
        weights    = self.get_weights(overall)
        vlm_weight = self.get_dynamic_vlm_weight(overall)

        print(f"[Entropy] {user_id} | "
              f"act={act_entropy:.2f} body={body_entropy:.2f} "
              f"held={held_entropy:.2f} overall={overall:.2f} "
              f"→ W_VLM={vlm_weight:.2f} mode={weights['mode']}")

        return {
            "activity_entropy":   act_entropy,
            "body_entropy":       body_entropy,
            "held_entropy":       held_entropy,
            "overall_entropy":    overall,
            "dynamic_vlm_weight": vlm_weight,
            "weights":            weights,
            "high_entropy":       overall >= ENTROPY_HIGH_THRESHOLD,
        }

    def get_last_entropy(self, user_id: str) -> float:
        return self._last_entropy.get(user_id, 0.0)