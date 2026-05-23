import math
from collections import Counter


ENTROPY_HIGH_THRESHOLD = 1.2
ENTROPY_LOW_THRESHOLD  = 0.4

VLM_WEIGHT_HIGH_ENTROPY  = 0.20
VLM_WEIGHT_LOW_ENTROPY   = 0.80
ZONE_WEIGHT_HIGH_ENTROPY = 0.60
ZONE_WEIGHT_LOW_ENTROPY  = 0.20


class EntropyMonitor:

    def __init__(self):
        self._last_entropy: dict[str, float] = {}

    def compute_entropy(self, votes: list[str]) -> float:
        if not votes:
            return float('inf')
        counts = Counter(votes)
        total  = len(votes)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def get_weights(self, entropy: float) -> dict:
        if entropy >= ENTROPY_HIGH_THRESHOLD:
            return {
                "vlm":  VLM_WEIGHT_HIGH_ENTROPY,
                "zone": ZONE_WEIGHT_HIGH_ENTROPY,
                "mode": "high_entropy_fallback",
            }
        elif entropy <= ENTROPY_LOW_THRESHOLD:
            return {
                "vlm":  VLM_WEIGHT_LOW_ENTROPY,
                "zone": ZONE_WEIGHT_LOW_ENTROPY,
                "mode": "low_entropy_confident",
            }
        else:
            ratio = (entropy - ENTROPY_LOW_THRESHOLD) / \
                    (ENTROPY_HIGH_THRESHOLD - ENTROPY_LOW_THRESHOLD)
            vlm_w  = VLM_WEIGHT_LOW_ENTROPY  + ratio * (VLM_WEIGHT_HIGH_ENTROPY  - VLM_WEIGHT_LOW_ENTROPY)
            zone_w = ZONE_WEIGHT_LOW_ENTROPY + ratio * (ZONE_WEIGHT_HIGH_ENTROPY - ZONE_WEIGHT_LOW_ENTROPY)
            return {
                "vlm":  round(vlm_w,  3),
                "zone": round(zone_w, 3),
                "mode": "interpolated",
            }

    def analyze(self, user_id: str,
                activity_votes: list[str],
                body_votes: list[str],
                held_votes: list[str]) -> dict:
        act_entropy  = self.compute_entropy(activity_votes)
        body_entropy = self.compute_entropy(body_votes)
        held_entropy = self.compute_entropy(held_votes)

        overall = (act_entropy * 0.6 +
                   body_entropy * 0.2 +
                   held_entropy * 0.2)

        self._last_entropy[user_id] = overall
        weights = self.get_weights(overall)

        print(f"[Entropy] {user_id} | "
              f"act={act_entropy:.2f} body={body_entropy:.2f} "
              f"held={held_entropy:.2f} overall={overall:.2f} "
              f"mode={weights['mode']}")

        return {
            "activity_entropy": act_entropy,
            "body_entropy":     body_entropy,
            "held_entropy":     held_entropy,
            "overall_entropy":  overall,
            "weights":          weights,
            "high_entropy":     overall >= ENTROPY_HIGH_THRESHOLD,
        }

    def get_last_entropy(self, user_id: str) -> float:
        return self._last_entropy.get(user_id, 0.0)