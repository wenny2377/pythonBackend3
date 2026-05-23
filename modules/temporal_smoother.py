import datetime
from collections import deque


TRANSITION_PRIOR = {
    ("Laying",    "Cooking"):    0.02,
    ("Laying",    "Cleaning"):   0.02,
    ("Laying",    "Typing"):     0.05,
    ("Laying",    "Opening"):    0.03,
    ("Watching",  "Cooking"):    0.10,
    ("Watching",  "Cleaning"):   0.05,
    ("Eating",    "Typing"):     0.05,
    ("Cooking",   "Laying"):     0.03,
    ("Typing",    "Cooking"):    0.05,
    ("Cleaning",  "Laying"):     0.03,
    ("SittingDrink", "Cooking"): 0.08,
}

HIGH_INERTIA = {
    "Laying", "Watching", "Typing", "Reading", "Cleaning"
}

WINDOW_SIZE      = 3
CONFIDENCE_FLOOR = 0.40


class TemporalSmoother:

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        self._history: dict[str, deque] = {}
        self._last_confirmed: dict[str, str] = {}
        self._last_time: dict[str, datetime.datetime] = {}

    def smooth(self, user_id: str, new_action: str,
               confidence: float = 1.0) -> tuple[str, str]:
        if user_id not in self._history:
            self._history[user_id]       = deque(maxlen=self._window_size)
            self._last_confirmed[user_id] = new_action
            self._last_time[user_id]      = datetime.datetime.utcnow()

        prev = self._last_confirmed.get(user_id, new_action)
        history = self._history[user_id]

        now = datetime.datetime.utcnow()
        elapsed = (now - self._last_time.get(user_id, now)).total_seconds()
        self._last_time[user_id] = now

        if elapsed > 120:
            self._history[user_id].clear()
            self._last_confirmed[user_id] = new_action
            history.append(new_action)
            return new_action, "time_reset"

        transition_prob = TRANSITION_PRIOR.get((prev, new_action), 1.0)

        if transition_prob < 0.10 and confidence < CONFIDENCE_FLOOR:
            print(f"[Smoother] Suspicious: {prev}→{new_action} "
                  f"(trans={transition_prob:.2f} conf={confidence:.2f}) "
                  f"→ keeping {prev}")
            history.append(prev)
            return prev, f"smoothed:{prev}"

        if prev in HIGH_INERTIA and new_action != prev:
            recent = list(history)
            if len(recent) >= 2 and recent[-1] == prev and recent[-2] == prev:
                if transition_prob < 0.20 and confidence < 0.60:
                    print(f"[Smoother] High inertia: {prev}→{new_action} "
                          f"→ keeping {prev}")
                    history.append(prev)
                    return prev, f"inertia:{prev}"

        history.append(new_action)
        self._last_confirmed[user_id] = new_action
        return new_action, ""

    def reset(self, user_id: str):
        self._history.pop(user_id, None)
        self._last_confirmed.pop(user_id, None)
        self._last_time.pop(user_id, None)