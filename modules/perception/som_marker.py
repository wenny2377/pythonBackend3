import base64
import math

import cv2
import numpy as np


def build_som_text(object_list: list) -> str:
    if not object_list:
        return ""
    lines = []
    for i, obj in enumerate(object_list, start=1):
        label  = obj.get("label", "unknown")
        status = obj.get("status", "")
        tag    = f"[M{i}]"
        if status:
            lines.append(f"{tag} {label} ({status})")
        else:
            lines.append(f"{tag} {label}")
    return "\n".join(lines)


def mark_objects_on_image(img_b64: str, object_list: list,
                           font_scale: float = 0.5,
                           thickness: int = 1) -> str:
    if not object_list:
        return img_b64

    try:
        raw   = img_b64.split(',')[1] if ',' in img_b64 else img_b64
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return img_b64
    except Exception as e:
        print(f"[SoM] image decode failed: {e}")
        return img_b64

    h, w = img.shape[:2]
    n    = len(object_list)

    panel_h    = 20
    panel_rows = math.ceil(n / 2)
    panel_h    = max(panel_h, panel_rows * 18 + 6)
    panel      = np.zeros((panel_h, w, 3), dtype=np.uint8)

    col_w = w // 2
    for i, obj in enumerate(object_list, start=1):
        label  = obj.get("label", "unknown")
        status = obj.get("status", "")
        text   = f"[M{i}] {label}"
        if status:
            text += f" ({status})"

        col   = i % 2
        row   = (i - 1) // 2
        px    = col * col_w + 4
        py    = row * 18 + 14

        cv2.putText(panel, text, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (200, 200, 200), thickness, cv2.LINE_AA)

    combined = np.vstack([img, panel])

    _, buf = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def get_som_objects_from_db(db, user_pos: dict, room_name: str,
                             user_id: str, held_event: str,
                             radius: float = 3.0) -> list:
    objects = []

    if held_event and held_event not in ("none", "unknown", ""):
        label = _extract_label_from_held_event(held_event)
        if label:
            objects.append({"label": label, "status": "held"})

    if user_pos:
        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))

        try:
            query = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
            docs  = list(db.dynamic_objects.find(
                query, {"label": 1, "sensor_pos": 1, "furniture_pos": 1, "held_by": 1}
            ))
            for doc in docs:
                label = doc.get("label", "").strip()
                if not label:
                    continue

                if doc.get("held_by") == user_id:
                    if not any(o["label"] == label for o in objects):
                        objects.append({"label": label, "status": "held"})
                    continue

                pos = doc.get("sensor_pos") or doc.get("furniture_pos")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                dist = math.sqrt((ux - pos[0]) ** 2 + (uz - pos[1]) ** 2)
                if dist <= radius:
                    objects.append({"label": label, "status": "nearby"})
        except Exception as e:
            print(f"[SoM] DB query failed: {e}")

    seen  = set()
    dedup = []
    for obj in objects:
        key = obj["label"]
        if key not in seen:
            seen.add(key)
            dedup.append(obj)

    return dedup[:8]


def _extract_label_from_held_event(held_event: str) -> str:
    if not held_event or held_event in ("none", "unknown", ""):
        return ""
    for kw in ("just picked up ", "holding ", "holding:"):
        if kw in held_event.lower():
            after = held_event.lower().split(kw, 1)[1]
            return after.split(" for")[0].strip()
    return ""