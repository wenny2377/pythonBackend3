import base64
import math

import cv2
import numpy as np


def extract_label_from_held_event(held_event: str) -> str:
    if not held_event or held_event in ("none", "unknown", ""):
        return ""
    for kw in ("just picked up ", "holding ", "holding:"):
        if kw in held_event.lower():
            after = held_event.lower().split(kw, 1)[1]
            return after.split(" for")[0].strip()
    return ""


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


BOX_HALF_SIZE   = 28
BOX_COLOR_HELD  = (0, 215, 255)
BOX_COLOR_NEAR  = (60, 220, 60)
LABEL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE     = 0.5
LABEL_THICKNESS = 1


def _draw_marker_box(img: np.ndarray, u: int, v: int, tag: str,
                      label: str, held: bool) -> None:
    h, w = img.shape[:2]
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))
    color = BOX_COLOR_HELD if held else BOX_COLOR_NEAR

    x1, y1 = max(0, u - BOX_HALF_SIZE), max(0, v - BOX_HALF_SIZE)
    x2, y2 = min(w - 1, u + BOX_HALF_SIZE), min(h - 1, v + BOX_HALF_SIZE)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"{tag} {label}"
    (tw, th), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    ty1 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, ty1), (min(w - 1, x1 + tw + 6), y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4),
                LABEL_FONT, LABEL_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)


def _draw_text_panel(img: np.ndarray, unplaced: list) -> np.ndarray:
    if not unplaced:
        return img
    h, w = img.shape[:2]
    panel_rows = math.ceil(len(unplaced) / 2)
    panel_h    = max(20, panel_rows * 18 + 6)
    panel      = np.zeros((panel_h, w, 3), dtype=np.uint8)

    col_w = w // 2
    for i, (tag, label, status) in enumerate(unplaced, start=1):
        text = f"{tag} {label}" + (f" ({status})" if status else "")
        col, row = (i - 1) % 2, (i - 1) // 2
        px, py = col * col_w + 4, row * 18 + 14
        cv2.putText(panel, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([img, panel])


def mark_objects_on_image(img_b64: str, object_list: list,
                           objects_2d: list = None,
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

    coord_by_label = {}
    if objects_2d:
        for obj in objects_2d:
            label = obj.get("label", "").strip()
            if not label:
                continue
            coord_by_label[label] = (
                obj.get("u"), obj.get("v"), bool(obj.get("held", False)))

    unplaced = []
    for i, obj in enumerate(object_list, start=1):
        label  = obj.get("label", "unknown")
        status = obj.get("status", "")
        tag    = f"[M{i}]"

        coord = coord_by_label.get(label)
        if coord and coord[0] is not None and coord[1] is not None:
            u, v, held = coord
            try:
                _draw_marker_box(img, int(u), int(v), tag, label, held)
                continue
            except Exception as e:
                print(f"[SoM] draw failed for {label}: {e}")

        unplaced.append((tag, label, status))

    combined = _draw_text_panel(img, unplaced)

    _, buf = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def get_som_objects_from_db(db, user_pos: dict, room_name: str,
                             user_id: str, held_event: str,
                             radius: float = 3.0) -> list:
    objects = []

    if held_event and held_event not in ("none", "unknown", ""):
        label = extract_label_from_held_event(held_event)
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
