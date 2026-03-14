import numpy as np
import faiss
import json
import os
import datetime
from sentence_transformers import SentenceTransformer


class VectorMemory:

    def __init__(self, index_path="robot_memory.index",
                       meta_path="robot_memory_meta.json",
                       dynamic_index_path="dynamic_memory.index",
                       dynamic_meta_path="dynamic_memory_meta.json",
                       device="cuda"):

        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
        self.dim = 384

        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"[FAISS] Habit index loaded: {self.index.ntotal} entries")
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            print("[FAISS] Habit index created")

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        self.dynamic_index_path = dynamic_index_path
        self.dynamic_meta_path = dynamic_meta_path

        if os.path.exists(dynamic_index_path):
            self.dynamic_index = faiss.read_index(dynamic_index_path)
            print(f"[FAISS] Dynamic object index loaded: {self.dynamic_index.ntotal} entries")
        else:
            self.dynamic_index = faiss.IndexFlatIP(self.dim)
            print("[FAISS] Dynamic object index created")

        if os.path.exists(dynamic_meta_path):
            with open(dynamic_meta_path, 'r', encoding='utf-8') as f:
                self.dynamic_metadata = json.load(f)
        else:
            self.dynamic_metadata = []

    def sync_from_mongo(self, dynamic_objects_collection):
        docs = list(dynamic_objects_collection.find({}))
        count = 0

        for doc in docs:
            label = doc.get("label", "").lower().strip()
            if not label:
                continue

            self.upsert_dynamic_object(
                label=label,
                room=doc.get("room", ""),
                last_seen_on=doc.get("last_seen_on", "unknown"),
                spatial_rel=doc.get("spatial_rel", "near"),
                furniture_pos=doc.get("furniture_pos") or doc.get("position"),
                seen_count=doc.get("seen_count", 1),
                interact_count=doc.get("interact_count", 0),
                interacted_by=doc.get("interacted_by", []),
            )
            count += 1

        print(f"[FAISS] sync_from_mongo: {count} dynamic objects synced")
        return count

    def add_memory(self, user_id, action, furniture_label, vlm_description,
                   detected_items=None, all_items=None, spatial_relations=None,
                   furniture_pos=None, mongo_id=None):

        detected_items = detected_items or []
        all_items = all_items or []
        spatial_relations = spatial_relations or []

        items_str = ", ".join(detected_items) if detected_items else "nothing"

        spatial_text = " ".join([
            f"{r['subject']} {r['relation']} {r['object']}."
            for r in spatial_relations
            if r.get('subject') and r.get('relation') and r.get('object')
        ])

        memory_text = (
            f"{user_id} {action} near {furniture_label} with {items_str}. "
            f"{vlm_description} {spatial_text}"
        ).strip()

        vec = self.model.encode([memory_text]).astype('float32')
        faiss.normalize_L2(vec)
        self.index.add(vec)

        entry = {
            "faiss_idx": self.index.ntotal - 1,
            "user": user_id,
            "action": action,
            "instance": furniture_label,
            "interacting_items": detected_items,
            "all_items": all_items,
            "spatial_relations": spatial_relations,
            "furniture_pos": furniture_pos,
            "mongo_id": str(mongo_id) if mongo_id else None,
            "description": vlm_description,
            "memory_text": memory_text,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.metadata.append(entry)
        self._save()

        print(f"[FAISS] Habit memory stored: {furniture_label} | {action}")
        return entry

    def upsert_dynamic_object(self, label: str, room: str, last_seen_on: str,
                               spatial_rel: str, furniture_pos,
                               seen_count: int, interact_count: int,
                               interacted_by: list):

        label = label.lower().strip()

        used_str = f"used by {', '.join(interacted_by)}." if interacted_by else ""

        memory_text = (
            f"{label} {spatial_rel} {last_seen_on} in {room}. "
            f"seen {seen_count} times. interacted {interact_count} times. "
            f"{used_str}"
        ).strip()

        existing_idx = next(
            (i for i, m in enumerate(self.dynamic_metadata) if m.get("label") == label),
            None
        )

        old = self.dynamic_metadata[existing_idx] if existing_idx is not None else None

        position_changed = (
            old is None or
            old.get("last_seen_on") != last_seen_on or
            old.get("room") != room
        )

        entry = {
            "label": label,
            "room": room,
            "last_seen_on": last_seen_on,
            "spatial_rel": spatial_rel,
            "furniture_pos": furniture_pos,
            "seen_count": seen_count,
            "interact_count": interact_count,
            "interacted_by": interacted_by,
            "memory_text": memory_text,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        if position_changed:
            vec = self.model.encode([memory_text]).astype('float32')
            faiss.normalize_L2(vec)

            faiss_idx = self.dynamic_index.ntotal
            self.dynamic_index.add(vec)

            entry["faiss_idx"] = faiss_idx

            reason = "new" if old is None else f"moved: {old.get('last_seen_on')} -> {last_seen_on}"
            print(f"[FAISS Dynamic] '{label}' encode triggered ({reason})")

        else:
            entry["faiss_idx"] = old["faiss_idx"]
            print(f"[FAISS Dynamic] '{label}' metadata updated only")

        if existing_idx is not None:
            self.dynamic_metadata[existing_idx] = entry
        else:
            self.dynamic_metadata.append(entry)

        self._save_dynamic()

    def search_habit(self, query, user_id=None, top_k=5):

        if self.index.ntotal == 0:
            return []

        vec = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(vec)

        k = min(top_k * 5, self.index.ntotal)

        scores, indices = self.index.search(vec, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0 or idx >= len(self.metadata):
                continue

            entry = self.metadata[idx]

            if user_id and entry.get('user') != user_id:
                continue

            results.append({
                "user": entry.get('user'),
                "action": entry.get('action'),
                "instance": entry.get('instance'),
                "interacting_items": entry.get('interacting_items', []),
                "all_items": entry.get('all_items', []),
                "spatial_relations": entry.get('spatial_relations', []),
                "furniture_pos": entry.get('furniture_pos'),
                "mongo_id": entry.get('mongo_id'),
                "description": entry.get('description'),
                "memory_text": entry.get('memory_text'),
                "similarity": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def search_dynamic(self, query, top_k=5, user_filter=None):

        if self.dynamic_index.ntotal == 0:
            return []

        vec = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(vec)

        k = min(top_k * 5, self.dynamic_index.ntotal)

        scores, indices = self.dynamic_index.search(vec, k)

        seen_labels = set()
        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            candidates = [
                m for m in self.dynamic_metadata if m.get("faiss_idx") == idx
            ]

            if not candidates:
                continue

            entry = candidates[-1]

            label = entry.get("label")

            if label in seen_labels:
                continue

            seen_labels.add(label)

            if user_filter and user_filter not in entry.get("interacted_by", []):
                continue

            results.append({
                "label": label,
                "room": entry.get("room"),
                "last_seen_on": entry.get("last_seen_on"),
                "spatial_rel": entry.get("spatial_rel"),
                "furniture_pos": entry.get("furniture_pos"),
                "seen_count": entry.get("seen_count", 0),
                "interact_count": entry.get("interact_count", 0),
                "interacted_by": entry.get("interacted_by", []),
                "memory_text": entry.get("memory_text"),
                "similarity": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def get_top_habit(self, query, user_id=None, top_k=1):

        results = self.search_habit(query, user_id=user_id, top_k=20)

        habit_count = {}

        for r in results:

            key = r['instance']

            if key not in habit_count:
                habit_count[key] = {
                    "instance": r['instance'],
                    "furniture_pos": r['furniture_pos'],
                    "count": 0,
                    "actions": [],
                    "interacting_items": [],
                    "all_items": [],
                }

            habit_count[key]['count'] += 1
            habit_count[key]['actions'].append(r['action'])
            habit_count[key]['interacting_items'].extend(r.get('interacting_items', []))
            habit_count[key]['all_items'].extend(r.get('all_items', []))

        sorted_habits = sorted(habit_count.values(), key=lambda x: x['count'], reverse=True)

        if not sorted_habits:
            return None

        top = sorted_habits[:top_k]

        for h in top:
            h['interacting_items'] = list(set(h['interacting_items']))
            h['all_items'] = list(set(h['all_items']))

        return top[0] if top_k == 1 else top

    def expand_query(self, query: str, candidate_items: list,
                     top_k=10, threshold=0.35) -> list:

        if not candidate_items:
            return []

        try:

            q_vec = self.model.encode(query)

            scored = []

            for item in candidate_items:

                i_vec = self.model.encode(item)

                sim = float(
                    np.dot(q_vec, i_vec) /
                    (np.linalg.norm(q_vec) * np.linalg.norm(i_vec) + 1e-8)
                )

                if sim >= threshold:
                    scored.append((item, sim))

            scored.sort(key=lambda x: x[1], reverse=True)

            result = [item for item, _ in scored[:top_k]]

            print(f"[SemanticExpand] '{query}' -> {result}")

            return result

        except Exception as e:

            print(f"[SemanticExpand] {e}")

            return []

    def get_all_known_items(self) -> list:

        items = set()

        for m in self.metadata:

            for item in m.get('interacting_items', []):
                items.add(item.lower())

            for item in m.get('all_items', []):
                items.add(item.lower())

        for m in self.dynamic_metadata:

            if m.get("label"):
                items.add(m["label"].lower())

        return list(items)

    def _save(self):

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _save_dynamic(self):

        faiss.write_index(self.dynamic_index, self.dynamic_index_path)

        with open(self.dynamic_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.dynamic_metadata, f, ensure_ascii=False, indent=2)