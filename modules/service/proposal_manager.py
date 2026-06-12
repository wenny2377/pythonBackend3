"""
proposal_manager.py
───────────────────
Service Layer — Proposal Management

Single responsibility:
  Manage the lifecycle of service proposals:
  - Push new proposals to queue
  - Serve next proposal to Unity (polling)
  - Handle user responses (accept/reject)
  - Prevent duplicate proposals

Collection written:
  - service_proposals
"""

import datetime
import uuid


class ProposalManager:

    def __init__(self, db):
        self.db  = db
        self.col = db.service_proposals

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, user_id: str, proposal: dict) -> str:
        """
        Store a new proposal.
        Returns proposal_id.
        """
        proposal_id = str(uuid.uuid4())
        self.col.insert_one({
            "proposal_id": proposal_id,
            "user_id":     user_id,
            "message":     proposal.get("message", ""),
            "item":        proposal.get("item", ""),
            "item_loc":    proposal.get("item_loc", ""),
            "need":        proposal.get("need", ""),
            "step1":       proposal.get("step1", ""),
            "step2":       proposal.get("step2", ""),
            "confidence":  proposal.get("confidence", 0.0),
            "time_slot":   proposal.get("time_slot", ""),
            "status":      "pending",   # pending / accepted / rejected / ignored
            "created_at":  datetime.datetime.utcnow(),
            "responded_at": None,
        })
        print(f"[ProposalManager] Pushed: {proposal_id} | {proposal.get('item')}")
        return proposal_id

    def get_next(self) -> dict | None:
        """
        Get the next pending proposal (FIFO).
        Called by Unity via GET /service_proposal.
        """
        doc = self.col.find_one(
            {"status": "pending"},
            sort=[("created_at", 1)],
        )
        if not doc:
            return None

        return {
            "proposal_id": doc["proposal_id"],
            "user_id":     doc["user_id"],
            "message":     doc["message"],
            "item":        doc["item"],
            "item_loc":    doc.get("item_loc", ""),
            "confidence":  doc.get("confidence", 0.0),
        }

    def handle_response(self, proposal_id: str, user_id: str,
                         result: str, manifold_engine=None) -> dict:
        """
        Handle user response to a proposal.
        result: "accepted" | "rejected" | "ignored"
        """
        now = datetime.datetime.utcnow()

        self.col.update_one(
            {"proposal_id": proposal_id},
            {"$set": {
                "status":       result,
                "responded_at": now,
            }},
        )

        doc = self.col.find_one({"proposal_id": proposal_id})
        if doc and manifold_engine:
            try:
                manifold_engine.update_service_result(
                    user_id=user_id,
                    action=doc.get("step1", ""),
                    result=result,
                )
            except Exception:
                pass

        print(f"[ProposalManager] Response: {proposal_id} → {result}")

        return {
            "status":      "ok",
            "proposal_id": proposal_id,
            "result":      result,
        }

    def get_history(self, user_id: str = None,
                    limit: int = 50) -> list:
        """Return proposal history, optionally filtered by user."""
        query = {"user_id": user_id} if user_id else {}
        docs  = list(
            self.col.find(query, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        for d in docs:
            for k in ["created_at", "responded_at"]:
                if k in d and hasattr(d[k], "isoformat"):
                    d[k] = d[k].isoformat()
        return docs

    def get_last_proposal_time(self, user_id: str) -> datetime.datetime | None:
        """Return the creation time of the most recent proposal."""
        doc = self.col.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)],
        )
        return doc.get("created_at") if doc else None