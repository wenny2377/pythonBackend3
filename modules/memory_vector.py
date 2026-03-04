import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorMemory:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        # 1. Load embedding model (recommended to match app.py to keep semantic space consistent)
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # MiniLM embedding dimension
        
        # 2. Initialize FAISS index (using L2 distance for similarity search)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # 3. Store raw text and metadata (FAISS stores only vectors, not text)
        self.metadata = []

    def add_memory(self, user, action, instance, description=""):
        """
        Convert behavior and detailed description into a vector and store it
        🚀 Update: Now accepts 4 parameters (user, action, instance, description)
        """
        # Construct richer semantic text for better future retrieval
        text = f"{user} is {action} at {instance}. Observation: {description}"
        
        # Convert to embedding vector
        vector = self.model.encode([text]).astype('float32')
        
        # Add to FAISS index
        self.index.add(vector)
        
        # Store metadata for returning results to Unity after retrieval
        self.metadata.append({
            "text": text,
            "user": user,
            "action": action,
            "instance": instance,
            "description": description
        })
        print(f"[Vector Memory] Semantic memory stored: {text}")

    def search_habit(self, query, top_k=2):
        """
        Semantic retrieval: e.g., input 'Where is the apple?' or 'What did Mom do?'
        """
        if self.index.ntotal == 0:
            print("⚠️ [FAISS] No memories stored yet.")
            return []

        # Convert query sentence into embedding vector
        query_vector = self.model.encode([query]).astype('float32')
        
        # Perform search
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # Ensure valid index
                match_data = self.metadata[idx].copy()
                match_data["score"] = float(distances[0][i])  # Smaller distance = more similar
                results.append(match_data)
        
        return results

    def clear(self):
        """Clear memory (for demo reset purposes)"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        print("🧹 [FAISS] Vector memory has been cleared.")