import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import glob

class PhysioFAISSRetriever:
    """
    Chapter 5.2: Risk-Conditioned Retrieval (RCR).
    Implements the mathematical retrieval model:
    Score(q,d) = α*Sim_text(q,d) + β*Sim_risk(r_q, r_d) + γ*RoutePrior(d|state)
    """
    def __init__(self, data_dir="references/rag_data", model_name="all-MiniLM-L6-v2"):
        print(f"Initializing Risk-Conditioned Retrieval (RCR) Engine with {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.data_dir = data_dir
        
        # Retrieval hyperparameters (α, β, γ)
        self.alpha = 0.4  # Weight for text similarity
        self.beta = 0.5   # Weight for physiological risk match
        self.gamma = 0.1  # Weight for document prior

        self.chunks = []
        self.chunk_embeddings = None
        self.index = None
        self.doc_metadata = [] # List of {'content', 'target_risk_profile', 'prior_weight'}

        # Load and index data
        self._build_index()

    def _get_risk_profile(self, category):
        """Assigns default risk targets based on clinical category."""
        if category == 'dbt':
            return {"hr_z_score": 3.0, "hrv_z_score": -1.0, "wakeup_z_score": 0} # Crisis/Self-harm
        elif category == 'act':
            return {"hr_z_score": 0.5, "hrv_z_score": -2.0, "wakeup_z_score": 0.5} # Stress/Acceptance
        elif category == 'cbt':
            return {"hr_z_score": 1.0, "hrv_z_score": -0.5, "wakeup_z_score": 2.0} # Anxiety/Insomnia
        return {"hr_z_score": 0, "hrv_z_score": 0, "wakeup_z_score": 0}

    def _build_index(self):
        print("Building vector index from PDFs...")
        categories = ['act', 'cbt', 'dbt']
        
        for cat in categories:
            cat_path = os.path.join(self.data_dir, cat, "*.pdf")
            files = glob.glob(cat_path)
            risk_profile = self._get_risk_profile(cat)
            
            for f in files:
                try:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Simple chunking (by paragraph or fixed size)
                    file_chunks = [text[i:i+500] for i in range(0, len(text), 400)]
                    
                    for chunk in file_chunks:
                        if len(chunk.strip()) > 50:
                            self.chunks.append(chunk)
                            self.doc_metadata.append({
                                "content": chunk,
                                "target_risk_profile": risk_profile,
                                "prior_weight": 1.0 if cat != 'dbt' else 1.2
                            })
                except Exception as e:
                    print(f"Error processing {f}: {e}")

        if not self.chunks:
            print("Warning: No data found in RAG directory. Falling back to mock data.")
            self._load_mock_data()
            return

        print(f"Total chunks indexed: {len(self.chunks)}")
        self.chunk_embeddings = self.model.encode(self.chunks)
        
        # Build FAISS index
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Inner Product (for cosine similarity with normalized vectors)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.chunk_embeddings)
        self.index.add(self.chunk_embeddings)

    def _load_mock_data(self):
        # Fallback if no PDFs are found
        self.chunks = ["Mock CBT Protocol", "Mock ACT Guide"]
        self.doc_metadata = [
            {"content": "CBT Protocol", "target_risk_profile": {"hr_z_score": 2.5, "hrv_z_score": 0, "wakeup_z_score": 0}, "prior_weight": 1.0},
            {"content": "ACT Guide", "target_risk_profile": {"hr_z_score": 0, "hrv_z_score": -2.0, "wakeup_z_score": 0}, "prior_weight": 1.0}
        ]
        self.chunk_embeddings = self.model.encode(self.chunks)
        self.index = faiss.IndexFlatIP(self.chunk_embeddings.shape[1])
        faiss.normalize_L2(self.chunk_embeddings)
        self.index.add(self.chunk_embeddings)

    def _sim_risk(self, user_z_scores, doc_target_z_scores):
        """Calculates physiological risk similarity using negative L2 distance."""
        dist = 0
        for key in ['hr_z_score', 'hrv_z_score', 'wakeup_z_score']:
            user_val = user_z_scores.get(key, 0)
            doc_val = doc_target_z_scores.get(key, 0)
            dist += (user_val - doc_val) ** 2
        
        similarity = 1.0 / (1.0 + np.sqrt(dist))
        return similarity

    def retrieve(self, text_intent, physio_data, top_k=5):
        """
        Executes Risk-Conditioned Retrieval (RCR).
        """
        if not self.index:
            return "No retrieval index available."

        # 1. Text Similarity (FAISS search)
        query_embedding = self.model.encode([text_intent])
        faiss.normalize_L2(query_embedding)
        
        # Search for more than top_k to re-rank with risk
        D, I = self.index.search(query_embedding, min(len(self.chunks), 20))
        
        user_z_scores = {
            'hr_z_score': physio_data.get('hr_z_score', 0),
            'hrv_z_score': physio_data.get('hrv_z_score', 0),
            'wakeup_z_score': physio_data.get('wakeup_z_score', 0)
        }

        best_doc = None
        best_score = -1.0
        
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            
            sim_text = D[0][i]
            metadata = self.doc_metadata[idx]
            sim_risk = self._sim_risk(user_z_scores, metadata["target_risk_profile"])
            route_prior = metadata["prior_weight"]
            
            # Score(q,d) = α·Sim_text + β·Sim_risk + γ·RoutePrior
            final_score = (self.alpha * sim_text) + (self.beta * sim_risk) + (self.gamma * route_prior)
            
            if final_score > best_score:
                best_score = final_score
                best_doc = metadata["content"]
                
        return best_doc

if __name__ == "__main__":
    retriever = PhysioFAISSRetriever()
    print("\nTesting Real RCR Retrieval:")
    
    # Case 1: High stress physiology (Low HRV)
    vitals_stress = {'hr_z_score': 0.5, 'hrv_z_score': -2.5, 'wakeup_z_score': 1.0}
    context1 = retriever.retrieve(text_intent="I feel overwhelmed by my thoughts.", physio_data=vitals_stress)
    print(f"Case 1 (Overwhelmed, High Stress Vitals) -> Retrieved snippet:\n{context1[:200]}...")
    
    # Case 2: Crisis physiology
    vitals_crisis = {'hr_z_score': 3.5, 'hrv_z_score': -1.0, 'wakeup_z_score': 0}
    context2 = retriever.retrieve(text_intent="Help.", physio_data=vitals_crisis)
    print(f"Case 2 (Minimal text, Crisis Vitals) -> Retrieved snippet:\n{context2[:200]}...")
