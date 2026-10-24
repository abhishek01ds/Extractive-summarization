import os
import hashlib
import pickle
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from src.preprocessing import Sentence
from src.utils import setup_logger

logger = setup_logger("SGCSumm.Embeddings")


def cartesian_to_hyperspherical(vectors: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Converts Cartesian coordinates to hyperspherical angles so all values are non-negative.
    """
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]
        
    N, m = vectors.shape
    B = np.zeros((N, m), dtype=np.float32)
    
    # Radius (magnitude)
    r = np.linalg.norm(vectors, axis=1)
    B[:, 0] = r
    
    # Calculate suffix sums of squares for angles
    reversed_sq = vectors[:, ::-1] ** 2
    suffix_norm_sq = np.cumsum(reversed_sq, axis=1)[:, ::-1]
    suffix_norm = np.sqrt(np.maximum(suffix_norm_sq, eps))
    
    # Angles between 0 and pi
    for j in range(m - 2):
        denom = suffix_norm[:, j]
        ratio = np.clip(vectors[:, j] / denom, -1.0, 1.0)
        phi_j = np.arccos(ratio)
        B[:, j + 1] = phi_j
        
    # Last angle between 0 and 2*pi
    phi_last = np.arctan2(vectors[:, m - 1], vectors[:, m - 2])
    phi_last = np.where(phi_last < 0, phi_last + 2 * np.pi, phi_last)
    B[:, m - 1] = phi_last
    
    return B


def eliminate_first_component(B: np.ndarray) -> np.ndarray:
    """
    Removes the first component (radius) to keep only the positive angles.
    """
    if B.ndim == 1:
        return B[1:]
    return B[:, 1:]


def modular_mapping_F(B_prime: np.ndarray) -> np.ndarray:
    """
    Ensures all features are non-negative.
    """
    return np.maximum(B_prime, 0.0)


class MockBERTEmbedder:
    """
    Offline mock embedder for fast local testing without downloading BERT.
    """
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def embed_sentences(self, sentences: List[Union[str, Sentence]]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
            
        embeddings = []
        for s in sentences:
            text = s.raw_text if isinstance(s, Sentence) else str(s)
            
            # Deterministic hash seed based on text content
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.RandomState(seed)
            
            base_vec = rng.randn(self.embedding_dim).astype(np.float32)
            layer_vecs = [base_vec + 0.1 * rng.randn(self.embedding_dim).astype(np.float32) for _ in range(11)]
            avg_emb = np.mean(layer_vecs, axis=0)
            embeddings.append(avg_emb)
            
        return np.array(embeddings, dtype=np.float32)


class BERTSentenceEmbedder:
    """
    Extracts sentence embeddings using pretrained BERT.
    Averages encoder layers 2 to 12.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        layers_to_average: Optional[List[int]] = None,
        device: str = "cpu",
        use_cache: bool = True,
        cache_dir: str = "./data/cache",
        use_mock: bool = False
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.layers_to_average = layers_to_average or list(range(2, 13))
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.use_mock = use_mock
        
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.tokenizer = None
        self.model = None
        self.mock_embedder = None
        
        if self.use_mock:
            logger.info("Initializing MockBERTEmbedder (offline mode)...")
            self.mock_embedder = MockBERTEmbedder(embedding_dim=self.embedding_dim)
        else:
            self._init_bert_model()

    def _init_bert_model(self):
        try:
            logger.info(f"Loading pretrained {self.model_name} onto {self.device}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(
                self.model_name,
                output_hidden_states=True
            ).to(self.device)
            self.model.eval()
            logger.info("BERT model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to load {self.model_name} online ({e}). "
                "Falling back to MockBERTEmbedder for offline execution."
            )
            self.use_mock = True
            self.mock_embedder = MockBERTEmbedder(embedding_dim=self.embedding_dim)

    def _get_cache_path(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.pkl")

    def embed_sentences(
        self,
        sentences: List[Union[str, Sentence]],
        batch_size: int = 16
    ) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self.use_mock:
            return self.mock_embedder.embed_sentences(sentences)

        raw_texts = [s.raw_text if isinstance(s, Sentence) else str(s) for s in sentences]
        results: List[Optional[np.ndarray]] = [None] * len(raw_texts)
        uncached_indices = []

        # Check local cache
        if self.use_cache:
            for idx, text in enumerate(raw_texts):
                cache_file = self._get_cache_path(text)
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "rb") as f:
                            results[idx] = pickle.load(f)
                    except Exception:
                        uncached_indices.append(idx)
                else:
                    uncached_indices.append(idx)
        else:
            uncached_indices = list(range(len(raw_texts)))

        # Process uncached sentences in batches
        if uncached_indices:
            for i in range(0, len(uncached_indices), batch_size):
                batch_idxs = uncached_indices[i:i + batch_size]
                batch_texts = [raw_texts[idx] for idx in batch_idxs]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.hidden_states
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)

                    # Average layers 2 through 12
                    selected_layers = [hidden_states[k] for k in self.layers_to_average]
                    stacked_layers = torch.stack(selected_layers, dim=0)
                    avg_layers = torch.mean(stacked_layers, dim=0)
                    
                    # Mean pooling over token embeddings
                    masked_embeddings = avg_layers * attention_mask
                    sum_embeddings = torch.sum(masked_embeddings, dim=1)
                    token_counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    sentence_embs = (sum_embeddings / token_counts).cpu().numpy()

                for local_idx, global_idx in enumerate(batch_idxs):
                    emb = sentence_embs[local_idx].astype(np.float32)
                    results[global_idx] = emb
                    if self.use_cache:
                        try:
                            with open(self._get_cache_path(raw_texts[global_idx]), "wb") as f:
                                pickle.dump(emb, f)
                        except Exception:
                            pass

        return np.array(results, dtype=np.float32)

    def extract_suitable_features(
        self,
        sentences: List[Union[str, Sentence]]
    ) -> np.ndarray:
        # 1. Get BERT embeddings
        A = self.embed_sentences(sentences)
        if len(A) == 0:
            return np.zeros((0, self.embedding_dim - 1), dtype=np.float32)
            
        # 2. Convert to hyperspherical coordinates
        B = cartesian_to_hyperspherical(A)
        # 3. Drop radius component
        B_prime = eliminate_first_component(B)
        # 4. Non-negative mapping
        C = modular_mapping_F(B_prime)
        return C
