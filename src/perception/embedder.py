"""
embedder.py
-----------
Generates CLIP (ViT-B/32) or SigLIP multimodal embeddings for frames.

Both image → vector and text → vector are supported, enabling cross-modal
retrieval (e.g. query "white truck near fence" to find matching frames).

Usage:
    embedder = FrameEmbedder()
    img_vec  = embedder.embed_image(pil_image)   # shape (512,)
    txt_vec  = embedder.embed_text("person at gate")  # shape (512,)
"""

import os
import numpy as np
from PIL import Image
from typing import List


class FrameEmbedder:
    """
    CLIP-based multimodal embedder.

    Set EMBED_BACKEND env var to "clip" (default) or "mock".
    """

    def __init__(self, backend: str | None = None, model_name: str = "ViT-B-32"):
        self.backend = backend or os.getenv("EMBED_BACKEND", "mock")
        self.dim = 512

        if self.backend == "clip":
            try:
                self._load_clip(model_name)
            except Exception as e:
                print(f"[Embedder] Failed to load CLIP ({e}), falling back to MOCK embedder.")
                self.backend = "mock"
        elif self.backend == "mock":
            print("[Embedder] Using MOCK embedder (random unit vectors, no GPU needed).")
        else:
            raise ValueError(f"Unknown embed backend: {self.backend!r}")

    def _load_clip(self, model_name: str):
        import open_clip
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Loading {model_name} on {self.device} …")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.to(self.device).eval()
        import torch
        self._torch = torch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Return a normalised float32 embedding vector for a PIL image."""
        if self.backend == "mock":
            return _random_unit_vector(self.dim)
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Return a normalised float32 embedding vector for a text query."""
        if self.backend == "mock":
            return _random_unit_vector(self.dim)
        tokens = self._tokenizer([text]).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple text strings. Returns shape (N, dim)."""
        return np.stack([self.embed_text(t) for t in texts])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _random_unit_vector(dim: int) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)
