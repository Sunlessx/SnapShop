from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class IndexStore:
    """FAISS index + a list of product dicts. Row i in the index == metadata[i]."""

    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        # write to .tmp first then swap, so a crash halfway doesn't leave a broken index
        tmp_index = self.index_path.with_suffix(".index.tmp")
        tmp_meta = self.metadata_path.with_suffix(".pkl.tmp")
        faiss.write_index(self.index, str(tmp_index))
        with open(tmp_meta, "wb") as f:
            pickle.dump(self.metadata, f)
        # replace() instead of rename(): rename fails on windows if the file already exists
        tmp_index.replace(self.index_path)
        tmp_meta.replace(self.metadata_path)

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        dim = embeddings.shape[1]
        # inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query: np.ndarray, top_k: int = 20, sources: set[str] | None = None) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        # no filter -> plain search
        # with a filter we ask faiss for more than we need and drop the other shops.
        # if that wasn't enough we go again with everything (flat index, still cheap)
        k = top_k if not sources else min(self.index.ntotal, top_k * 10)
        while True:
            results = self._search_k(query, k, sources)
            if len(results) >= top_k or k >= self.index.ntotal:
                return results[:top_k]
            k = self.index.ntotal

    def _search_k(self, query: np.ndarray, k: int, sources: set[str] | None) -> list[dict]:
        scores, indices = self.index.search(query.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.metadata[idx]
            # old indexes don't have "source", treat them as the main catalog
            if sources and item.get("source", "main") not in sources:
                continue
            results.append({**item, "score": float(score)})
        return results

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.metadata:
            s = item.get("source", "main")
            counts[s] = counts.get(s, 0) + 1
        return counts

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0
