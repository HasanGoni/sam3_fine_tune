"""Simple on-disk cache for per-image DINOv3 embeddings.

Keyed by absolute image path so re-running ``build_diverse_wire_dataset.py``
with a different ``target_count`` (e.g. a small dry run followed by the real
1000-per-category run) reuses embeddings already computed for the same
candidate pool instead of re-running the backbone over every image again.
"""

from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Loads/saves a ``{absolute_path: embedding}`` map to a single .npz file."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._store: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            paths = data["paths"]
            embeds = data["embeds"]
            self._store = {str(p): e for p, e in zip(paths.tolist(), embeds)}

    def __len__(self):
        return len(self._store)

    def get_many(self, paths: list[str]) -> dict[str, np.ndarray]:
        """Return the subset of ``paths`` already present in the cache."""
        return {p: self._store[p] for p in paths if p in self._store}

    def update(self, new_items: dict[str, np.ndarray]):
        self._store.update(new_items)

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._store:
            return
        paths = np.array(list(self._store.keys()), dtype="U")
        embeds = np.stack(list(self._store.values())).astype(np.float32)
        # np.savez() always appends ".npz" if the given name doesn't already
        # end with it, so the tmp path itself must end in ".npz" or the
        # rename below silently targets a file that was never created.
        tmp_path = self.cache_path.with_name(self.cache_path.stem + "_tmp.npz")
        np.savez(tmp_path, paths=paths, embeds=embeds)
        tmp_path.replace(self.cache_path)
