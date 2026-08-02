"""Diversity-based subset selection via farthest-point (k-center greedy) sampling."""

import numpy as np


def farthest_point_sample(embeddings: np.ndarray, k: int, seed: int = 42) -> list[int]:
    """Select ``k`` maximally diverse indices from L2-normalized ``embeddings``.

    Greedy k-center: start from a deterministic seed point, then repeatedly add
    the point with the largest minimum cosine-distance to the already-selected
    set. Returns a list of row indices into ``embeddings``, in selection order.
    """
    n = embeddings.shape[0]
    k = min(k, n)
    if k <= 0:
        return []

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = [first]

    # cosine distance = 1 - dot product (embeddings are L2-normalized)
    min_dist = 1.0 - embeddings @ embeddings[first]
    min_dist[first] = -np.inf

    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist_to_new = 1.0 - embeddings @ embeddings[next_idx]
        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[next_idx] = -np.inf

    return selected
