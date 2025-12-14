from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize (supports (D,) or (N, D)). Returns float32."""
    x = v.astype(np.float32, copy=False)
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return (x / n).astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / n).astype(np.float32)

def _now() -> int:
    return int(time.time())

@dataclass
class Cluster:
    id: int
    centroid: np.ndarray               # (D,) float32, L2-normalized
    members: List[int] = field(default_factory=list)
    last_used: int = field(default_factory=_now)
    hits: int = 0

class EmbeddingSimCache:
    """
    A redundancy-aware, cluster-based semantic cache for lemmas.

    - Clusters maintain mean-pooled, L2-normalized centroids.
    - New items route to the nearest centroid; if too similar to checked members, they are skipped.
    - When a cluster exceeds `cluster_cap`, we select a centroid-proximal batch and
      consolidate near-duplicates: for any group with sim >= `dup_threshold`, keep 1, drop m-1.

    Storage:
      * self.vecs:   (N, D) float32 L2-normalized embedding matrix
      * self.texts:  list[str]
      * self.subjects: list[str]
      * self.cluster_of: list[int]  (cluster id per item)
      * per-cluster member index lists
    """

    def __init__(
        self,
        dim: int,
        max_items: int = 10000,
       
        dup_threshold: float = 0.95,
        merge_threshold: float = 0.90,     # assign to cluster if sim(centroid) >= this
        admit_sample_size: int = 100,      # sample size for near-dup gate in large clusters
      
        rng: Optional[random.Random] = None,
    ):
        self.dim = dim
        self.max_items = max_items
      
        self.dup_threshold = float(dup_threshold)
        self.merge_threshold = float(merge_threshold)
        self.admit_sample_size = int(admit_sample_size)
       
        self.rng = rng or random.Random(123)

        # global storage
        self.vecs = np.empty((0, dim), dtype=np.float32)  # L2-normalized
        self.texts: List[str] = []
        self.subjects: List[str] = []
        self.cluster_of: List[int] = []

        # clusters
        self.clusters: List[Cluster] = []
        self._next_cluster_id = 0

    def add(self, text: str, emb: np.ndarray, subject: str = "", utility: float = 0.0) -> Dict:
        # hard cap: stop admitting once full
        if self.vecs.shape[0] >= self.max_items:
            return {"status": "capacity_full"}

        v = _l2_normalize(emb.reshape(-1))  # (D,)

        # Route to nearest centroid (or create first cluster)
        cluster, sim_c = self._nearest_cluster(v)
        if cluster is None or sim_c < self.merge_threshold:
            cid = self._create_cluster(v)
            idx = self._append_member(cid, text, v, subject)
            # no pruning, no global cap enforcement
            return {"status": "new_cluster", "cluster": cid, "idx": idx, "sim_centroid": float(sim_c)}

        # Near-dup gate against members of that cluster
        if self._is_near_duplicate_in_cluster(cluster, v):
            cluster.hits += 1
            cluster.last_used = _now()
            return {"status": "skipped_dup", "cluster": cluster.id, "sim_centroid": float(sim_c)}

        # Admit to cluster
        idx = self._append_member(cluster.id, text, v, subject)
        self._recompute_centroid(cluster)  # keep centroid up to date
        return {"status": "added", "cluster": cluster.id, "idx": idx, "sim_centroid": float(sim_c)}


    def query_topk(self, q_emb: np.ndarray, k: int = 3) -> List[Dict]:
        """Exact cosine top-k over the whole cache (fast at N ≤ ~10k)."""
        if self.vecs.shape[0] == 0:
            return []
        q = _l2_normalize(q_emb.reshape(-1))
        sims = self.vecs @ q
        k_eff = min(k, sims.shape[0])
        top = np.argpartition(-sims, k_eff - 1)[:k_eff]
        top = top[np.argsort(-sims[top])]
        now = _now()
        results = []
        for i in top:
            cid = self.cluster_of[i]
            self._touch_cluster(cid, now)
            results.append(
                {
                    "idx": int(i),
                    "subject": self.subjects[i],
                    "cluster": int(cid),
                    "score": float(sims[i]),
                    "text": self.texts[i],
                }
            )
        return results

    def stats(self) -> Dict:
        sizes = [len(c.members) for c in self.clusters]
        return {
            "N_items": int(self.vecs.shape[0]),
            "N_clusters": len(self.clusters),
            "sizes": sizes,
            "max_size": max(sizes) if sizes else 0,
        }
    def _nearest_cluster(self, v: np.ndarray) -> Tuple[Optional[Cluster], float]:
        if not self.clusters:
            return None, -1.0
        C = np.vstack([c.centroid for c in self.clusters])  # (M, D)
        sims = C @ v                                        # (M,)
        j = int(np.argmax(sims))
        return self.clusters[j], float(sims[j])

    def _create_cluster(self, v: np.ndarray) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self.clusters.append(Cluster(id=cid, centroid=v.copy()))
        return cid

    def _append_member(self, cid: int, text: str, v: np.ndarray, subject: str) -> int:
        idx = self.vecs.shape[0]
        self.vecs = np.vstack([self.vecs, v.reshape(1, -1)])
        self.texts.append(text)
        self.subjects.append(subject)
        self.cluster_of.append(cid)
        # register to cluster
        self._cluster_by_id(cid).members.append(idx)
        self._touch_cluster(cid)
        return idx

    def _touch_cluster(self, cid: int, t: Optional[int] = None):
        c = self._cluster_by_id(cid)
        c.hits += 1
        c.last_used = t if t is not None else _now()

    def _cluster_by_id(self, cid: int) -> Cluster:
        # clusters indexed by position; small N so linear search is fine
        for c in self.clusters:
            if c.id == cid:
                return c
        raise KeyError(f"Cluster {cid} not found")

    def _is_near_duplicate_in_cluster(self, cluster: Cluster, v_new: np.ndarray) -> bool:
        m = len(cluster.members)
        if m == 0:
            return False
        # small cluster -> exact check
        if m <= self.admit_sample_size:
            sims = self.vecs[cluster.members] @ v_new
            return bool((sims >= self.dup_threshold).any())

        # large cluster -> centroid-proximal sample (biased) up to admit_sample_size
        sims_c = self.vecs[cluster.members] @ cluster.centroid
        # take the top half by centroid similarity and sample from them
        order = np.argsort(-sims_c)
        top_half = order[: max(self.admit_sample_size * 2, self.admit_sample_size)]
        cand_ids = [cluster.members[i] for i in top_half]
        if len(cand_ids) > self.admit_sample_size:
            cand_ids = self.rng.sample(cand_ids, k=self.admit_sample_size)
        sims = self.vecs[cand_ids] @ v_new
        return bool((sims >= self.dup_threshold).any())

    

    
    def _recompute_centroid(self, cluster: Cluster):
        if not cluster.members:
            # keep a zero vector to avoid NaNs; it won't be picked anyway
            cluster.centroid = np.zeros(self.dim, dtype=np.float32)
            return
        mu = self.vecs[cluster.members].mean(axis=0)
        cluster.centroid = _l2_normalize(mu)




