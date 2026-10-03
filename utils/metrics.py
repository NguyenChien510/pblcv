"""
Evaluation Metrics Utility for Spatio-Temporal DAMR (Text-to-Video Person Retrieval)
Computes Rank@K (Rank@1, 5, 10, 50), Median Rank (MdR), Mean Rank, and mAP.

Supports:
  1. Direct Video-Text Similarity: Score(Q_i, V_j) = CosSim(T_i, V_j)
  2. Appearance-Enhanced Similarity: Score = beta * CosSim(T, V_video) + (1 - beta) * CosSim(T, V_app)
  3. Legacy dual-stream similarity: alpha * CosSim(T_app, V_app) + (1 - alpha) * CosSim(T_mot, V_mot)
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F


def compute_similarity_matrix(
    *args,
    **kwargs
) -> np.ndarray:
    """
    Computes cross-modal similarity matrix [N_text, N_video].
    Accepts:
      1. compute_similarity_matrix(T, V)
      2. compute_similarity_matrix(T, V_video, V_app, beta=0.85)
      3. compute_similarity_matrix(T_app, T_mot, V_app, V_mot, alpha=0.85)
    """
    if len(args) == 2:
        T, V = args
        T_n = F.normalize(T, p=2, dim=-1)
        V_n = F.normalize(V, p=2, dim=-1)
        return torch.matmul(T_n, V_n.t()).cpu().numpy()

    if len(args) == 3:
        T, V_video, V_app = args
        beta = kwargs.get("beta", 0.85)
        T_n = F.normalize(T, p=2, dim=-1)
        V_vid_n = F.normalize(V_video, p=2, dim=-1)
        V_app_n = F.normalize(V_app, p=2, dim=-1)
        sim = beta * torch.matmul(T_n, V_vid_n.t()) + (1.0 - beta) * torch.matmul(T_n, V_app_n.t())
        return sim.cpu().numpy()

    if len(args) >= 4:
        T_app, T_mot, V_app, V_mot = args[:4]
        alpha = kwargs.get("alpha", 0.85)
        T_app_n = F.normalize(T_app, p=2, dim=-1)
        T_mot_n = F.normalize(T_mot, p=2, dim=-1)
        V_app_n = F.normalize(V_app, p=2, dim=-1)
        V_mot_n = F.normalize(V_mot, p=2, dim=-1)

        sim_app = torch.matmul(T_app_n, V_app_n.t())
        sim_mot = torch.matmul(T_mot_n, V_mot_n.t())
        sim_total = alpha * sim_app + (1.0 - alpha) * sim_mot
        return sim_total.cpu().numpy()

    # Keyword argument handling
    if "T" in kwargs and ("V" in kwargs or "V_video" in kwargs):
        T = kwargs["T"]
        V = kwargs.get("V", kwargs.get("V_video"))
        T_n = F.normalize(T, p=2, dim=-1)
        V_n = F.normalize(V, p=2, dim=-1)
        return torch.matmul(T_n, V_n.t()).cpu().numpy()

    raise ValueError("Invalid arguments passed to compute_similarity_matrix.")


def evaluate_rank_metrics(
    sim_matrix: np.ndarray,
    pids_text: np.ndarray,
    pids_video: np.ndarray,
    ranks: Tuple[int, ...] = (1, 5, 10, 50)
) -> Dict[str, float]:
    """
    Computes standard Text-to-Video Person Retrieval benchmarks:
      - Rank@1, Rank@5, Rank@10, Rank@50 (%)
      - Median Rank (MdR)
      - Mean Rank
      - Mean Average Precision (mAP)
    """
    num_queries = sim_matrix.shape[0]
    rank_counts = {r: 0 for r in ranks}
    all_ranks = []
    ap_list = []

    for q_idx in range(num_queries):
        pid_q = pids_text[q_idx]
        scores = sim_matrix[q_idx]

        # Sort gallery video samples in descending order of similarity
        sorted_indices = np.argsort(-scores)
        matched = (pids_video[sorted_indices] == pid_q)

        first_match_rank = np.where(matched)[0]
        if len(first_match_rank) > 0:
            rank = first_match_rank[0] + 1
            all_ranks.append(rank)
            for r in ranks:
                if rank <= r:
                    rank_counts[r] += 1
        else:
            all_ranks.append(10000)

        # Average Precision calculation
        if np.any(matched):
            hits = np.where(matched)[0]
            precisions = [(i + 1) / (hit_idx + 1) for i, hit_idx in enumerate(hits)]
            ap_list.append(np.mean(precisions))
        else:
            ap_list.append(0.0)

    results = {}
    for r in ranks:
        results[f"R@{r}"] = (rank_counts[r] / num_queries) * 100.0

    results["MdR"] = float(np.median(all_ranks)) if all_ranks else 0.0
    results["MeanRank"] = float(np.mean(all_ranks)) if all_ranks else 0.0
    results["mAP"] = float(np.mean(ap_list)) * 100.0 if ap_list else 0.0
    return results
