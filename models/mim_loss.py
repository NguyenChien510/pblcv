r"""
Mutual Information Minimization (MIM) Loss Layer
Enforces orthogonality between Appearance and Motion representation spaces:
  1. V_app \perp V_mot (Static Appearance does not collapse into dynamic trajectory)
  2. Optional: T_app \perp T_mot if dual text streams are used.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualInformationMinimizationLoss(nn.Module):
    """
    Computes cosine-orthogonality MIM Loss between disentangled streams.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        V_app: torch.Tensor,
        V_mot: torch.Tensor,
        T_app: Optional[torch.Tensor] = None,
        T_mot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            V_app: Video Appearance embeddings [B, D]
            V_mot: Video Motion embeddings [B, D]
            T_app: Optional Text Appearance embeddings [B, D]
            T_mot: Optional Text Motion embeddings [B, D]
        Returns:
            l_mim: Scalar MIM Loss value
        """
        v_app_n = F.normalize(V_app, p=2, dim=-1)
        v_mot_n = F.normalize(V_mot, p=2, dim=-1)

        # Video orthogonality penalty: |<v_app, v_mot>|
        video_overlap = torch.abs(torch.sum(v_app_n * v_mot_n, dim=-1)) # [B]
        l_mim = torch.mean(video_overlap)

        if T_app is not None and T_mot is not None:
            t_app_n = F.normalize(T_app, p=2, dim=-1)
            t_mot_n = F.normalize(T_mot, p=2, dim=-1)
            text_overlap = torch.abs(torch.sum(t_app_n * t_mot_n, dim=-1))
            l_mim = l_mim + torch.mean(text_overlap)

        return l_mim
