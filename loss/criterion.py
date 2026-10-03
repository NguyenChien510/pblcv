"""
Loss Criterion for Spatio-Temporal DAMR (Text-to-Video Person Retrieval)
Pure ViT + BERT pipeline.

Implements:
  1. L_common: Common Space InfoNCE Contrastive Loss between full Query T and Fused Video V_video
  2. L_app: Appearance Stream InfoNCE Contrastive Loss between Query T and Static Appearance V_app
  3. L_MIM: Mutual Information Minimization (Orthogonality between V_app and V_mot)
  4. L_total = L_common + lambda_app * L_app + lambda_mim * L_MIM
"""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Contrastive Loss with Softmax Temperature tau.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_embeds: torch.Tensor, video_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeds: Normalized text tensor [B, D]
            video_embeds: Normalized video tensor [B, D]
        Returns:
            loss: Symmetric cross-entropy InfoNCE loss
        """
        b = text_embeds.size(0)
        labels = torch.arange(b, device=text_embeds.device, dtype=torch.long)

        # Similarity matrix scaled by temperature
        sim = torch.matmul(text_embeds, video_embeds.t()) / self.temperature # [B, B]

        loss_t2v = F.cross_entropy(sim, labels)
        loss_v2t = F.cross_entropy(sim.t(), labels)

        return 0.5 * (loss_t2v + loss_v2t)


class DAMRCriterion(nn.Module):
    """
    Spatio-Temporal DAMR Loss Objective.
    Combines common-space alignment, appearance alignment, and motion-appearance orthogonality.
    """
    def __init__(
        self,
        temperature: float = 0.05,
        lambda_app: float = 0.5,
        lambda_mot: float = 0.25,
        lambda_mim: float = 0.1,
        lambda_cross_neg: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_app = lambda_app
        self.lambda_mot = lambda_mot
        self.lambda_mim = lambda_mim
        self.infonce = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        T: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        V_app: Optional[torch.Tensor] = None,
        V_mot: Optional[torch.Tensor] = None,
        l_mim: Optional[torch.Tensor] = None,
        T_app: Optional[torch.Tensor] = None,
        T_mot: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        r"""
        Computes the complete Disentangled Semantic Concept Alignment (DSCA) objective:
          1. L_common: Joint query T_full <-> Motion-enhanced video V_video
          2. L_app: Appearance sub-prompt T_app <-> Static appearance V_app
          3. L_mot: Motion sub-prompt T_mot <-> Dynamic trajectory V_mot
          4. L_MIM: Mutual information minimization (V_app \perp V_mot)
        """
        # Resolve full text & video tensors
        text_feat = T if T is not None else kwargs.get("T_full", T_app)
        vid_feat = V if V is not None else kwargs.get("V_video", V_app)

        # 1. Primary Joint Alignment Loss: Query Text <-> Motion-Enhanced Video
        l_common = self.infonce(text_feat, vid_feat)

        # 2. Fine-Grained Appearance Alignment Loss: T_app <-> V_app
        app_text = T_app if T_app is not None else text_feat
        if V_app is not None:
            l_app = self.infonce(app_text, V_app)
        else:
            l_app = torch.tensor(0.0, device=text_feat.device)

        # 3. Fine-Grained Motion Alignment Loss: T_mot <-> V_mot
        mot_text = T_mot if T_mot is not None else text_feat
        if V_mot is not None:
            l_mot = self.infonce(mot_text, V_mot)
        else:
            l_mot = torch.tensor(0.0, device=text_feat.device)

        # 4. Orthogonality Loss (Mutual Information Minimization)
        if l_mim is None:
            if V_app is not None and V_mot is not None:
                v_app_n = F.normalize(V_app, p=2, dim=-1)
                v_mot_n = F.normalize(V_mot, p=2, dim=-1)
                l_mim = torch.mean(torch.abs(torch.sum(v_app_n * v_mot_n, dim=-1)))
            else:
                l_mim = torch.tensor(0.0, device=text_feat.device)

        # 5. Total Hierarchical DSCA Loss
        l_total = (
            l_common
            + (self.lambda_app * l_app)
            + (self.lambda_mot * l_mot)
            + (self.lambda_mim * l_mim)
        )

        breakdown = {
            "loss_total": l_total.item(),
            "loss_common": l_common.item(),
            "loss_app": l_app.item(),
            "loss_mot": l_mot.item(),
            "loss_mim": l_mim.item(),
        }

        return l_total, breakdown
