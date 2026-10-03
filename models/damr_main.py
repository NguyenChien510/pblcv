r"""
Divided Space-Time DAMR Architecture for Text-to-Video Person Retrieval (TVPR)
Pure ViT + BERT pipeline (NO OpenAI CLIP dependence).

Integrates:
  - SpaceTimeViT (Divided Space-Time Attention across 16 video frames):
      * V_app: Static Appearance Representation (Attention-pooled across frames)
      * V_mot: Dynamic Trajectory & Viewpoint Evolution (Bidirectional GRU)
      * V_video: Motion-Enhanced Fused Representation (MLP[V_app, V_mot])
  - TextEncoder (BERT-base-uncased):
      * T: Natural language person description query embedding
  - MutualInformationMinimizationLoss:
      * L_MIM: Enforces V_app \perp V_mot (orthogonality)
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .spacetime_vit import SpaceTimeViT
from .mim_loss import MutualInformationMinimizationLoss


class DAMRModel(nn.Module):
    """
    Spatio-Temporal Disentangled Appearance-Motion Representation Model (SpaceTime-DAMR).
    """
    def __init__(
        self,
        embed_dim: int = 768,
        text_model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        motion_frame_num: int = 16,
        img_size: int = 224,
        vit_depth: int = 6,
        vit_heads: int = 12,
        drop_rate: float = 0.1,
        pretrained_vit: bool = True,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = motion_frame_num
        self.img_size = img_size

        # 1. Text Encoder (BERT-base-uncased, 768-D)
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            model_name=text_model_name,
            pretrained=pretrained
        )

        # 2. Spatio-Temporal Video Encoder (Divided Space-Time ViT, 768-D)
        load_spatial = kwargs.get("pretrained_vit", pretrained_vit)
        self.video_encoder = SpaceTimeViT(
            img_size=img_size,
            patch_size=16,
            num_frames=motion_frame_num,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            drop_rate=drop_rate,
            pretrained=load_spatial
        )

        # 3. Orthogonality Loss
        self.mim_loss_layer = MutualInformationMinimizationLoss()

    def encode_text(
        self,
        texts: Union[List[str], Tuple[List[str], ...]],
        q_mot: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes query texts. Supports both single full query and legacy (q_app, q_mot) calls.
        """
        if device is None:
            device = next(self.parameters()).device

        if q_mot is not None:
            t_app = self.text_encoder.encode_text(texts, device=device)
            t_mot = self.text_encoder.encode_text(q_mot, device=device)
            return t_app, t_mot

        if isinstance(texts, (list, tuple)) and len(texts) > 0 and isinstance(texts[0], (list, tuple)):
            t_app = self.text_encoder.encode_text(texts[0], device=device)
            t_mot = self.text_encoder.encode_text(texts[1], device=device)
            return t_app, t_mot

        return self.text_encoder.encode_text(texts, device=device)

    def encode_video(
        self,
        video_frames: torch.Tensor,
        motion_frames: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes video frames using SpaceTimeViT.
        Args:
            video_frames: [B, T, C, H, W]
            motion_frames: Optional legacy argument.
        Returns:
            V_app: [B, D]
            V_mot: [B, D]
            V_video: [B, D]
        """
        if motion_frames is not None and motion_frames.size(1) >= video_frames.size(1):
            target_frames = motion_frames
        else:
            target_frames = video_frames

        # SpaceTimeViT expects [B, T, C, H, W]
        if target_frames.dim() == 4:
            target_frames = target_frames.unsqueeze(1)

        return self.video_encoder(target_frames)

    def forward(
        self,
        *args,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Disentangled Semantic Concept Alignment (DSCA) forward pass:
          - Encodes full caption T_full
          - Encodes appearance sub-prompt T_app
          - Encodes motion sub-prompt T_mot
          - Encodes video into V_app, V_mot, V_video
        """
        device = next(self.parameters()).device

        # Unpack arguments
        q_app = kwargs.get("q_app", None)
        q_mot = kwargs.get("q_mot", None)

        if len(args) == 4:
            # Legacy call: q_app, q_mot, keyframes, motion_frames
            q_app, q_mot, keyframes, motion_frames = args
            texts = q_app
            target_frames = motion_frames if motion_frames.size(1) >= keyframes.size(1) else keyframes
        elif len(args) == 2:
            texts, target_frames = args
        elif len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            texts = batch.get("caption", batch.get("q_app"))
            q_app = batch.get("q_app", None)
            q_mot = batch.get("q_mot", None)
            target_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes")))
        else:
            texts = kwargs.get("caption", kwargs.get("captions", kwargs.get("q_app")))
            target_frames = kwargs.get("video_frames", kwargs.get("motion_frames", kwargs.get("keyframes")))

        # 1. Encode text queries
        if isinstance(texts, str):
            texts = [texts]
        T_full = self.text_encoder.encode_text(texts, device=device)

        if q_app is not None and q_mot is not None:
            if isinstance(q_app, str): q_app = [q_app]
            if isinstance(q_mot, str): q_mot = [q_mot]
            T_app = self.text_encoder.encode_text(q_app, device=device)
            T_mot = self.text_encoder.encode_text(q_mot, device=device)
        else:
            T_app = T_full
            T_mot = T_full

        # 2. Encode video frames
        target_frames = target_frames.to(device)
        V_app, V_mot, V_video = self.encode_video(target_frames)

        # 3. Enforce orthogonality between Appearance and Motion representations
        l_mim = self.mim_loss_layer(V_app, V_mot)

        return {
            "T": T_full,
            "T_full": T_full,
            "T_app": T_app,
            "T_mot": T_mot,
            "V": V_video,
            "V_video": V_video,
            "V_app": V_app,
            "V_mot": V_mot,
            "l_mim": l_mim,
        }
