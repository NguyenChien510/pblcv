"""
Video encoder with the same backbones as MFGF (Ni et al., ACM MM 2024):
  - Visual Encoder: Frozen-in-Time Space-Time ViT-B/16 on 4 discrete frames (models/frozen_vit.py)
  - Motion Encoder: S3D on 16 consecutive frames (torchvision, Kinetics-400 weights)
on top of which the SpaceTime-DSCA heads produce the three streams used by DAMRModel:
  - V_app:   CLS token of the Space-Time ViT (the Frozen-in-Time video representation) -> app_head
  - V_mot:   pooled 1024-D S3D clip feature -> mot_head
  - V_video: fusion MLP([V_app, V_mot])

Input layout (data/dataset.py, frame_layout="mfgf"): frames [B, 20, C, H, W], ImageNet-normalised;
frames[:, :4] are the discrete frames, frames[:, 4:] the 16 consecutive frames.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .frozen_vit import SpaceTimeTransformer, load_frozen_weights

NUM_VISUAL_FRAMES = 4
NUM_MOTION_FRAMES = 16


class MFGFVideoEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, drop_rate: float = 0.1, pretrained: bool = True,
                 use_temporal_attn: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.vit = SpaceTimeTransformer(num_frames=NUM_VISUAL_FRAMES, time_init="zeros",
                                        use_temporal_attn=use_temporal_attn)
        self.weights_source = load_frozen_weights(self.vit) if pretrained else "none"

        from torchvision.models.video import s3d, S3D_Weights
        self.s3d = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None).features
        self.freeze_s3d_bn = pretrained                    # random init (no_init) has no statistics worth keeping
        # frames arrive ImageNet-normalised; S3D expects the Kinetics statistics
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("k_mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("k_std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1), persistent=False)

        self.app_head = nn.Sequential(nn.Linear(self.vit.embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.mot_head = nn.Sequential(nn.Linear(1024, embed_dim), nn.LayerNorm(embed_dim))
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.freeze_s3d_bn:
            return self
        for m in self.s3d.modules():                       # keep the Kinetics BatchNorm statistics (batch is only 16)
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        return self

    def _s3d_features(self, clip: torch.Tensor) -> torch.Tensor:
        x = (clip * self.in_std + self.in_mean - self.k_mean) / self.k_std
        x = x.permute(0, 2, 1, 3, 4).contiguous()          # [B, C, T, H, W]
        if self.training:
            x = checkpoint.checkpoint_sequential(self.s3d, 4, x, use_reentrant=False)
        else:
            x = self.s3d(x)
        return x.mean(dim=(2, 3, 4))

    def forward(self, video_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if video_frames.size(1) == 1:
            # Image mode (text-image person pre-training, experiments/pretrain_cuhk.py): the Space-Time ViT sees one
            # frame, the S3D motion stream is skipped and V_mot is a zero vector.
            cls = self.vit.forward_tokens(video_frames)[:, 0]
            V_app = F.normalize(self.app_head(cls), p=2, dim=-1)
            V_mot = torch.zeros_like(V_app)
            V_video = F.normalize(self.fusion_head(torch.cat([V_app, V_mot], dim=-1)), p=2, dim=-1)
            return V_app, V_mot, V_video
        if video_frames.size(1) != NUM_VISUAL_FRAMES + NUM_MOTION_FRAMES:
            raise ValueError(f"MFGFVideoEncoder expects {NUM_VISUAL_FRAMES + NUM_MOTION_FRAMES} frames "
                             f"(frame_layout='mfgf'), got {video_frames.size(1)}")
        visual, motion = video_frames[:, :NUM_VISUAL_FRAMES], video_frames[:, NUM_VISUAL_FRAMES:]
        cls = self.vit.forward_tokens(visual)[:, 0]
        V_app = F.normalize(self.app_head(cls), p=2, dim=-1)
        V_mot = F.normalize(self.mot_head(self._s3d_features(motion)), p=2, dim=-1)
        V_video = F.normalize(self.fusion_head(torch.cat([V_app, V_mot], dim=-1)), p=2, dim=-1)
        return V_app, V_mot, V_video
