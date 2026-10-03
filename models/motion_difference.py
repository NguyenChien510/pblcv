r"""
Motion Difference Stream Encoder (V_mot) for DAMR-LLM
Processes L=16 consecutive frames to extract dynamic motion & action trajectories.
Key Mechanism:
  1. Computes Frame Differences: \Delta F_t = |F_{t+1} - F_t| for t=1..L-1
     - Eliminates static background pixels and fixed clothing color attributes (subtracts to ~0)
     - Retains pure motion vector boundaries (>0)
  2. Temporal Difference Transformer (Diff-Trans) encodes motion sequence into V_mot in R^{B x D}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDifferenceExtractor(nn.Module):
    r"""
    Computes frame-to-frame absolute difference maps: \Delta F_t = |F_{t+1} - F_t|.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Sequence tensor of shape [B, L, C, H, W]
        Returns:
            diffs: Frame difference maps of shape [B, L-1, C, H, W]
        """
        # Calculate elementwise absolute difference between consecutive frames
        f_current = frames[:, :-1, :, :, :] # [B, L-1, C, H, W]
        f_next = frames[:, 1:, :, :, :]    # [B, L-1, C, H, W]
        
        diffs = torch.abs(f_next - f_current)
        return diffs


class MotionDifferenceEncoder(nn.Module):
    """
    Diff-Trans: Lightweight Temporal Difference Transformer Encoder.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_frames: int = 16, # L = 16 (L-1 = 15 difference maps)
        embed_dim: int = 512,
        depth: int = 3,
        num_heads: int = 8
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_diff_frames = num_frames - 1
        self.embed_dim = embed_dim

        self.diff_extractor = FrameDifferenceExtractor()

        # Patch projection for 2D difference maps
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        # Temporal Positional Embeddings across (L-1) difference steps
        self.temporal_pos = nn.Parameter(torch.randn(1, self.num_diff_frames, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Spatial pooling per frame difference map
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)

        # Temporal Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)

    def forward(self, motion_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_frames: Tensor of shape [B, L, C, H, W]
        Returns:
            V_mot: L2-normalized motion embedding of shape [B, embed_dim]
        """
        b, l, c, h, w = motion_frames.shape

        # Step 1: Extract frame difference maps \Delta F_t
        diff_maps = self.diff_extractor(motion_frames) # [B, L-1, C, H, W]
        l_diff = self.num_diff_frames

        # Reshape to process patches: [B * (L-1), C, H, W]
        x = diff_maps.view(b * l_diff, c, h, w)
        patches = self.patch_proj(x).flatten(2) # [B * (L-1), D, N_patches]

        # Spatial average pooling to get 1 motion token per frame difference map
        frame_diff_tokens = self.spatial_pool(patches).squeeze(-1) # [B * (L-1), D]
        frame_diff_tokens = frame_diff_tokens.view(b, l_diff, self.embed_dim) # [B, L-1, D]

        # Add Temporal Positional Embeddings
        frame_diff_tokens = frame_diff_tokens + self.temporal_pos

        # Prepend Motion CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1) # [B, 1, D]
        seq_tokens = torch.cat([cls_tokens, frame_diff_tokens], dim=1) # [B, 1 + (L-1), D]

        # Step 2: Temporal Difference Transformer Encoding
        encoded = self.transformer(seq_tokens) # [B, 1 + (L-1), D]
        encoded = self.norm(encoded)

        # Extract CLS token as V_mot
        V_mot = encoded[:, 0, :] # [B, embed_dim]

        return F.normalize(V_mot, p=2, dim=-1)
