"""
Visual Appearance Stream Encoder (V_app) for DAMR-LLM
Processes K=4 keyframes of a video clip to extract static appearance attributes (clothing, colors, accessories).
Uses Pretrained CLIP-ViT (openai/clip-vit-base-patch16) + Keyframe Attention Pooling:
  V_app = sum_k ( w_k * CLIP_ViT(F_k) ) in R^{B x D}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModelWithProjection
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

CLIP_NAME_MAP = {
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
}


class KeyframeAttentionPooling(nn.Module):
    """
    Learns dynamic attention weights w_k to aggregate K keyframe features into single V_app vector.
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Keyframe features of shape [B, K, embed_dim]
        Returns:
            V_app: Aggregated visual appearance representation of shape [B, embed_dim]
        """
        weights = self.attn(x) # [B, K, 1]
        weights = F.softmax(weights, dim=1) # Normalize over K keyframes
        v_app = torch.sum(weights * x, dim=1) # [B, D]
        return v_app


class VisualAppearanceEncoder(nn.Module):
    """
    Appearance Stream Encoder processing K keyframes with Pretrained CLIP-ViT backbone.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        use_clip: bool = True,
        pretrained: bool = True,
        depth: int = 4,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_clip = use_clip
        self.clip_model_name = CLIP_NAME_MAP.get(clip_model_name, clip_model_name)
        self.pretrained = pretrained
        self.clip_vision = None
        self.fallback = False

        if self.use_clip:
            if HAS_TRANSFORMERS and pretrained:
                try:
                    self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(self.clip_model_name)
                    clip_dim = getattr(self.clip_vision.config, "projection_dim", 512)
                    self.clip_proj = nn.Linear(clip_dim, embed_dim) if clip_dim != embed_dim else nn.Identity()
                except Exception as e:
                    print(f"[!] Warning: Failed loading pretrained CLIP Vision ({e}). Fallback to lightweight ViT.")
                    self.fallback = True
            else:
                self.fallback = True

        if not self.use_clip or self.fallback:
            # Fallback lightweight custom ViT
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            num_patches = (img_size // patch_size) ** 2
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, activation="gelu",
                batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(embed_dim)
            nn.init.normal_(self.cls_token, std=0.02)

        # Keyframe Attention Pooling
        self.keyframe_pool = KeyframeAttentionPooling(embed_dim=embed_dim)

    def forward(self, keyframes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keyframes: Keyframe tensor of shape [B, K, C, H, W]
        Returns:
            V_app: L2-normalized visual appearance embedding [B, embed_dim]
        """
        b, k, c, h, w = keyframes.shape
        x = keyframes.view(b * k, c, h, w)

        if self.use_clip and not self.fallback and self.clip_vision is not None:
            outputs = self.clip_vision(pixel_values=x)
            cls_feat = outputs.image_embeds # [B * K, 512]
            cls_feat = self.clip_proj(cls_feat)
            cls_feat = cls_feat.view(b, k, self.embed_dim)
        else:
            # Fallback lightweight ViT
            feat = self.proj(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(b * k, -1, -1)
            feat = torch.cat([cls_tokens, feat], dim=1) + self.pos_embed
            feat = self.transformer(feat)
            feat = self.norm(feat)
            cls_feat = feat[:, 0, :].view(b, k, self.embed_dim)

        # Aggregate across K keyframes via Attention Pooling
        V_app = self.keyframe_pool(cls_feat) # [B, D]
        return F.normalize(V_app, p=2, dim=-1)
