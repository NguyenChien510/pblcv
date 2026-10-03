"""
Divided Space-Time Vision Transformer (SpaceTimeViT) for Video Person Representation
Implements Divided Space-Time Attention (TimeSformer style):
  - Spatial Self-Attention: captures body parts, clothing attributes, and static appearance within each frame.
  - Temporal Self-Attention: captures temporal trajectory, viewpoint variations, and dynamic interactions across frames.
Disentangles into:
  - V_app: Static Appearance Representation (Temporal Attention Pooling)
  - V_mot: Dynamic Trajectory & Viewpoint Representation (Temporal Variation Modeling via Bi-GRU)
  - V_video: Motion-Enhanced Video Representation = MLP([V_app, V_mot])
Pure ViT architecture (NO CLIP dependence).
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class PatchEmbed(nn.Module):
    """2D Image Patch Embedding."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B * T, C, H, W] -> [B * T, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DividedSpaceTimeBlock(nn.Module):
    """
    Divided Space-Time Transformer Block:
      1. Temporal Self-Attention: attends across T frames for each spatial patch.
      2. Spatial Self-Attention: attends across N patches within each frame.
      3. Feedforward MLP.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Temporal Attention
        self.norm_temp = nn.LayerNorm(embed_dim)
        self.attn_temp = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        # TimeSformer zero-init for temporal projection: residual connection acts as identity at start
        nn.init.constant_(self.attn_temp.out_proj.weight, 0.0)
        nn.init.constant_(self.attn_temp.out_proj.bias, 0.0)

        # 2. Spatial Attention
        self.norm_spat = nn.LayerNorm(embed_dim)
        self.attn_spat = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)

        # 3. MLP Feedforward
        self.norm_mlp = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 1 + T * N, embed_dim]
               where x[:, 0] is the global CLS token, followed by T * N patch tokens.
        """
        cls_token = x[:, :1, :] # [B, 1, D]
        patch_tokens = x[:, 1:, :] # [B, T * N, D]

        # --- STEP 1: TEMPORAL ATTENTION ---
        # Reshape to [B * N, T, D] so that each spatial location attends across time
        p_t = patch_tokens.reshape(B, T, N, self.embed_dim).transpose(1, 2).reshape(B * N, T, self.embed_dim)
        p_t_norm = self.norm_temp(p_t)
        attn_out_temp, _ = self.attn_temp(p_t_norm, p_t_norm, p_t_norm, need_weights=False)
        p_t = p_t + attn_out_temp

        # Reshape back to [B, T, N, D] -> [B * T, N, D]
        p_s = p_t.reshape(B, N, T, self.embed_dim).transpose(1, 2).reshape(B * T, N, self.embed_dim)

        # --- STEP 2: SPATIAL ATTENTION ---
        # Expand CLS token to each frame: [B * T, 1, D]
        cls_exp = cls_token.repeat_interleave(T, dim=0) # [B * T, 1, D]
        spat_seq = torch.cat([cls_exp, p_s], dim=1) # [B * T, 1 + N, D]

        spat_norm = self.norm_spat(spat_seq)
        attn_out_spat, _ = self.attn_spat(spat_norm, spat_norm, spat_norm, need_weights=False)
        spat_seq = spat_seq + attn_out_spat

        # Pool updated CLS tokens back to [B, 1, D]
        new_cls = spat_seq[:, :1, :].reshape(B, T, self.embed_dim).mean(dim=1, keepdim=True) # [B, 1, D]
        new_patches = spat_seq[:, 1:, :].reshape(B, T * N, self.embed_dim) # [B, T * N, D]

        # --- STEP 3: MLP FEEDFORWARD ---
        merged = torch.cat([new_cls, new_patches], dim=1) # [B, 1 + T * N, D]
        out = merged + self.mlp(self.norm_mlp(merged))

        return out


class SpaceTimeViT(nn.Module):
    """
    Spatio-Temporal Vision Transformer with Divided Space-Time Attention.
    Supports ImageNet Pretrained Spatial Transfer (TimeSformer / ViViT standard).
    Produces:
      - V_app: Static Appearance Representation
      - V_mot: Dynamic Trajectory & Viewpoint Representation
      - V_video: Integrated Motion-Enhanced Representation
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_frames: int = 16,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        drop_rate: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # 2. Learnable Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, num_patches, embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # 3. Divided Space-Time Transformer Blocks
        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(embed_dim=embed_dim, num_heads=num_heads, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Disentangled Heads
        # A. Appearance Stream: Attention Pooling across frames
        self.app_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )
        self.app_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # B. Motion Stream: Temporal Trajectory & Viewpoint Evolution
        self.motion_gru = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.mot_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # C. Motion-Enhanced Fusion Head: V_video = MLP([V_app, V_mot])
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        nn.init.normal_(self.cls_token, std=0.02)

        if pretrained:
            self.load_pretrained_spatial()

    def load_pretrained_spatial(self) -> bool:
        """
        Loads ImageNet-1K / ImageNet-21k pretrained weights into spatial components:
          - PatchEmbed (conv projection)
          - cls_token
          - spatial_pos (196 patch positions)
          - Block spatial attention, norm layers, and MLPs.
        Supports torchvision.models.vit_b_16 with fallback to transformers.ViTModel.
        """
        # 1. Attempt torchvision ViT-B/16
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            tv_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            sd = tv_vit.state_dict()

            if "conv_proj.weight" in sd:
                self.patch_embed.proj.weight.data.copy_(sd["conv_proj.weight"])
                if self.patch_embed.proj.bias is not None and "conv_proj.bias" in sd:
                    self.patch_embed.proj.bias.data.copy_(sd["conv_proj.bias"])

            if "class_token" in sd:
                self.cls_token.data.copy_(sd["class_token"])

            if "encoder.pos_embedding" in sd:
                # Shape [1, 197, 768] -> pos[:, 1:, :] is [1, 196, 768]
                pos = sd["encoder.pos_embedding"][:, 1:, :]
                if pos.shape[1] == self.num_patches:
                    self.spatial_pos.data.copy_(pos.unsqueeze(0))

            num_layers = min(len(self.blocks), 12)
            for i in range(num_layers):
                pfx = f"encoder.layers.encoder_layer_{i}."
                blk = self.blocks[i]
                if f"{pfx}ln_1.weight" in sd:
                    blk.norm_spat.weight.data.copy_(sd[f"{pfx}ln_1.weight"])
                    blk.norm_spat.bias.data.copy_(sd[f"{pfx}ln_1.bias"])
                if f"{pfx}self_attention.in_proj_weight" in sd:
                    blk.attn_spat.in_proj_weight.data.copy_(sd[f"{pfx}self_attention.in_proj_weight"])
                    blk.attn_spat.in_proj_bias.data.copy_(sd[f"{pfx}self_attention.in_proj_bias"])
                    blk.attn_spat.out_proj.weight.data.copy_(sd[f"{pfx}self_attention.out_proj.weight"])
                    blk.attn_spat.out_proj.bias.data.copy_(sd[f"{pfx}self_attention.out_proj.bias"])
                if f"{pfx}ln_2.weight" in sd:
                    blk.norm_mlp.weight.data.copy_(sd[f"{pfx}ln_2.weight"])
                    blk.norm_mlp.bias.data.copy_(sd[f"{pfx}ln_2.bias"])
                if f"{pfx}mlp.linear_1.weight" in sd:
                    blk.mlp[0].weight.data.copy_(sd[f"{pfx}mlp.linear_1.weight"])
                    blk.mlp[0].bias.data.copy_(sd[f"{pfx}mlp.linear_1.bias"])
                    blk.mlp[3].weight.data.copy_(sd[f"{pfx}mlp.linear_2.weight"])
                    blk.mlp[3].bias.data.copy_(sd[f"{pfx}mlp.linear_2.bias"])
                # Ensure temporal output projection starts at 0 (TimeSformer residual identity)
                nn.init.constant_(blk.attn_temp.out_proj.weight, 0.0)
                nn.init.constant_(blk.attn_temp.out_proj.bias, 0.0)

            print(f"[+] [SpaceTimeViT] Successfully loaded {num_layers} pretrained spatial layers from torchvision ViT-Base (ImageNet-1K).")
            return True
        except Exception as e_tv:
            # 2. Fallback to transformers ViTModel
            try:
                from transformers import ViTModel
                hf_vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
                sd = hf_vit.state_dict()

                if "embeddings.patch_embeddings.projection.weight" in sd:
                    self.patch_embed.proj.weight.data.copy_(sd["embeddings.patch_embeddings.projection.weight"])
                    self.patch_embed.proj.bias.data.copy_(sd["embeddings.patch_embeddings.projection.bias"])

                if "embeddings.cls_token" in sd:
                    self.cls_token.data.copy_(sd["embeddings.cls_token"])

                if "embeddings.position_embeddings" in sd:
                    pos = sd["embeddings.position_embeddings"][:, 1:, :]
                    if pos.shape[1] == self.num_patches:
                        self.spatial_pos.data.copy_(pos.unsqueeze(0))

                num_layers = min(len(self.blocks), 12)
                for i in range(num_layers):
                    pfx = f"encoder.layer.{i}."
                    blk = self.blocks[i]
                    if f"{pfx}layernorm_before.weight" in sd:
                        blk.norm_spat.weight.data.copy_(sd[f"{pfx}layernorm_before.weight"])
                        blk.norm_spat.bias.data.copy_(sd[f"{pfx}layernorm_before.bias"])
                    if f"{pfx}attention.attention.query.weight" in sd:
                        qw = sd[f"{pfx}attention.attention.query.weight"]
                        kw = sd[f"{pfx}attention.attention.key.weight"]
                        vw = sd[f"{pfx}attention.attention.value.weight"]
                        blk.attn_spat.in_proj_weight.data.copy_(torch.cat([qw, kw, vw], dim=0))
                        qb = sd[f"{pfx}attention.attention.query.bias"]
                        kb = sd[f"{pfx}attention.attention.key.bias"]
                        vb = sd[f"{pfx}attention.attention.value.bias"]
                        blk.attn_spat.in_proj_bias.data.copy_(torch.cat([qb, kb, vb], dim=0))
                        blk.attn_spat.out_proj.weight.data.copy_(sd[f"{pfx}attention.output.dense.weight"])
                        blk.attn_spat.out_proj.bias.data.copy_(sd[f"{pfx}attention.output.dense.bias"])
                    if f"{pfx}layernorm_after.weight" in sd:
                        blk.norm_mlp.weight.data.copy_(sd[f"{pfx}layernorm_after.weight"])
                        blk.norm_mlp.bias.data.copy_(sd[f"{pfx}layernorm_after.bias"])
                    if f"{pfx}intermediate.dense.weight" in sd:
                        blk.mlp[0].weight.data.copy_(sd[f"{pfx}intermediate.dense.weight"])
                        blk.mlp[0].bias.data.copy_(sd[f"{pfx}intermediate.dense.bias"])
                        blk.mlp[3].weight.data.copy_(sd[f"{pfx}output.dense.weight"])
                        blk.mlp[3].bias.data.copy_(sd[f"{pfx}output.dense.bias"])
                    nn.init.constant_(blk.attn_temp.out_proj.weight, 0.0)
                    nn.init.constant_(blk.attn_temp.out_proj.bias, 0.0)

                print(f"[+] [SpaceTimeViT] Successfully loaded {num_layers} pretrained spatial layers from Hugging Face ViT-Base.")
                return True
            except Exception as e_hf:
                print(f"[!] [SpaceTimeViT] Pretrained spatial load failed (TorchVision: {e_tv} | HF: {e_hf}). Initializing randomly.")
                return False

    def forward(self, video_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video_frames: Tensor of shape [B, T, C, H, W]
        Returns:
            V_app: Static Appearance Representation [B, embed_dim]
            V_mot: Dynamic Trajectory Representation [B, embed_dim]
            V_video: Fused Motion-Enhanced Representation [B, embed_dim]
        """
        B, T, C, H, W = video_frames.shape
        N = self.num_patches

        # Patch Embed: [B * T, C, H, W] -> [B * T, N, D] -> [B, T, N, D]
        x = video_frames.reshape(B * T, C, H, W)
        patches = self.patch_embed(x).reshape(B, T, N, self.embed_dim)

        # Add Spatial & Temporal Positional Embeddings
        # spatial_pos: [1, 1, N, D], temporal_pos: [1, T, 1, D]
        patches = patches + self.spatial_pos[:, :, :, :] + self.temporal_pos[:, :T, :, :]
        patches = self.pos_drop(patches)

        # Flatten patches: [B, T * N, D]
        patches = patches.reshape(B, T * N, self.embed_dim)

        # Prepend global CLS token: [B, 1 + T * N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)

        # Divided Space-Time Transformer Blocks with Gradient Checkpointing
        for blk in self.blocks:
            if self.training and tokens.requires_grad:
                tokens = checkpoint.checkpoint(blk, tokens, B, T, N, use_reentrant=False)
            else:
                tokens = blk(tokens, B=B, T=T, N=N)

        tokens = self.norm(tokens)

        # --- STREAM 1: APPEARANCE REPRESENTATION (V_app) ---
        # Extract per-frame representations by pooling patches within each frame
        frame_patches = tokens[:, 1:, :].reshape(B, T, N, self.embed_dim) # [B, T, N, D]
        frame_feats = frame_patches.mean(dim=2) # [B, T, D] - frame summary representations

        # Dynamic attention weights across T frames
        attn_weights = F.softmax(self.app_attn(frame_feats), dim=1) # [B, T, 1]
        V_app_raw = torch.sum(attn_weights * frame_feats, dim=1) # [B, D]
        V_app = F.normalize(self.app_head(V_app_raw), p=2, dim=-1)

        # --- STREAM 2: MOTION & TRAJECTORY REPRESENTATION (V_mot) ---
        # Models temporal evolution, viewpoint change, and velocity dynamics through GRU sequence modeling
        gru_out, _ = self.motion_gru(frame_feats) # [B, T, D]
        # Motion vector is the aggregated temporal dynamics
        V_mot_raw = gru_out[:, -1, :] - gru_out[:, 0, :] + gru_out.mean(dim=1)
        V_mot = F.normalize(self.mot_head(V_mot_raw), p=2, dim=-1)

        # --- STREAM 3: MOTION-ENHANCED FUSED REPRESENTATION (V_video) ---
        concat_feats = torch.cat([V_app, V_mot], dim=-1) # [B, 2 * D]
        V_video = F.normalize(self.fusion_head(concat_feats), p=2, dim=-1)

        return V_app, V_mot, V_video
