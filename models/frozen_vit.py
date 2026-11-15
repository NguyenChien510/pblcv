"""
Space-Time Transformer of Frozen in Time (Bain et al., ICCV 2021), the visual encoder MFGF builds on.

Adapted from https://github.com/m-bain/frozen-in-time/blob/main/model/video_transformer.py
    MIT License, Copyright (c) 2021 Max Bain
Changes: no timm.models.layers import (removed in recent timm), attention through
F.scaled_dot_product_attention, gradient checkpointing per block, forward_tokens() returns all tokens.

Weights (load_frozen_weights; PBL_NO_FROZEN=1 skips 1., PBL_REQUIRE_FROZEN=1 raises instead of using 2.):
  1. the Frozen-in-Time CC-3M + WebVid-2M checkpoint (4 frames, base_patch16_224), or
  2. the timm ImageNet-21k ViT-B/16 (fine-tuned on ImageNet-1k) the Frozen-in-Time training started from,
     with the temporal attention initialised as in Frozen in Time (time_init='zeros').
"""

import os
import pickle
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

FROZEN_URL = ("https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/"
              "cc-webvid2m-4f_stformer_b_16_224.pth.tar")
TIMM_INIT = "vit_base_patch16_224.augreg_in21k_ft_in1k"


def _attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, scale=1.0)     # q is already scaled


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class VideoPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=4):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2 * num_frames
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, Fr, C, H, W = x.shape
        assert Fr <= self.num_frames
        return self.proj(x.reshape(-1, C, H, W))


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0., initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        cls_out = _attn(cls_q, k, v)                       # CLS attends to every patch of every frame
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))
        out = _attn(q_, torch.cat((cls_k, k_), dim=1), torch.cat((cls_v, v_), dim=1))
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        out = torch.cat((cls_out, out), dim=1)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.proj_drop(self.proj(out))


class SpaceTimeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, time_init='zeros', use_temporal_attn=True):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.timeattn = VarAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                     proj_drop=drop, initialize=time_init)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x, time_n, space_f):
        if self.use_temporal_attn:
            time_residual = x + self.timeattn(self.norm3(x), 'b (f n) d', '(b n) f d', n=time_n)
        else:                                              # ablation: frames only meet through the CLS token
            time_residual = x
        space_output = self.attn(self.norm1(time_residual), 'b (f n) d', '(b f) n d', f=space_f)
        space_residual = x + space_output                  # 'frozen-in-time' attention style
        return space_residual + self.mlp(self.norm2(space_residual))


class SpaceTimeTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., num_frames=4, time_init='zeros', use_temporal_attn=True):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = VideoPatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames)
        self.patches_per_frame = self.patch_embed.num_patches // num_frames
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patches_per_frame + 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, norm_layer=norm_layer,
                           time_init=time_init, use_temporal_attn=use_temporal_attn)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward_tokens(self, x):
        """[B, F, C, H, W] -> all tokens after the final norm, [B, 1 + F * N, D] (CLS first)."""
        b, curr_frames = x.shape[:2]
        x = self.patch_embed(x).flatten(2).transpose(2, 1).reshape(b, -1, self.embed_dim)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = torch.cat([cls_embed, tile_pos_embed + tile_temporal_embed], dim=1)
        x = self.pos_drop(x + total_pos_embed[:, :x.shape[1]])
        n = self.patches_per_frame
        for blk in self.blocks:
            if self.training and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, n, curr_frames, use_reentrant=False)
            else:
                x = blk(x, n, curr_frames)
        return self.norm(x)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _LenientUnpickler(pickle.Unpickler):
    """The Frozen-in-Time checkpoint pickles objects of its own repo (config parser); replace them with stubs."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Dummy


class _LenientPickle:
    Unpickler = _LenientUnpickler
    load = pickle.load
    __name__ = "pickle"


def _download(url: str, dst: str) -> bool:
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000_000:
        return True
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.hub.download_url_to_file(url, dst + ".part", progress=False)
        os.replace(dst + ".part", dst)
        return True
    except Exception as e:
        print(f"[!] [FrozenViT] download failed ({e})")
        return False


def load_frozen_weights(model: SpaceTimeTransformer, cache_dir: str = None) -> str:
    """Loads pretrained weights into model; returns which ones ('frozen-in-time-cc-webvid2m', 'timm:<name>', 'none')."""
    cache_dir = cache_dir or os.path.join(torch.hub.get_dir(), "checkpoints")
    dst = os.path.join(cache_dir, os.path.basename(FROZEN_URL))
    if os.environ.get("PBL_NO_FROZEN", "0") != "1" and _download(FROZEN_URL, dst):
        try:
            ck = torch.load(dst, map_location="cpu", pickle_module=_LenientPickle, weights_only=False)
            sd = ck.get("state_dict", ck)
            pfx = "module.video_model."
            vsd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
            if not vsd:
                pfx = "video_model."
                vsd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
            missing, unexpected = model.load_state_dict(vsd, strict=False)
            missing = [k for k in missing if not k.startswith(("head", "pre_logits", "fc"))]
            if not missing:
                print(f"[+] [FrozenViT] loaded Frozen-in-Time CC-3M+WebVid-2M weights ({len(vsd)} tensors)")
                return "frozen-in-time-cc-webvid2m"
            print(f"[!] [FrozenViT] Frozen-in-Time checkpoint incomplete: missing={missing[:5]}")
        except Exception as e:
            print(f"[!] [FrozenViT] could not read the Frozen-in-Time checkpoint ({e})")
    if os.environ.get("PBL_REQUIRE_FROZEN", "0") == "1":
        raise RuntimeError("PBL_REQUIRE_FROZEN=1 but the Frozen-in-Time checkpoint could not be loaded")
    try:
        import timm
        vit = timm.create_model(TIMM_INIT, pretrained=True)
        sd = vit.state_dict()
        sd = {k: v for k, v in sd.items() if not k.startswith("head")}
        if "pos_embed" in sd and sd["pos_embed"].shape != model.pos_embed.shape:
            raise RuntimeError(f"pos_embed shape {tuple(sd['pos_embed'].shape)}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "timeattn" not in k and "norm3" not in k and k != "temporal_embed"]
        if missing:
            raise RuntimeError(f"missing={missing[:5]}")
        print(f"[+] [FrozenViT] Frozen-in-Time checkpoint unavailable -> spatial weights from timm {TIMM_INIT}")
        return f"timm:{TIMM_INIT}"
    except Exception as e:
        print(f"[!] [FrozenViT] timm ImageNet weights failed ({e}); random init")
        return "none"
