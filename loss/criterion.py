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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Contrastive Loss with Softmax Temperature tau.

    Optional (used by recipe v2, off by default so the original loss is unchanged):
      - keys:  per-sample video ids; other captions of the same video in the batch count as positives
               instead of negatives (multi-positive InfoNCE).
      - queue: (text, video, keys) of earlier batches (detached); extra negatives for both directions.
               Queue entries of the same video as the query are masked out.
      - scale: logit scale 1/tau (tensor, e.g. a learnable parameter) used instead of 1/temperature.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_embeds: torch.Tensor, video_embeds: torch.Tensor,
                keys: Optional[torch.Tensor] = None, queue: Optional[Tuple[torch.Tensor, ...]] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_embeds: Normalized text tensor [B, D]
            video_embeds: Normalized video tensor [B, D]
        Returns:
            loss: Symmetric cross-entropy InfoNCE loss
        """
        b = text_embeds.size(0)
        if keys is None and queue is None and scale is None:
            labels = torch.arange(b, device=text_embeds.device, dtype=torch.long)

            sim = torch.matmul(text_embeds, video_embeds.t()) / self.temperature # [B, B]

            loss_t2v = F.cross_entropy(sim, labels)
            loss_v2t = F.cross_entropy(sim.t(), labels)

            return 0.5 * (loss_t2v + loss_v2t)

        scale = scale if scale is not None else torch.tensor(1.0 / self.temperature, device=text_embeds.device)
        if keys is None:
            keys = torch.arange(b, device=text_embeds.device)
        pos = keys[:, None] == keys[None, :]                                   # [B, B], diagonal always True
        sim = torch.matmul(text_embeds, video_embeds.t()).float() * scale     # [B, B]
        t2v, v2t = sim, sim.t()
        if queue is not None and queue[0].size(0) > 0:
            q_text, q_video, q_keys = queue
            same = keys[:, None] == q_keys[None, :]                            # [B, Q] -> masked out
            q_t2v = (torch.matmul(text_embeds, q_video.t().to(text_embeds.dtype)).float() * scale).masked_fill(same, float("-inf"))
            q_v2t = (torch.matmul(video_embeds, q_text.t().to(video_embeds.dtype)).float() * scale).masked_fill(same, float("-inf"))
            t2v = torch.cat([t2v, q_t2v], dim=1)
            v2t = torch.cat([v2t, q_v2t], dim=1)
        pad = t2v.size(1) - b
        pos_full = torch.cat([pos, pos.new_zeros(b, pad)], dim=1) if pad else pos

        def multi_pos_ce(logits):
            return (torch.logsumexp(logits, dim=1)
                    - torch.logsumexp(logits.masked_fill(~pos_full, float("-inf")), dim=1)).mean()

        return 0.5 * (multi_pos_ce(t2v) + multi_pos_ce(v2t))


class EmbeddingQueue:
    """FIFO memory of detached (text, video, key) embeddings from earlier batches, one per loss stream."""

    def __init__(self, size: int):
        self.size = int(size)
        self.text = self.video = self.keys = None

    def get(self):
        return None if self.text is None else (self.text, self.video, self.keys)

    @torch.no_grad()
    def push(self, text: torch.Tensor, video: torch.Tensor, keys: torch.Tensor) -> None:
        text, video = text.detach().float(), video.detach().float()
        if self.text is None:
            self.text, self.video, self.keys = text, video, keys.clone()
        else:
            self.text = torch.cat([text, self.text])[: self.size]
            self.video = torch.cat([video, self.video])[: self.size]
            self.keys = torch.cat([keys, self.keys])[: self.size]


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
        learnable_tau: bool = False,
        queue_size: int = 0,
        **kwargs
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_app = lambda_app
        self.lambda_mot = lambda_mot
        self.lambda_mim = lambda_mim
        self.infonce = InfoNCELoss(temperature=temperature)
        # Recipe v2 options (off by default): learnable 1/tau (clamped at 100, as in CLIP) and a cross-batch queue
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log()) if learnable_tau else None
        self.queue_size = int(queue_size)
        self.queues = {k: EmbeddingQueue(queue_size) for k in ("common", "app", "mot")} if queue_size > 0 else None
        self.queue_active = False
        self._key_ids: Dict[str, int] = {}
        # text_pos (recipe m3): in the app / mot streams, batch items whose sub-prompt Q_app / Q_mot is the same
        # string are positives of each other instead of negatives (identical texts cannot be told apart)
        self.text_pos = bool(kwargs.get("text_pos", False))

    def _scale(self) -> Optional[torch.Tensor]:
        if self.logit_scale is None:
            return None
        return self.logit_scale.clamp(max=math.log(100.0)).exp()

    def _keys_tensor(self, video_keys, device) -> Optional[torch.Tensor]:
        if video_keys is None:
            return None
        ids = [self._key_ids.setdefault(str(k), len(self._key_ids)) for k in video_keys]
        return torch.tensor(ids, device=device, dtype=torch.long)

    @staticmethod
    def _stream_keys(video_keys, texts, device) -> torch.Tensor:
        """Group id per batch item: same video OR same normalised sub-prompt -> same group (union-find)."""
        n = len(video_keys)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for attr in (list(map(str, video_keys)), [" ".join(str(t).lower().split()) for t in texts]):
            first = {}
            for i, a in enumerate(attr):
                if a in first:
                    parent[find(i)] = find(first[a])
                else:
                    first[a] = i
        return torch.tensor([find(i) for i in range(n)], device=device, dtype=torch.long)

    def _nce(self, stream: str, text, video, keys):
        if keys is None and self.logit_scale is None and self.queues is None:
            return self.infonce(text, video)
        queue = self.queues[stream].get() if (self.queues is not None and self.queue_active) else None
        loss = self.infonce(text, video, keys=keys, queue=queue, scale=self._scale())
        if self.queues is not None and keys is not None and self.training:
            self.queues[stream].push(F.normalize(text.float(), dim=-1), F.normalize(video.float(), dim=-1), keys)
        return loss

    def forward(
        self,
        T: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        V_app: Optional[torch.Tensor] = None,
        V_mot: Optional[torch.Tensor] = None,
        l_mim: Optional[torch.Tensor] = None,
        T_app: Optional[torch.Tensor] = None,
        T_mot: Optional[torch.Tensor] = None,
        video_keys: Optional[Any] = None,
        app_texts: Optional[Any] = None,
        mot_texts: Optional[Any] = None,
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
        keys = self._keys_tensor(video_keys, text_feat.device)
        l_common = self._nce("common", text_feat, vid_feat, keys)

        # 2. Fine-Grained Appearance Alignment Loss: T_app <-> V_app
        app_text = T_app if T_app is not None else text_feat
        app_keys = mot_keys = keys
        if self.text_pos and app_texts is not None and mot_texts is not None and video_keys is not None:
            app_keys = self._stream_keys(video_keys, app_texts, text_feat.device)
            mot_keys = self._stream_keys(video_keys, mot_texts, text_feat.device)
        if V_app is not None:
            l_app = self._nce("app", app_text, V_app, app_keys)
        else:
            l_app = torch.tensor(0.0, device=text_feat.device)

        # 3. Fine-Grained Motion Alignment Loss: T_mot <-> V_mot
        mot_text = T_mot if T_mot is not None else text_feat
        if V_mot is not None:
            l_mot = self._nce("mot", mot_text, V_mot, mot_keys)
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
        if self.logit_scale is not None:
            breakdown["tau"] = float(1.0 / self._scale().item())

        return l_total, breakdown
