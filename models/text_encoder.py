"""
Standard BERT Text Encoder for Text-to-Video Person Retrieval (TVPR)
Pure ViT + BERT pipeline (NO OpenAI CLIP dependence).

Encodes natural language person descriptions into 768-D normalized embeddings.
Supports:
  - Hugging Face BERT ('bert-base-uncased')
  - Automatic fallback lightweight Transformer if running in isolated offline environment
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModel
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TextEncoder(nn.Module):
    """
    BERT-based Text Encoder mapping natural language person retrieval queries
    into R^{B x D} unit-normalized embedding space.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        max_length: int = 64,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.pretrained = pretrained
        self.max_length = max_length
        self.tokenizer = None
        self.text_backbone = None
        self.fallback = False

        if HAS_TRANSFORMERS and pretrained:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.text_backbone = AutoModel.from_pretrained(model_name)
                hidden_size = self.text_backbone.config.hidden_size
                self.proj = nn.Linear(hidden_size, embed_dim) if hidden_size != embed_dim else nn.Identity()
            except Exception as e:
                print(f"[!] Warning: Could not download/load '{model_name}' ({e}). Falling back to lightweight Transformer.")
                self.fallback = True
        else:
            self.fallback = True

        if self.fallback:
            self.vocab_size = 30522
            self.embedding = nn.Embedding(self.vocab_size, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4, enable_nested_tensor=False)
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(embed_dim)

    def _simple_tokenize(self, texts: List[str], max_len: int = 64, device: torch.device = None) -> torch.Tensor:
        """Tokenize string list into input_ids tensor when offline."""
        input_ids = []
        for t in texts:
            words = str(t).lower().split()[:max_len]
            ids = [hash(w) % (self.vocab_size - 2) + 1 for w in words]
            padded = ids + [0] * (max_len - len(ids))
            input_ids.append(padded)
        return torch.tensor(input_ids, dtype=torch.long, device=device)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encodes a list of natural language text strings into L2-normalized embeddings.
        Args:
            texts: List of N strings
            device: torch.device
        Returns:
            Tensor of shape [N, embed_dim], L2 normalized.
        """
        if not self.fallback and self.tokenizer is not None and self.text_backbone is not None:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(device)
            outputs = self.text_backbone(**inputs)
            # Pool [CLS] token (index 0)
            cls_feat = outputs.last_hidden_state[:, 0, :]
            feat = self.proj(cls_feat)
        else:
            ids = self._simple_tokenize(texts, max_len=self.max_length, device=device)
            x = self.embedding(ids)
            x = self.transformer(x)
            feat = self.proj(x[:, 0, :])

        feat = self.norm(feat)
        return F.normalize(feat, p=2, dim=-1)

    def forward(
        self,
        texts: Union[List[str], Tuple[List[str], List[str]]],
        q_mot: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method supporting both single text list and legacy dual (q_app, q_mot) calls.
        """
        # Determine device
        if device is None:
            if hasattr(self, "proj") and hasattr(self.proj, "weight") and self.proj.weight is not None:
                device = self.proj.weight.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If called as forward(q_app, q_mot, device)
        if q_mot is not None:
            q_app = texts
            t_app = self.encode_text(q_app, device=device)
            t_mot = self.encode_text(q_mot, device=device)
            return t_app, t_mot

        # Single list of texts
        if isinstance(texts, (list, tuple)) and len(texts) > 0 and isinstance(texts[0], (list, tuple)):
            # Nested tuple (q_app, q_mot)
            t_app = self.encode_text(texts[0], device=device)
            t_mot = self.encode_text(texts[1], device=device)
            return t_app, t_mot

        return self.encode_text(texts, device=device)
