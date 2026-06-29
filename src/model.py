"""
model.py
--------
Arhitektura neuronske mreže (zahtev: "Arhitektura mora biti definisana u
zasebnom Python modulu").

Implementiran je Transformer ENKODER "od nule", u skladu sa gradivom sa
predavanja (lekcija "Transformeri i mehanizam pažnje"). Model se sastoji od:

    Ulaz (indeksi reči)
        │
        ▼
    Embedding sloj  (reč -> vektor), skaliran sa sqrt(d_model)
        │  + Poziciono kodiranje (sinusoidno ili naučeno)
        ▼
    N x Transformer enkoder blok:
          ├─ Multi-Head Self-Attention  (Q, K, V, skalirani dot-product)
          ├─ Add & Norm                 (rezidualna veza + LayerNorm)
          ├─ Feed-Forward mreža (FFN)    (Linear -> aktivacija -> Linear)
          └─ Add & Norm                 (rezidualna veza + LayerNorm)
        │
        ▼
    Maskirano usrednjavanje po tokenima (mean pooling, bez PAD tokena)
        │
        ▼
    Klasifikaciona glava (Linear) -> 2 logita (negativan / pozitivan)

Svi delovi su namerno napisani eksplicitno kako bi se svaki korak mogao
objasniti na odbrani.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Konfiguracija modela
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int = 200
    d_model: int = 64          # dimenzija embedding vektora / modela
    n_heads: int = 2           # broj glava pažnje (Multi-Head Attention)
    n_layers: int = 1          # broj enkoder blokova
    dim_ff: int = 128          # unutrašnja dimenzija Feed-Forward mreže
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"   # "sinusoidal" ili "learned"
    activation: str = "relu"           # "relu" ili "gelu"
    num_classes: int = 2
    pad_id: int = 0


# ----------------------------------------------------------------------------
# Poziciono kodiranje
# ----------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Determinističko sinusoidno poziciono kodiranje (Vaswani et al., 2017).

    Za svaku poziciju `pos` i dimenziju `i`:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Ne trenira se (nema parametara), samo dodaje informaciju o redosledu reči.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Naučeno poziciono kodiranje: poseban embedding vektor za svaku poziciju."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos_emb(positions)


# ----------------------------------------------------------------------------
# Multi-Head Self-Attention (implementiran od nule)
# ----------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Paralelni mehanizam pažnje sa više glava.

    Za svaku glavu se računa skalirani dot-product attention:
        Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V
    Deljenje sa sqrt(d_k) sprečava prevelike vrednosti pre softmax-a
    (stabilizuje gradijente). Izlazi svih glava se konkateniraju i propuste
    kroz izlaznu linearnu projekciju.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model mora biti deljivo sa n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Jedna linearna projekcija po ulogama Q, K, V (za sve glave odjednom).
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # izlazna projekcija
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, d_model);  key_padding_mask: (B, T) True gde je PAD
        B, T, _ = x.shape

        def split_heads(t):
            # (B, T, d_model) -> (B, n_heads, T, d_k)
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.w_q(x))
        K = split_heads(self.w_k(x))
        V = split_heads(self.w_v(x))

        # Skalirani skalarni proizvod: (B, n_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if key_padding_mask is not None:
            # Maskiramo PAD pozicije tako da posle softmax-a imaju težinu ~0.
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, T)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Ponderisana suma vrednosti: (B, n_heads, T, d_k)
        context = torch.matmul(attn, V)
        # Spajanje glava nazad: (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(context)


# ----------------------------------------------------------------------------
# Feed-Forward mreža
# ----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> aktivacija -> Dropout -> Linear.

    Primenjuje se nezavisno na svaki token i uvodi nelinearnost."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ----------------------------------------------------------------------------
# Enkoder blok (Add & Norm nakon svakog podsloja - "post-norm")
# ----------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn = FeedForward(cfg.d_model, cfg.dim_ff, cfg.dropout, cfg.activation)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None):
        # 1) Self-attention + rezidualna veza + LayerNorm
        attn_out = self.attn(x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # 2) Feed-Forward + rezidualna veza + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ----------------------------------------------------------------------------
# Ceo model za klasifikaciju sentimenta
# ----------------------------------------------------------------------------
class SentimentTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)

        if cfg.pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(cfg.d_model, cfg.max_len)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)

        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.cfg.pad_id].fill_(0.0)

    def forward(self, x):
        # x: (B, T) indeksi reči
        pad_mask = x == self.cfg.pad_id           # (B, T) True gde je PAD
        h = self.embedding(x) * math.sqrt(self.cfg.d_model)
        h = self.pos_enc(h)
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)

        # Maskirano usrednjavanje: prosek reprezentacija SAMO nad stvarnim tokenima.
        valid = (~pad_mask).unsqueeze(-1).float()          # (B, T, 1)
        summed = (h * valid).sum(dim=1)                     # (B, d_model)
        counts = valid.sum(dim=1).clamp(min=1.0)            # (B, 1)
        pooled = summed / counts
        return self.classifier(pooled)                     # (B, num_classes)


def build_model(cfg: ModelConfig) -> SentimentTransformer:
    return SentimentTransformer(cfg)
