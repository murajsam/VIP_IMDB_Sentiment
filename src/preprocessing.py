"""
preprocessing.py
----------------
Reproduktivan pipeline za pripremu podataka (zahtev domaćeg zadatka:
"Svi koraci čišćenja i transformacije moraju biti reproducibilno zapisani i
sačuvani").

Ideja: sva logika kojom se tekst pretvara u ulaz za model (tokenizacija,
mapiranje reči u indekse, skraćivanje/dopuna) enkapsulirana je u jednu klasu
`SentimentPreprocessor`. Ta klasa se može SAČUVATI na disk (`save`) i kasnije
UČITATI (`load`) tako da trening, evaluacija i Streamlit aplikacija koriste
POTPUNO ISTU pripremu podataka. To garantuje da model u produkciji "vidi" ulaz
na isti način kao tokom treninga.

IMDB skup je već tokenizovan (reči su zamenjene rangovima), pa se "čišćenje"
svodi na:
  * proveru i uklanjanje nevalidnih tokena,
  * ograničavanje vokabulara na `num_words` najčešćih reči (retke reči -> OOV),
  * skraćivanje/dopunu sekvenci na `max_len`.
Za sirov tekst (Streamlit) dodatno se radi standardna tokenizacija
(mala slova + izdvajanje reči).
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict

import numpy as np

from . import data as D


@dataclass
class PreprocessConfig:
    num_words: int = 20000
    max_len: int = 200


class SentimentPreprocessor:
    """Pretvara recenzije (sirove rangove ili sirov tekst) u matrice indeksa."""

    def __init__(self, cfg: PreprocessConfig, word_index: dict | None = None):
        self.cfg = cfg
        self.word_index = word_index if word_index is not None else D.load_word_index()
        self.reverse_index = D.build_reverse_index(self.word_index)

    # -- osnovne osobine ----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return D.vocab_size_for(self.cfg.num_words)

    @property
    def pad_id(self) -> int:
        return D.PAD_ID

    # -- transformacije -----------------------------------------------------
    def transform_ranks(self, sequences) -> np.ndarray:
        """Lista sirovih rang-sekvenci -> matrica (N, max_len)."""
        return D.encode_batch(sequences, self.cfg.num_words, self.cfg.max_len)

    def transform_text(self, text: str) -> np.ndarray:
        """Jedan sirov tekst -> vektor (max_len,) [koristi Streamlit aplikacija]."""
        return D.encode_text(text, self.word_index, self.cfg.num_words, self.cfg.max_len)

    def transform_texts(self, texts) -> np.ndarray:
        """Lista sirovih tekstova -> matrica (N, max_len)."""
        return np.stack([self.transform_text(t) for t in texts])

    def decode(self, seq) -> str:
        """Sirov niz rangova -> tekst (za proveru/EDA)."""
        return D.decode_ranks(seq, self.reverse_index)

    # -- upis / čitanje sa diska -------------------------------------------
    def save(self, path: str) -> None:
        """Čuva konfiguraciju pipeline-a (num_words, max_len) u JSON.

        `word_index` se ne duplira jer se preuzima iz keširanog IMDB fajla,
        ali čuvamo njegov identitet (broj reči) radi provere konzistentnosti.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "word_index_size": len(self.word_index),
            "pad_id": D.PAD_ID,
            "oov_id": D.OOV_ID,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SentimentPreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = PreprocessConfig(**payload["config"])
        return cls(cfg)


def build_preprocessor(num_words: int, max_len: int) -> SentimentPreprocessor:
    return SentimentPreprocessor(PreprocessConfig(num_words=num_words, max_len=max_len))
