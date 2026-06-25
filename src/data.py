"""
data.py
-------
Učitavanje i priprema IMDB skupa podataka za analizu sentimenta.

IMDB "Large Movie Review Dataset" sadrži 50.000 filmskih recenzija
(25.000 za trening + 25.000 za test), ravnomerno podeljenih na:
    * pozitivne recenzije  -> labela 1
    * negativne recenzije  -> labela 0

Koristi se zvanična, već tokenizovana verzija skupa (Keras/TensorFlow format:
`imdb.npz` + `imdb_word_index.json`). U toj verziji je svaka recenzija zapisana
kao niz celih brojeva, gde svaki broj predstavlja RANG reči po učestalosti
(1 = najčešća reč "the", 17 = "movie", ...). Time je korak tokenizacije već
urađen na standardan, reproduktivan način, pa se možemo fokusirati na model.

Ovaj modul obezbeđuje:
  * preuzimanje i keširanje podataka (bez potrebe za TensorFlow-om),
  * mapiranje sirovih rangova reči u kompaktan vokabular sa rezervisanim
    tokenima PAD (0) i OOV (1),
  * dekodiranje niza brojeva nazad u tekst (koristi se u EDA i za proveru),
  * kodiranje SIROVOG teksta korisnika u nizove indeksa (koristi se kasnije u
    Streamlit aplikaciji za seminarski rad),
  * `padding`/`truncation` do fiksne dužine,
  * PyTorch `Dataset` klasu.
"""

from __future__ import annotations

import os
import re
import json
import urllib.request
from dataclasses import dataclass

import numpy as np

# Rezervisani tokeni u kompaktnom vokabularu.
PAD_ID = 0   # popuna kraćih sekvenci
OOV_ID = 1   # reč van vokabulara (Out Of Vocabulary)
FIRST_WORD_ID = 2  # prvi slobodan indeks za stvarne reči

# URL-ovi zvaničnog IMDB skupa (Keras ogledalo na Google Storage-u).
_IMDB_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
_IMDB_WORDS_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json"

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ----------------------------------------------------------------------------
# Preuzimanje
# ----------------------------------------------------------------------------
def _ensure_file(url: str, path: str) -> str:
    """Preuzima fajl samo ako već ne postoji lokalno (keširanje)."""
    if not os.path.exists(path):
        print(f"[data] Preuzimam {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
    return path


def download_imdb(data_dir: str = _DATA_DIR) -> tuple[str, str]:
    """Preuzima IMDB podatke ako već nisu keširani. Vraća putanje do fajlova."""
    os.makedirs(data_dir, exist_ok=True)
    npz_path = _ensure_file(_IMDB_NPZ_URL, os.path.join(data_dir, "imdb.npz"))
    words_path = _ensure_file(_IMDB_WORDS_URL, os.path.join(data_dir, "imdb_word_index.json"))
    return npz_path, words_path


# ----------------------------------------------------------------------------
# Vokabular
# ----------------------------------------------------------------------------
def load_word_index(data_dir: str = _DATA_DIR) -> dict:
    """Učitava mapu reč -> rang (npr. 'the' -> 1).

    Preuzima SAMO rečnik reči (ne i ceo dataset) — bitno za aplikaciju/Docker,
    gde trening podaci nisu potrebni.
    """
    os.makedirs(data_dir, exist_ok=True)
    words_path = _ensure_file(_IMDB_WORDS_URL, os.path.join(data_dir, "imdb_word_index.json"))
    with open(words_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reverse_index(word_index: dict) -> dict:
    """Mapa rang -> reč (za dekodiranje sekvenci u tekst)."""
    return {rank: word for word, rank in word_index.items()}


def rank_to_id(rank: int, num_words: int) -> int:
    """Mapira sirovi rang reči u kompaktan indeks vokabulara.

    - rangovi 1..(num_words-1) -> stvarne reči (indeksi 2..num_words),
    - sve ostalo (preretke reči) -> OOV (indeks 1).
    """
    if 1 <= rank < num_words:
        return rank + FIRST_WORD_ID - 1
    return OOV_ID


def vocab_size_for(num_words: int) -> int:
    """Ukupna veličina embedding tabele: PAD + OOV + (num_words-1) reči."""
    return num_words + 1


# ----------------------------------------------------------------------------
# Učitavanje sirovih sekvenci
# ----------------------------------------------------------------------------
def load_raw(data_dir: str = _DATA_DIR):
    """Vraća (x_train, y_train, x_test, y_test) sa sirovim rangovima reči."""
    npz_path, _ = download_imdb(data_dir)
    with np.load(npz_path, allow_pickle=True) as f:
        x_train = list(f["x_train"])
        y_train = f["y_train"].astype(np.int64)
        x_test = list(f["x_test"])
        y_test = f["y_test"].astype(np.int64)
    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Kodiranje / dekodiranje
# ----------------------------------------------------------------------------
def encode_ranks(seq, num_words: int, max_len: int) -> np.ndarray:
    """Sirov niz rangova -> kompaktni indeksi, skraćen/dopunjen na `max_len`."""
    ids = [rank_to_id(int(r), num_words) for r in seq][:max_len]
    if len(ids) < max_len:
        ids = ids + [PAD_ID] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def encode_batch(sequences, num_words: int, max_len: int) -> np.ndarray:
    """Kodira listu sekvenci u matricu oblika (N, max_len)."""
    return np.stack([encode_ranks(s, num_words, max_len) for s in sequences])


_TOKEN_RE = re.compile(r"[a-z']+")


def encode_text(text: str, word_index: dict, num_words: int, max_len: int) -> np.ndarray:
    """Kodira SIROV tekst korisnika (koristi se u Streamlit aplikaciji).

    Postupak (isti kao standardna Keras tokenizacija):
      1. sve na mala slova,
      2. izdvajanje reči (bez interpunkcije),
      3. reč -> rang preko `word_index`,
      4. rang -> kompaktni indeks,
      5. skraćivanje/dopuna na `max_len`.
    """
    words = _TOKEN_RE.findall(text.lower())
    ranks = [word_index.get(w) for w in words]
    ranks = [r for r in ranks if r is not None]
    return encode_ranks(ranks, num_words, max_len)


def decode_ranks(seq, reverse_index: dict) -> str:
    """Dekodira sirov niz rangova nazad u tekst (za EDA / proveru)."""
    return " ".join(reverse_index.get(int(r), "?") for r in seq)


# ----------------------------------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Parametri pripreme podataka (svesno odvojeni od hiperparametara modela)."""
    num_words: int = 20000   # veličina vokabulara (top N najčešćih reči)
    max_len: int = 200       # dužina sekvence posle padding/truncation
    subset_size: int | None = None  # ako je zadato, koristi se podskup (brže na CPU)
    seed: int = 42


def make_dataset(sequences, labels, cfg: DataConfig):
    """Kreira PyTorch TensorDataset iz sekvenci i labela."""
    import torch
    from torch.utils.data import TensorDataset

    X = encode_batch(sequences, cfg.num_words, cfg.max_len)
    y = np.asarray(labels, dtype=np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def get_train_test(cfg: DataConfig, data_dir: str = _DATA_DIR):
    """Vraća (train_sequences, train_labels, test_sequences, test_labels).

    Ako je `subset_size` zadat, uzima se stratifikovan (uravnotežen) podskup
    trening podataka radi bržeg treniranja na CPU-u.
    """
    x_train, y_train, x_test, y_test = load_raw(data_dir)

    if cfg.subset_size is not None and cfg.subset_size < len(x_train):
        rng = np.random.RandomState(cfg.seed)
        # Uzimamo jednak broj primera iz obe klase (uravnotežen podskup).
        half = cfg.subset_size // 2
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]
        sel = np.concatenate([
            rng.choice(idx_pos, half, replace=False),
            rng.choice(idx_neg, cfg.subset_size - half, replace=False),
        ])
        rng.shuffle(sel)
        x_train = [x_train[i] for i in sel]
        y_train = y_train[sel]

    return x_train, y_train, x_test, y_test


CLASS_NAMES = ["Negativan (0)", "Pozitivan (1)"]
