"""
app.py
------
Streamlit front-end aplikacija za seminarski rad.

Korisnik unese filmsku recenziju (na engleskom), a aplikacija pomoću
istreniranog Transformer modela predviđa da li je recenzija POZITIVNA ili
NEGATIVNA i prikazuje verovatnoću.

Ključno: aplikacija koristi ISTI preprocessing pipeline kao trening
(models/preprocessor.json + src/preprocessing.py), pa model "vidi" tekst na
identičan način kao tokom treniranja.

Pokretanje (iz korena projekta):
    streamlit run app/app.py
"""

import os
import sys
import json

import numpy as np
import torch
import streamlit as st

# Omogućavamo uvoz paketa src iz korena projekta.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.preprocessing import SentimentPreprocessor          # noqa: E402
from src.model import ModelConfig, build_model               # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")

EXAMPLES = {
    "Pozitivan primer": "This movie was absolutely wonderful. The acting was "
                        "brilliant, the story kept me on the edge of my seat and "
                        "the ending was perfect. One of the best films I have seen!",
    "Negativan primer": "What a terrible waste of time. The plot made no sense, "
                        "the acting was awful and I was bored from the first "
                        "minute. I would not recommend this movie to anyone.",
    "Mešovit primer":   "The visuals were beautiful and the soundtrack was great, "
                        "but the story was weak and way too long.",
}


@st.cache_resource
def load_model_and_preprocessor():
    """Model i preprocessor se učitavaju samo jednom (keširano)."""
    with open(os.path.join(MODEL_DIR, "best_model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = ModelConfig(**meta["model"])
    model = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                                     map_location="cpu"))
    model.eval()
    pre = SentimentPreprocessor.load(os.path.join(MODEL_DIR, "preprocessor.json"))
    return model, pre, meta


@torch.no_grad()
def predict(text: str, model, pre):
    x = torch.from_numpy(pre.transform_text(text)).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1).squeeze(0).numpy()
    return float(probs[0]), float(probs[1])  # (negativan, pozitivan)


# ----------------------------------------------------------------------------
# Interfejs
# ----------------------------------------------------------------------------
st.set_page_config(page_title="IMDB Analiza sentimenta", page_icon="🎬")

st.title("🎬 Analiza sentimenta filmskih recenzija")
st.markdown(
    "Transformer model (PyTorch) predviđa da li je filmska recenzija "
    "**pozitivna** ili **negativna**. Unesite recenziju na engleskom jeziku "
    "ili izaberite jedan od primera."
)

model, pre, meta = load_model_and_preprocessor()

# Primeri na klik
cols = st.columns(len(EXAMPLES))
for col, (name, txt) in zip(cols, EXAMPLES.items()):
    if col.button(name):
        st.session_state["review_text"] = txt

text = st.text_area(
    "Filmska recenzija:",
    value=st.session_state.get("review_text", ""),
    height=150,
    placeholder="npr. This movie was fantastic, I really enjoyed it...",
)

if st.button("Analiziraj", type="primary"):
    if not text.strip():
        st.warning("Unesite tekst recenzije.")
    else:
        p_neg, p_pos = predict(text, model, pre)
        if p_pos >= 0.5:
            st.success(f"**POZITIVNA recenzija** — pouzdanost {p_pos:.1%}")
        else:
            st.error(f"**NEGATIVNA recenzija** — pouzdanost {p_neg:.1%}")
        st.progress(p_pos, text=f"Verovatnoća pozitivnog sentimenta: {p_pos:.1%}")

        with st.expander("Kako model 'vidi' ovaj tekst?"):
            ids = pre.transform_text(text)
            n_tokens = int((ids != 0).sum())
            st.write(f"- Broj prepoznatih tokena: **{n_tokens}** (max_len = {pre.cfg.max_len})")
            st.write(f"- Veličina vokabulara: **{pre.vocab_size}**")
            st.write(f"- Prvih 20 indeksa: `{ids[:20].tolist()}`")

# Bočna traka sa informacijama o modelu
with st.sidebar:
    st.header("O modelu")
    m = meta["model"]
    st.markdown(
        f"- **Arhitektura:** Transformer enkoder\n"
        f"- **d_model:** {m['d_model']}\n"
        f"- **Glave pažnje:** {m['n_heads']}\n"
        f"- **Enkoder slojevi:** {m['n_layers']}\n"
        f"- **Poziciono kodiranje:** {m['pos_encoding']}\n"
        f"- **Vokabular:** {m['vocab_size']:,} reči\n"
        f"- **Dužina sekvence:** {m['max_len']}"
    )
    tm = meta.get("test_metrics", {})
    if tm:
        st.header("Metrike (test skup, 25k)")
        st.markdown(
            f"- **Accuracy:** {tm['accuracy']:.3f}\n"
            f"- **F1:** {tm['f1']:.3f}\n"
            f"- **ROC-AUC:** {tm['roc_auc']:.3f}"
        )
    st.caption("Seminarski rad – Veštačka inteligencija sa primenama")
