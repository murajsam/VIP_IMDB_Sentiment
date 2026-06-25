# Analiza sentimenta filmskih recenzija (IMDB) – Transformer

Domaći zadatak iz predmeta **Veštačka inteligencija sa primenama** (master studije).

**Tema:** `Transformer – Sentiment Analysis of Movie Reviews`
**Skup podataka:** IMDB Dataset of 50K Movie Reviews
**Tip problema:** binarna klasifikacija teksta (pozitivan / negativan sentiment)

## Plan projekta

- [ ] Učitavanje i eksploratorna analiza podataka (EDA)
- [ ] Reproduktivan preprocessing pipeline
- [ ] Transformer enkoder u PyTorch-u (zaseban modul)
- [ ] Treniranje sa kros-validacijom + MLflow logovanje
- [ ] Poređenje najmanje 5 konfiguracija modela
- [ ] Dokumentacija i rezultati

## Instalacija

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
