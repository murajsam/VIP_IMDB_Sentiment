# Analiza sentimenta filmskih recenzija (IMDB) – Transformer

Domaći zadatak iz predmeta **Veštačka inteligencija sa primenama** (master studije).

**Tema:** `Transformer – Sentiment Analysis of Movie Reviews`
**Skup podataka:** IMDB Dataset of 50K Movie Reviews
**Tip problema:** binarna klasifikacija teksta (pozitivan / negativan sentiment)

Projekat pokazuje kompletan tok: eksploratorna analiza podataka → reproduktivan
preprocessing → Transformer enkoder implementiran „od nule" u PyTorch-u →
treniranje sa kros-validacijom i pretragom hiperparametara → praćenje
eksperimenata u MLflow-u → poređenje verzija modela kroz tabele i grafikone.

---

## 1. Struktura projekta

```
VIP_IMDB_Sentiment/
├── README.md                     # ovaj fajl
├── requirements.txt              # zavisnosti
├── configs/
│   └── experiments.yaml          # 6 konfiguracija modela + parametri treninga
├── notebooks/
│   └── 01_eda.ipynb              # eksploratorna analiza podataka (EDA)
├── src/
│   ├── data.py                   # preuzimanje/učitavanje IMDB, kodiranje, vokabular
│   ├── preprocessing.py          # reproduktivan pipeline pripreme podataka
│   ├── model.py                  # ARHITEKTURA (Transformer enkoder od nule)
│   ├── train.py                  # trening, k-fold CV, MLflow, metrike, grafici
│   └── utils.py                  # seed, verzije, merenje vremena, grafici
├── scripts/
│   └── predict.py                # CLI za predikciju nad jednom recenzijom
├── results/
│   ├── figures/                  # grafici (learning curve, konfuziona matrica, ROC/PR...)
│   ├── tables/                   # comparison.csv (poređenje konfiguracija)
│   └── versions.json             # verzije biblioteka (reproduktivnost)
├── models/                       # sačuvani model + preprocessor + meta
└── mlruns/                       # MLflow evidencija eksperimenata
```

Arhitektura mreže je, prema zahtevu, u **zasebnom modulu** [`src/model.py`](src/model.py),
a svi koraci pripreme podataka su reproducibilno zapisani u
[`src/preprocessing.py`](src/preprocessing.py).

---

## 2. Instalacija

```bash
# 1) (preporučeno) virtuelno okruženje
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 2) PyTorch (CPU verzija)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) ostale zavisnosti
pip install -r requirements.txt
```

---

## 3. Pokretanje

**Eksploratorna analiza (EDA):**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Treniranje svih konfiguracija:**
```bash
# Brzi režim (podskup od 3000 recenzija, 3 folda, 3 epohe) – nekoliko minuta na CPU:
python -m src.train --quick

# Pun režim (podešava se u configs/experiments.yaml):
python -m src.train
```
Trening automatski:
- učita i pripremi podatke (preuzme IMDB pri prvom pokretanju),
- pokrene **5-fold kros-validaciju** za svih 6 konfiguracija,
- loguje sve u **MLflow** (`mlruns/`),
- sačuva grafike u `results/figures/` i tabelu poređenja u `results/tables/comparison.csv`,
- istrenira finalni (najbolji) model i evaluira ga na zvaničnom test skupu,
- sačuva model u `models/` (koristi se za predikciju i za seminarski rad).

**Pregled MLflow istorije:**
```bash
mlflow ui           # zatim otvoriti http://127.0.0.1:5000
```

**Brza predikcija (demo):**
```bash
python scripts/predict.py "This movie was fantastic, I really enjoyed it!"
```

---

## 4. Konfiguracije modela (pretraga hiperparametara)

Poredimo **6 konfiguracija** (zahtev: najmanje 5). Sve su definisane u
[`configs/experiments.yaml`](configs/experiments.yaml):

| Konfiguracija | d_model | glave | slojevi | dim_ff | dropout | poz. kodiranje | posebno |
|---|---|---|---|---|---|---|---|
| `baseline` | 64 | 2 | 1 | 128 | 0.1 | sinusoidno | osnovni model |
| `deeper_2_layers` | 64 | 2 | 2 | 128 | 0.1 | sinusoidno | dublja mreža |
| `wider_d128` | 128 | 4 | 1 | 256 | 0.1 | sinusoidno | širi model + GELU |
| `more_heads_4` | 64 | 4 | 1 | 128 | 0.1 | sinusoidno | više glava pažnje |
| `regularized_dropout03` | 64 | 2 | 1 | 128 | 0.3 | sinusoidno | jača regularizacija |
| `learned_positional` | 64 | 2 | 1 | 128 | 0.1 | naučeno | naučeno poz. kodiranje |

---

## 5. Rezultati

> Vrednosti su dobijene **5-fold kros-validacijom** na uravnoteženom podskupu od
> **8.000 recenzija** (6 epoha po fold-u, CPU). Kompletni logovi su u `mlruns/`,
> grafici u `results/figures/`.

**Poređenje konfiguracija (prosek ± std preko 5 foldova, sortirano po F1):**

| Konfiguracija | Accuracy | F1 | ROC-AUC | PR-AUC | Parametara | Veličina | Vreme/fold |
|---|---|---|---|---|---|---|---|
| `wider_d128` | 0.835 ± 0.009 | **0.837 ± 0.009** | 0.913 | 0.912 | 2.69 M | 10.37 MB | 283.5 s |
| `learned_positional` | 0.832 ± 0.008 | 0.835 ± 0.008 | 0.916 | 0.914 | 1.33 M | 5.06 MB | 120.9 s |
| `more_heads_4` | 0.835 ± 0.006 | 0.834 ± 0.011 | **0.920** | 0.919 | 1.31 M | 5.06 MB | 192.4 s |
| `regularized_dropout03` | 0.836 ± 0.010 | 0.832 ± 0.011 | 0.917 | 0.915 | 1.31 M | 5.06 MB | 121.1 s |
| `deeper_2_layers` | 0.837 ± 0.007 | 0.830 ± 0.008 | 0.920 | 0.917 | 1.35 M | 5.19 MB | 259.5 s |
| `baseline` | 0.835 ± 0.006 | 0.830 ± 0.008 | **0.921** | 0.920 | 1.31 M | 5.06 MB | 134.3 s |

Razlike među konfiguracijama su unutar standardne devijacije — za **finalni model**
izabran je `more_heads_4`: praktično isti F1 kao najveći model, ali sa **4× manje
parametara** i bržom inferencijom (bolji odnos kvaliteta i veličine).

**Finalni model** (`more_heads_4`, istreniran na 8.000 i evaluiran na zvaničnom
test skupu od 25.000 recenzija):

| Metrika | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Test skup | **0.823** | 0.832 | 0.809 | **0.820** | **0.904** | 0.898 |

> Osnovna linija (slučajno pogađanje) je 50%, pa model jasno uči obrasce iz teksta.

Glavni grafici (u `results/figures/`):
- `comparison_metrics.png` – poređenje F1 i ROC-AUC po konfiguraciji
- `comparison_time.png` – vreme treniranja po konfiguraciji
- `learning_curve_<naziv>.png` – kriva učenja (gubitak po epohi)
- `confusion_matrix_<naziv>.png` – konfuziona matrica
- `roc_<naziv>.png`, `pr_<naziv>.png` – ROC i PR krive
- `*_TEST.png` – evaluacija finalnog modela na zvaničnom test skupu

---

## 6. Reproduktivnost

- Svi random seed-ovi su fiksirani (`src/utils.py: set_seed`).
- Verzije biblioteka se beleže u `results/versions.json` i u MLflow.
- Pipeline pripreme podataka se čuva (`models/preprocessor.json`), pa
  trening, evaluacija i buduća aplikacija koriste **identičnu** pripremu.

---

## 7. Aplikacija i Docker (seminarski rad)

**Streamlit aplikacija** (korisnički interfejs za interakciju sa modelom):
```bash
streamlit run app/app.py
```
Aplikacija učitava sačuvani model (`models/best_model.pt`) i **isti**
preprocessing pipeline korišćen u treningu (`models/preprocessor.json`).
Korisnik unese recenziju (ili izabere primer), a aplikacija prikaže predikciju
(POZITIVNA/NEGATIVNA) sa verovatnoćom.

**Docker:**
```bash
docker build -t imdb-sentiment .
docker run -p 8501:8501 imdb-sentiment
# zatim otvoriti http://localhost:8501
```

## 8. Git tok rada

Razvoj je vođen kroz Git u više koraka (skelet → podaci → EDA → preprocessing →
arhitektura → trening/MLflow → eksperimenti i rezultati → dokumentacija), sa
posebnim eksperimentalnim branch-om za razvoj modela. Istorija commit-ova
prikazuje napredak, u skladu sa zahtevom zadatka.
