# Veštačka inteligencija sa primenama
## Domaći zadatak: Analiza sentimenta filmskih recenzija (IMDB) primenom Transformer mreže

**Tip problema:** binarna klasifikacija teksta (pozitivan / negativan sentiment)
**Arhitektura:** Transformer enkoder (mehanizam pažnje)
**Alati:** Python, PyTorch, scikit-learn, MLflow, Pandas/NumPy

---

## 1. Uvod

Analiza sentimenta je zadatak klasifikacije teksta u kome se utvrđuje emocionalni
stav izražen u tekstu. U ovom radu rešava se konkretan problem: na osnovu
filmske recenzije potrebno je odrediti da li je ona **pozitivna** ili
**negativna**. Reč je o *binarnoj klasifikaciji*, jednom od osnovnih problema
mašinskog učenja.

Za rešavanje je izabrana **Transformer** arhitektura, jer se ona zasniva na
**mehanizmu pažnje (attention)** koji je pokazao izuzetne rezultate u obradi
prirodnog jezika. Za razliku od rekurentnih mreža (RNN, LSTM) koje sekvencu
obrađuju reč po reč i imaju problem sa dugoročnim zavisnostima i sporim
treniranjem, Transformer sve reči obrađuje paralelno i svakoj reči omogućava da
direktno „pogleda" sve ostale reči u recenziji. Time se efikasnije uočava
kontekst koji određuje sentiment (npr. reč „not" ispred „good").

Osnovna ideja pristupa je sledeća: svaka reč se prevodi u numerički vektor
(embedding), dodaje se informacija o poziciji reči, a zatim se kroz slojeve
pažnje gradi reprezentacija cele recenzije, na osnovu koje linearni klasifikator
donosi odluku.

---

## 2. Skup podataka i eksploratorna analiza

### 2.1. Opis skupa

Korišćen je poznati **IMDB** skup podataka („Large Movie Review Dataset") koji
sadrži **50.000** filmskih recenzija na engleskom jeziku, podeljenih na
**25.000** za trening i **25.000** za test. Skup je **savršeno uravnotežen** –
tačno polovina recenzija je pozitivna (labela **1**), a polovina negativna
(labela **0**).

Korišćena je već tokenizovana verzija skupa, u kojoj je svaka recenzija
predstavljena kao **niz celih brojeva**. Svaki broj je **rang reči** po
učestalosti u celom skupu (1 = najčešća reč „the", 17 = „movie"). Time je korak
tokenizacije obavljen na standardan i reproduktivan način. Reči se lako vraćaju u
tekst pomoću inverznog rečnika (rang → reč), što je iskorišćeno u analizi.

### 2.2. Eksploratorna analiza (EDA)

Kompletna analiza nalazi se u notebooku [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).
Ključni zaključci:

- **Provera kvaliteta podataka:** ne postoje prazne recenzije ni nedostajuće
  labele, sve labele pripadaju skupu {0, 1}. Skup je, dakle, kompletan i ne
  zahteva imputaciju nedostajućih vrednosti.
- **Balans klasa:** odnos pozitivnih i negativnih recenzija je 12.500 : 12.500.
  Zbog uravnoteženosti, *tačnost (accuracy)* je pouzdana metrika, a osnovna
  linija slučajnog pogađanja iznosi 50%.
- **Distribucija dužine recenzija:** dužine su vrlo promenljive (prosek 237.7,
  medijana 177 reči), uz dugačak „rep" veoma dugih recenzija (do 2493 reči). Na
  osnovu histograma i percentila odabrana je dužina sekvence **`max_len = 200`**:
  ona u potpunosti obuhvata oko 58% recenzija, dok se duže skraćuju na prvih 200
  reči, što je sasvim dovoljno za procenu sentimenta uz razuman utrošak resursa.
- **Odnos dužine i sentimenta:** point-biserial korelacija između dužine i klase
  je vrlo slaba (r ≈ 0.016), pa dužina recenzije **nije** dobar samostalni
  prediktor sentimenta.
- **Analiza vokabulara i OOV:** ograničavanjem vokabulara na **`num_words =
  20000`** najčešćih reči udeo *Out Of Vocabulary* tokena je svega **2.86%**,
  tj. zadržava se skoro sav sadržaj uz znatno manju embedding tabelu.
- **Word cloud** pozitivnih i negativnih recenzija pokazuje jasnu razliku u
  rečniku (pozitivne: „great", „love", „best"; negativne: „bad", „worst",
  „boring"), što opravdava pristup zasnovan na rečima.

Odgovarajući grafici (balans klasa, histogram dužina, najčešće reči, oblaci
reči) sačuvani su u `results/figures/eda_*.png`.

---

## 3. Priprema podataka (preprocessing pipeline)

Priprema podataka je enkapsulirana u klasi `SentimentPreprocessor`
([`src/preprocessing.py`](../src/preprocessing.py)) kako bi bila **reproduktivna**
i kako bi trening, evaluacija i buduća aplikacija koristili **identičan** postupak.
Koraci su:

1. **Mapiranje vokabulara:** zadržava se `num_words = 20000` najčešćih reči.
   Uvode se dva rezervisana tokena:
   - `PAD = 0` – popuna kraćih recenzija,
   - `OOV = 1` – sve reči van vokabulara.
   Stvarne reči dobijaju indekse od 2 naviše.
2. **Skraćivanje/dopuna (padding/truncation):** sve sekvence se svode na
   `max_len = 200`. Duže se skraćuju, kraće dopunjuju PAD tokenom.
3. **Kodiranje sirovog teksta (za aplikaciju):** za tekst koji unese korisnik
   radi se standardna tokenizacija (mala slova, izdvajanje reči), pa mapiranje
   reč → rang → indeks, po istim pravilima kao u treningu.

Pipeline se čuva na disk (`models/preprocessor.json`) i može se ponovo učitati,
čime je zagarantovana konzistentnost pripreme podataka.

---

## 4. Arhitektura modela

Arhitektura je implementirana „od nule" u PyTorch-u, u zasebnom modulu
[`src/model.py`](../src/model.py), prema gradivu sa predavanja o Transformerima.
Tok podataka kroz mrežu je:

```
Indeksi reči
   → Embedding (reč → vektor, skaliran sa √d_model)
   → Poziciono kodiranje (sinusoidno ili naučeno)
   → N × Transformer enkoder blok:
         Multi-Head Self-Attention → Add & Norm
         Feed-Forward mreža (FFN)   → Add & Norm
   → Maskirano usrednjavanje po tokenima (bez PAD)
   → Linearni klasifikator → 2 logita (negativan / pozitivan)
```

### 4.1. Embedding i poziciono kodiranje

**Embedding** sloj pretvara indeks reči u gust vektor dimenzije `d_model`. Cilj
je da reči sličnog značenja imaju bliske vektore. Pošto attention obrađuje reči
paralelno i **ne poznaje redosled**, dodaje se **poziciono kodiranje**. Korišćene
su dve varijante: **sinusoidno** (determinističko, na bazi sin/cos funkcija
različitih frekvencija) i **naučeno** (poseban vektor po poziciji).

### 4.2. Multi-Head Self-Attention

Srž modela je **skalirani dot-product attention**:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

Za svaku reč se računaju tri vektora: **Query (Q)** – šta reč traži, **Key (K)** –
obeležja reči za poređenje, i **Value (V)** – informacija koja se preuzima.
Sličnost upita i ključeva (skalarni proizvod) skalira se sa `√d_k` da bi se
sprečile prevelike vrednosti pre `softmax`-a (stabilnost gradijenata), a zatim se
vrednosti kombinuju dobijenim težinama pažnje.

**Multi-Head** pristup pokreće više „glava" pažnje paralelno; svaka uči drugačiju
vrstu odnosa među rečima. Izlazi glava se konkateniraju i propuštaju kroz izlaznu
linearnu projekciju.

### 4.3. Feed-Forward mreža, rezidualne veze i LayerNorm

Nakon pažnje, svaki token nezavisno prolazi kroz **Feed-Forward mrežu**
(Linear → ReLU/GELU → Linear) koja uvodi nelinearnost. Oko oba podsloja (pažnja i
FFN) postavljene su **rezidualne veze** (`x + Sloj(x)`) i **Layer Normalization**
(„Add & Norm"). Rezidualne veze olakšavaju protok gradijenata kroz duboke mreže,
a LayerNorm stabilizuje treniranje.

### 4.4. Agregacija i klasifikacija

Enkoder daje po jedan vektor za svaku reč. Za klasifikaciju se radi **maskirano
usrednjavanje (mean pooling)** – prosek vektora svih stvarnih reči (PAD pozicije
se preskaču). Dobijeni vektor cele recenzije prolazi kroz **linearni sloj** koji
daje dva izlaza (logita), a `softmax` ih pretvara u verovatnoće klasa.

Napomena: koristi se **samo enkoder** deo Transformera, jer je za klasifikaciju
dovoljno „razumeti" ulaz; dekoder (sa maskiranom i encoder-decoder pažnjom) služi
za generisanje sekvenci (npr. prevođenje).

---

## 5. Treniranje i validacija

### 5.1. Postupak treniranja

- **Funkcija gubitka:** Cross-Entropy Loss (standard za klasifikaciju).
- **Optimizator:** Adam.
- **Kros-validacija:** `StratifiedKFold` sa **5 foldova** (čuva odnos klasa u
  svakom fold-u). Model se trenira na 4, validira na 1 fold-u, i to 5 puta.
  Prijavljuje se **prosek ± standardna devijacija** metrika.
- **Reproduktivnost:** fiksirani random seed-ovi; verzije biblioteka i svi
  hiperparametri se automatski beleže u MLflow i u `results/versions.json`.

### 5.2. Pretraga hiperparametara

Poređeno je **6 konfiguracija** (variran je broj slojeva, dimenzija modela, broj
glava pažnje, dropout, regularizacija i tip pozicionog kodiranja):

| Konfiguracija | d_model | glave | slojevi | dim_ff | dropout | poz. kodiranje |
|---|---|---|---|---|---|---|
| `baseline` | 64 | 2 | 1 | 128 | 0.1 | sinusoidno |
| `deeper_2_layers` | 64 | 2 | 2 | 128 | 0.1 | sinusoidno |
| `wider_d128` | 128 | 4 | 1 | 256 | 0.1 | sinusoidno (GELU) |
| `more_heads_4` | 64 | 4 | 1 | 128 | 0.1 | sinusoidno |
| `regularized_dropout03` | 64 | 2 | 1 | 128 | 0.3 | sinusoidno |
| `learned_positional` | 64 | 2 | 1 | 128 | 0.1 | naučeno |

### 5.3. Praćenje eksperimenata (MLflow)

Svaki eksperiment se automatski loguje u MLflow (`mlruns/`): hiperparametri,
**gubitak po epohi**, **metrike po fold-u**, prosečne vrednosti i artefakti
(grafici). Istorija se pregleda komandom `mlflow ui`. Time je omogućeno lako
poređenje svih verzija modela na jednom mestu.

---

## 6. Rezultati i poređenje modela

Rezultati su dobijeni **5-fold kros-validacijom** na uravnoteženom podskupu od
**8.000 recenzija** (6 epoha po fold-u), radi izvodljivosti na običnom računaru
(CPU). Kompletni logovi i grafici su u `mlruns/` i `results/`.

### 6.1. Tabela poređenja (prosek preko foldova)

| Konfiguracija | Accuracy | F1 | ROC-AUC | PR-AUC | Parametara | Vreme/fold |
|---|---|---|---|---|---|---|
| `wider_d128` | 0.835 ± 0.009 | **0.837 ± 0.009** | 0.913 | 0.912 | 2.69 M | 283.5 s |
| `learned_positional` | 0.832 ± 0.008 | 0.835 ± 0.008 | 0.916 | 0.914 | 1.33 M | 120.9 s |
| `more_heads_4` | 0.835 ± 0.006 | 0.834 ± 0.011 | **0.920** | 0.919 | 1.31 M | 192.4 s |
| `regularized_dropout03` | 0.836 ± 0.010 | 0.832 ± 0.011 | 0.917 | 0.915 | 1.31 M | 121.1 s |
| `deeper_2_layers` | 0.837 ± 0.007 | 0.830 ± 0.008 | 0.920 | 0.917 | 1.35 M | 259.5 s |
| `baseline` | 0.835 ± 0.006 | 0.830 ± 0.008 | **0.921** | 0.920 | 1.31 M | 134.3 s |

### 6.2. Analiza rezultata

- **Metrike:** za svaku konfiguraciju prikazani su accuracy, precision, recall,
  F1, ROC-AUC i PR-AUC. Kod uravnoteženih klasa F1 i ROC-AUC daju najpotpuniju
  sliku kvaliteta.
- **Krive učenja** (`results/figures/learning_curve_*.png`): prate gubitak po
  epohi za trening i validaciju; koriste se za uočavanje pre/potprilagođavanja.
- **Konfuziona matrica** (`results/figures/confusion_matrix_*.png`): pokazuje
  raspodelu tačnih i pogrešnih predikcija po klasama.
- **ROC i PR krive** (`results/figures/roc_*.png`, `pr_*.png`): kvalitet
  razdvajanja klasa nezavisno od izbora praga.
- **Efikasnost:** za svaku konfiguraciju mereni su vreme treniranja po fold-u,
  vreme inferencije po uzorku, broj parametara i veličina modela (u MB).

**Zapažanja:** Sve konfiguracije daju ROC-AUC 0.91–0.92, što znači da model
dobro razdvaja klase, a razlike u F1 (0.830–0.837) su unutar standardne
devijacije. Najveći model (`wider_d128`) ima nominalno najviši F1, ali sa
duplo više parametara (2.69 M) i ubedljivo najdužim treniranjem (283 s po
fold-u) — dobitak ne opravdava cenu. `more_heads_4` postiže praktično isti F1
sa 4× manje parametara: više glava pažnje omogućava modelu da paralelno uči
različite vrste odnosa među rečima, bez povećanja ukupne veličine. Jača
regularizacija (`regularized_dropout03`) daje stabilne rezultate, dok dublja
mreža (`deeper_2_layers`) ne donosi poboljšanje na ovoj količini podataka.

Za **finalni model** zato je izabran `more_heads_4` — najbolji odnos kvaliteta,
veličine i brzine inferencije.

**Finalni model** (`more_heads_4`, istreniran na svih 8.000 primera) na
zvaničnom test skupu od **25.000 recenzija** postiže: Accuracy **0.823**,
Precision 0.832, Recall 0.809, F1 **0.820**, ROC-AUC **0.904**, PR-AUC 0.898.
Bliskost test metrika i kros-validacionih procena potvrđuje da model dobro
generalizuje i da procena nije bila optimistična.

### 6.3. Finalni model

Konfiguracija sa najboljim prosečnim F1 skorom istrenirana je na celom
(podskupu) trening skupa i evaluirana na **zvaničnom test skupu** od 25.000
recenzija. Njena konfuziona matrica i ROC/PR krive na test skupu nalaze se u
`results/figures/*_TEST.png`, a model je sačuvan u `models/best_model.pt` (koristi
se za predikciju i za seminarski rad).

---

## 7. Reproduktivnost i organizacija koda

- **Arhitektura** je u zasebnom modulu (`src/model.py`).
- **Priprema podataka** je reproduktivan pipeline (`src/preprocessing.py`) koji se
  čuva i ponovo učitava.
- **Seed-ovi** su fiksirani; **verzije biblioteka** se beleže.
- **MLflow** čuva celu istoriju eksperimenata.
- **Git** istorija prikazuje razvoj kroz jasne korake (skelet → podaci → EDA →
  arhitektura → trening → rezultati → dokumentacija).

---

## 8. Zaključak

U radu je uspešno implementiran kompletan tok za analizu sentimenta filmskih
recenzija primenom Transformer enkodera napisanog „od nule" u PyTorch-u. Sprovedena
je eksploratorna analiza, reproduktivna priprema podataka, treniranje sa
5-fold kros-validacijom i poređenje 6 konfiguracija modela, uz automatsko praćenje
eksperimenata u MLflow-u i vizuelizaciju rezultata (krive učenja, konfuzione
matrice, ROC/PR krive). Model ubedljivo nadmašuje osnovnu liniju (50%), čime je
potvrđeno da Transformer uspešno uči obrasce koji određuju sentiment.

Kao naredni korak (seminarski rad), model će biti proširen u kompletnu aplikaciju:
korisnički interfejs (Streamlit) za unos recenzije i prikaz predikcije, uz
pakovanje u Docker kontejner.

---

## Literatura

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Maas, A. et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL
   (IMDB Large Movie Review Dataset).
3. Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern
   Approach*, 4th ed. Pearson.
4. Materijali sa predavanja: „Transformeri i mehanizam pažnje".
