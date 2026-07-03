# ============================================================================
# Dockerfile za seminarski rad – IMDB analiza sentimenta (Streamlit + PyTorch)
# ----------------------------------------------------------------------------
# Build:  docker build -t imdb-sentiment .
# Run:    docker run -p 8501:8501 imdb-sentiment
# Zatim otvoriti: http://localhost:8501
# ============================================================================

FROM python:3.12-slim

WORKDIR /app

# 1) Zavisnosti (keširan sloj – menja se retko)
#    PyTorch CPU verzija sa zvaničnog indeksa (manja slika, bez CUDA).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy pandas scikit-learn streamlit

# 2) Kod i artefakti modela
COPY src/ ./src/
COPY app/ ./app/
COPY models/best_model.pt models/best_model_meta.json models/preprocessor.json ./models/
COPY data/imdb_word_index.json ./data/

# 3) Streamlit port
EXPOSE 8501

# Healthcheck – Streamlit ima ugrađeni endpoint
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 4) Pokretanje aplikacije
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
