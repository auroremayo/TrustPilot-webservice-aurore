FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge les ressources NLTK au build (évite le téléchargement à chaque démarrage)
RUN python -c "\
import nltk; \
[nltk.download(r, quiet=True) for r in ['punkt','stopwords','wordnet','punkt_tab']]"

# Code du frontend
COPY common ./common
COPY frontend ./frontend

# Les modèles sont montés via docker-compose (volume ./models:/app/models)

CMD ["streamlit", "run", "frontend/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
