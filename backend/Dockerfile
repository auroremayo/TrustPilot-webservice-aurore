FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet
# Code de l'application
COPY common ./common
COPY backend/app ./backend/app


# Les modèles et données sont montés via docker-compose (volumes)

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "80"]
