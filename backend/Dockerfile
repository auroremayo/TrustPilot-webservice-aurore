FROM python:3.12-slim

# Dépendances système
RUN apt-get update && apt-get install -y libgomp1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Installation de uv
RUN pip install --no-cache-dir uv

# 2. Installation des dépendances Python
COPY backend/requirements.txt ./backend/requirements.txt
RUN uv pip install --system --no-cache-dir -r backend/requirements.txt

# 3. Installation des ressources NLTK
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# 4. Copie du reste du code et du dossier .dvc
COPY . .

# 5. Initialisation git (nécessaire pour que DVC fonctionne dans le conteneur)
RUN git init


CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "80"]