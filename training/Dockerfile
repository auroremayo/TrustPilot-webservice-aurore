FROM python:3.12-slim

# ── 1. Installation des paquets système nécessaires (git pour DVC, libgomp1 pour LightGBM)
RUN apt-get update && apt-get install -y git libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 2. Copier et installer les dépendances Python
COPY training/requirements.txt ./training/requirements.txt
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache-dir -r training/requirements.txt

# ── 3. Copier le code source et la configuration DVC
COPY common ./common
COPY training/app ./training/app
COPY .dvc ./.dvc
COPY .dvcignore ./.dvcignore

# ── 4. Initialisation Git pour DVC
RUN git init

# ── 5. Lancer l'API
CMD ["uvicorn", "training.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
