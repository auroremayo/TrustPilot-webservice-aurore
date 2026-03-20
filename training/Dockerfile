FROM python:3.12

WORKDIR /app


 # Installer libgomp pour LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY /training/requirements_train.txt .

# Installer les dépendances
RUN pip install --no-cache-dir dvc[s3]
RUN pip install --no-cache-dir -r requirements_train.txt

# Copier l'application
COPY . .

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "training.app.main:app", "--host", "0.0.0.0", "--port", "8000"]