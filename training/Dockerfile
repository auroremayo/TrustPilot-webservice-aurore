# Utilise bien une image de base python (ex: python:3.9-slim) si elle n'est pas déjà définie plus haut
FROM python:3.12 

WORKDIR /app

# 1. Copier d'abord le fichier de dépendances
# Suppression du "/" devant training
COPY training/requirements_train.txt .

# 2. Installer les dépendances
# On peut grouper les pip install pour réduire le nombre de couches
RUN pip install --no-cache-dir dvc[s3] && \
    pip install --no-cache-dir -r requirements_train.txt

# 3. Copier les dossiers de code
# Suppression des "/" devant les sources
COPY common ./common
COPY training/app ./training/app

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "training.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
