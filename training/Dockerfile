FROM python:3.9-slim

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements_train.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements_train.txt

# Copier l'application
COPY app/ ./app/

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]