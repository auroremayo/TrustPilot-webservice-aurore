# Image Airflow customisée pour SentimentAI
# Base : apache/airflow 2.9 (Python 3.12)
FROM apache/airflow:2.9.3-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Dépendances Python nécessaires aux tâches du DAG
# (requests est déjà inclus dans airflow, listés ici pour la clarté)
RUN pip install --no-cache-dir \
    requests==2.32.5 \
    apache-airflow-providers-http==4.10.1
