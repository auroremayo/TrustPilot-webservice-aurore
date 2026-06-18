FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone le repo GitHub (auth via token au build, voir docker-compose.yml)
ARG GIT_USER
ARG GIT_TOKEN
RUN git clone https://${GIT_USER}:${GIT_TOKEN}@github.com/auroremayo/TrustPilot-webservice-aurore.git . \
    && rm -rf .git/hooks

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "80"]