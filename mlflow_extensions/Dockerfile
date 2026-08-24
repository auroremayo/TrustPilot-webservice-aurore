FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (can be overridden by a bind-mount in compose)
COPY . /app

CMD ["uvicorn", "webhook_handler:app", "--host", "0.0.0.0", "--port", "8000"]
