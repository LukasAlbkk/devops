FROM python:3.11-slim

# Instala gcc para compilar pyfim
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# roda e finaliza (Job)
CMD ["python", "train.py"]
