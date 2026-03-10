FROM python:3.12-slim

# Install system-level build tools needed by some TF wheels / h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Copy model & pre-processor files (large – consider a volume mount in prod)
COPY *.keras *.h5 *.pkl ./

ENV FLASK_ENV=production
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
