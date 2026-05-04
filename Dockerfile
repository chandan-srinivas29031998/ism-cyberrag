# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git (required for some HF/Torch dependencies) and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
# CPU-only PyTorch (no torchvision/torchaudio needed, saves ~500MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY .env.example ./.env

# Pre-download embedding and reranking models during build
# so the Space starts instantly without downloading on first request
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
               SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); \
               CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# HF Spaces runs containers as user 1000
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
