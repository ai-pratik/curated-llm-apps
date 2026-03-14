FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI (8000) and Gradio (7860)
EXPOSE 8000 7860

# Default: start API server.
# Override CMD to start Gradio separately if needed.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
