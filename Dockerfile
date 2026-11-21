# Single-container build: React frontend + FastAPI backend with the trained
# VGG-16 + GRU models, served together on one port.
#
#   docker build -t jewellery-analyzer .
#   docker run -p 8000:8000 jewellery-analyzer      ->  http://localhost:8000

# ---------- 1. Build the React frontend ----------
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV CI=false
RUN npm run build

# ---------- 2. Python runtime ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY --from=frontend /app/frontend/build frontend/build

WORKDIR /app/backend
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
