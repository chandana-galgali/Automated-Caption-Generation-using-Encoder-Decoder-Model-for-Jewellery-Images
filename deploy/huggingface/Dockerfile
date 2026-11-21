# Hugging Face Space Dockerfile.
# The Space only needs this file and README.md: it pulls the latest code and
# model weights from GitHub at build time, so the live demo stays in sync
# with the repository. Rebuild the Space (Settings -> Factory rebuild) after
# pushing changes to GitHub.

FROM node:20-alpine AS source
RUN apk add --no-cache git
RUN git clone --depth 1 https://github.com/chandana-galgali/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images.git /repo
WORKDIR /repo/frontend
ENV CI=false
RUN npm ci --no-audit --no-fund && npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --from=source /repo/backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY --from=source /repo/backend backend
COPY --from=source /repo/frontend/build frontend/build

WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
