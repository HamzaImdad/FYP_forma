# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend

WORKDIR /forma
COPY app/web/package.json app/web/package-lock.json ./app/web/
WORKDIR /forma/app/web
RUN npm ci

WORKDIR /forma
COPY app/web ./app/web
WORKDIR /forma/app/web
RUN npm run build
# Build output lands at /forma/app/static/dist (per vite.config.ts)


# Stage 2: Python runtime
FROM python:3.10-slim-bookworm

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    libx11-6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bring in the built React SPA from the frontend stage
COPY --from=frontend /forma/app/static/dist ./app/static/dist

ENV PORT=5000
ENV HOST=0.0.0.0

CMD ["python", "app/server.py"]
