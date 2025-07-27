# Dockerfile para ML Santiago API
# Optimizado para AWS ECR y producción - Imagen ligera
# Soporta multi-arquitectura con foco en linux/amd64 para ECR

# ================================
# Stage 1: Builder (dependencias de compilación)
# ================================
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Variables de entorno para el build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias de compilación solo para el build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo temporal
WORKDIR /build

# Copiar requirements y crear wheels optimizados
COPY requirements.txt .

# Crear un requirements optimizado para producción (sin dependencias de desarrollo)
RUN echo "fastapi==0.104.1" > requirements-prod.txt && \
    echo "uvicorn[standard]==0.24.0" >> requirements-prod.txt && \
    echo "tensorflow-cpu==2.15.0" >> requirements-prod.txt && \
    echo "numpy==1.24.3" >> requirements-prod.txt && \
    echo "Pillow==10.1.0" >> requirements-prod.txt && \
    echo "python-multipart==0.0.6" >> requirements-prod.txt && \
    echo "python-dotenv==1.0.0" >> requirements-prod.txt && \
    echo "scikit-learn==1.3.2" >> requirements-prod.txt && \
    echo "pandas==2.1.4" >> requirements-prod.txt

# Instalar dependencias en un directorio separado
RUN pip install --prefix=/build/pip-install -r requirements-prod.txt

# ================================
# Stage 2: Runtime (imagen final optimizada)
# ================================
FROM --platform=linux/amd64 python:3.11-slim AS base

# Metadatos
LABEL maintainer="ML Santiago Team" \
      description="API de Machine Learning para clasificación Iris - Optimizada" \
      version="1.0.0"

# Variables de entorno optimizadas
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_PORT=8000 \
    APP_HOST=0.0.0.0 \
    PYTHONPATH=/app

# Instalar solo runtime dependencies mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar dependencias de Python desde el builder
COPY --from=builder /build/pip-install /usr/local

# Establecer directorio de trabajo
WORKDIR /app

# Copiar solo los archivos necesarios para runtime
COPY main.py train_iris_model.py ./
COPY models/ ./models/

# Crear directorio para modelos si no existe y ajustar permisos
RUN mkdir -p models && \
    chown -R appuser:appuser /app

# Cambiar al usuario no root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check optimizado
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ================================
# Stage 3: Development
# ================================
FROM base AS development

# Cambiar a root temporalmente para instalar dependencias de desarrollo
USER root

# Instalar herramientas de desarrollo mínimas
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    httpx==0.25.2

# Volver al usuario no root
USER appuser

# Variables de entorno para desarrollo
ENV ENVIRONMENT=development

# Comando para desarrollo con hot reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Stage 4: Production (optimizada)
# ================================
FROM base AS production

# Variables de entorno para producción
ENV ENVIRONMENT=production

# Comando optimizado para producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
