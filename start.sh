#!/bin/bash

# Script para iniciar la API en modo desarrollo

echo "🚀 Iniciando ML Santiago API..."

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -r requirements.txt


# Iniciar el servidor
echo "🌟 Iniciando servidor FastAPI..."
echo "📖 Documentación disponible en: http://localhost:8000/docs"
echo "🔗 API disponible en: http://localhost:8000"
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000