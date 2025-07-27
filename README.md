# 🌸 ML Classifier API - Con Modelo Iris

API para Machine Learning que incluye el clásico clasificador de flores Iris como "Hello World" del ML, más soporte para modelos de imágenes.

## 🚀 ¿Por qué empezar con Iris?

El dataset Iris es el "Hello World" del machine learning porque:

- **Simple**: Solo 4 medidas (largo/ancho de sépalos y pétalos)
- **Pequeño**: 150 muestras, 3 especies
- **Limpio**: Sin valores faltantes
- **Intuitivo**: Flores diferentes → tamaños diferentes
- **Rápido**: Entrena en segundos
- **Visualizable**: Puedes ver los resultados gráficamente

## 🐳 Deployment con Docker

### Opción 2: Desarrollo Local
```bash
# Validar configuración Docker
./validate-docker.sh

# Usar docker-compose
docker-compose up -d

# Ver logs
docker-compose logs -f
```

### Deployment a AWS ECR
```bash
# Configurar AWS CLI primero
aws configure

# Deploy automático
./deploy-ecr.sh

# Deploy con tag específico
./deploy-ecr.sh --tag v1.0.0
```

## 🛠️ Desarrollo Local (Sin Docker)

### 1. Configurar el entorno
```bash
./start.sh
```

### 2. Entrenar el modelo Iris
```bash
python train_iris_model.py
```

### 3. Probar la API
```bash
# En otra terminal
python test_api.py
```

## 🎯 Inicio Rápido

### 1. Configurar el entorno
```bash
./start.sh
```

### 2. Entrenar el modelo Iris
```bash
python train_iris_model.py
```

### 3. Probar la API
```bash
# En otra terminal
python test_api.py
```

## 🎭 Endpoints Principales

### 🌸 Clasificación Iris
```bash
curl -X POST "http://localhost:8000/predict/iris" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5, 
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

**Respuesta:**
```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 0.9876,
  "all_predictions": [0.9876, 0.0098, 0.0026],
  "input_data": {...},
  "timestamp": "2025-07-22T10:30:00"
}
```

## 🌺 Las 3 Especies de Iris

| Especie | Características |
|---------|----------------|
| **Setosa** | Pétalos muy pequeños (1.0-1.9 cm largo) |
| **Versicolor** | Tamaño medio (3.0-5.1 cm largo de pétalo) |
| **Virginica** | Pétalos grandes (4.5-6.9 cm largo) |

## 📁 Estructura del Proyecto

```
ml-santiago/api/
├── 🧠 main.py                    # API FastAPI principal
├── 🎓 train_iris_model.py        # Entrenamiento modelo Iris
├── 🧪 test_api.py               # Tests automatizados
├── 📋 requirements.txt          # Dependencias
├── 🚀 start.sh                 # Script de inicio
├── 📖 api_examples.md          # Ejemplos de uso
├── 🗂️ models/                  # Modelos entrenados
│   ├── iris_classifier.h5      # Modelo TensorFlow
│   ├── model_info.json         # Metadata del modelo
│   ├── test_examples.json      # Casos de prueba
│   └── iris_analysis.png       # Visualizaciones
└── 📚 README.md                # Este archivo
```

## 🧪 Casos de Prueba Incluidos

El script `train_iris_model.py` genera:

- ✅ **Modelo entrenado** (iris_classifier.h5)
- ✅ **Visualizaciones** (gráficos de análisis)
- ✅ **Datos de prueba** (ejemplos para cada especie)
- ✅ **Métricas** (precisión, matriz de confusión)

## 📊 Resultados Esperados

Con el modelo Iris deberías obtener:
- **Precisión**: >95%
- **Tiempo de entrenamiento**: <30 segundos
- **Predicciones**: Instantáneas

## 🌐 Documentación Interactiva

Una vez iniciado el servidor:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔄 Flujo de Trabajo ML

```mermaid
graph LR
    A[Datos Iris] --> B[Entrenar Modelo]
    B --> C[Guardar Modelo .h5]
    C --> D[Cargar en API]
    D --> E[Hacer Predicciones]
    E --> F[Obtener Resultados]
```

## 💡 Expandir a Otros Modelos

Una vez que entiendas Iris, puedes:

1. **Modelos de imágenes**: Usar endpoint `/predict/image`
2. **Otros datasets**: Modificar `train_iris_model.py`
3. **Modelos complejos**: Cambiar arquitectura de red neuronal

## 🚀 Comandos Útiles

```bash
# Ver estructura del proyecto
tree

# Entrenar modelo
python train_iris_model.py

# Iniciar API
uvicorn main:app --reload

# Probar API
python test_api.py

# Ver logs en tiempo real
tail -f api.log
```

## 📚 Aprender Más

- **Dataset Iris**: [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- **FastAPI**: [Documentación oficial](https://fastapi.tiangolo.com/)
- **TensorFlow**: [Guías oficiales](https://www.tensorflow.org/)

---

