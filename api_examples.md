# Comandos para probar la API ML Classifier

## 🌸 Modelo Iris (Recomendado para empezar)

### 1. Verificar estado de la API
```bash
curl http://localhost:8000/
```

### 2. Verificar salud del sistema
```bash
curl http://localhost:8000/health
```

### 3. Clasificar flor Iris - Setosa
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

### 4. Clasificar flor Iris - Versicolor
```bash
curl -X POST "http://localhost:8000/predict/iris" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 6.2,
       "sepal_width": 2.9,
       "petal_length": 4.3,
       "petal_width": 1.3
     }'
```

### 5. Clasificar flor Iris - Virginica
```bash
curl -X POST "http://localhost:8000/predict/iris" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 7.2,
       "sepal_width": 3.0,
       "petal_length": 5.8,
       "petal_width": 1.6
     }'
```

## 🖼️ Modelo de Imágenes (Futuro)

### Subir imagen para clasificación
```bash
curl -X POST "http://localhost:8000/predict/image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@imagen.jpg"
```

### Endpoint legacy (compatible con versión anterior)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@imagen.jpg"
```

## 🔄 Reentrenamiento

### Reentrenar modelo
```bash
curl -X POST "http://localhost:8000/retrain" \
     -H "accept: application/json"
```

## 📊 Respuestas Esperadas

### Respuesta de /health:
```json
{
  "status": "healthy",
  "model_ready": true,
  "model_type": "iris_classifier",
  "model_info": {
    "model_type": "iris_classifier",
    "classes": ["setosa", "versicolor", "virginica"],
    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "input_shape": [4],
    "output_classes": 3
  },
  "timestamp": "2025-07-22T10:30:00.123456"
}
```

### Respuesta de /predict/iris:
```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 0.9876,
  "all_predictions": [0.9876, 0.0098, 0.0026],
  "input_data": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  "timestamp": "2025-07-22T10:30:00.123456"
}
```

## 🚀 Pasos para empezar

### 1. Entrenar el modelo Iris (primera vez)
```bash
python train_iris_model.py
```

### 2. Iniciar la API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Probar la API
```bash
python test_api.py
```

## 🌐 Para uso en producción

Reemplaza `localhost:8000` con tu IP pública:

```bash
# Verificar estado
curl http://TU_IP_PUBLICA:8000/health

# Clasificar Iris
curl -X POST "http://TU_IP_PUBLICA:8000/predict/iris" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 💡 Datos de ejemplo para probar

### Setosa (pétalos pequeños)
- sepal_length: 4.3-5.8, sepal_width: 2.3-4.4
- petal_length: 1.0-1.9, petal_width: 0.1-0.6

### Versicolor (tamaño medio)  
- sepal_length: 4.9-7.0, sepal_width: 2.0-3.4
- petal_length: 3.0-5.1, petal_width: 1.0-1.8

### Virginica (pétalos grandes)
- sepal_length: 4.9-7.9, sepal_width: 2.2-3.8  
- petal_length: 4.5-6.9, petal_width: 1.4-2.5
