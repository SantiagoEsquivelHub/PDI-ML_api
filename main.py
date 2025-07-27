from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime

app = FastAPI(title="ML Image Classifier API", version="1.0.0")

# Variables globales
model = None
model_info = None
MODEL_PATH = "models/iris_classifier.h5"  # Cambiar para desarrollo local
MODEL_INFO_PATH = "models/model_info.json"

# Esquemas Pydantic para el modelo Iris
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float  
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    predicted_class: int
    predicted_species: str
    confidence: float
    all_predictions: List[float]
    input_data: dict
    timestamp: str

def load_model():
    """Cargar modelo entrenado"""
    global model, model_info
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Modelo cargado desde: {MODEL_PATH}")
            
            # Cargar información del modelo si existe
            if os.path.exists(MODEL_INFO_PATH):
                with open(MODEL_INFO_PATH, 'r') as f:
                    model_info = json.load(f)
                print(f"Información del modelo cargada: {model_info['model_type']}")
            
            return True
        else:
            print(f"Modelo no encontrado en: {MODEL_PATH}")
            print("💡 Ejecuta 'python train_iris_model.py' para entrenar el modelo")
            return False
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return False

def preprocess_image(image_bytes):
    """Preprocesar imagen para predicción"""
    try:
        # Abrir imagen
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar
        image = image.resize((128, 128))
        
        # Convertir a array numpy
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")

@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la aplicación"""
    success = load_model()
    if not success:
        print("Advertencia: No se pudo cargar el modelo. Entrenar primero.")

@app.get("/")
async def root():
    model_type = model_info.get("model_type", "unknown") if model_info else "unknown"
    return {
        "message": "ML Classifier API",
        "status": "active",
        "model_loaded": model is not None,
        "model_type": model_type,
        "endpoints": {
            "health": "/health",
            "predict_iris": "/predict/iris",
            "predict_image": "/predict/image",
            "retrain": "/retrain"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    model_ready = model is not None
    model_type = model_info.get("model_type", "unknown") if model_info else "unknown"
    
    return {
        "status": "healthy",
        "model_ready": model_ready,
        "model_type": model_type,
        "model_info": model_info if model_info else None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/iris", response_model=IrisResponse)
async def predict_iris(iris_data: IrisInput):
    """Clasificar especie de flor Iris basado en medidas de sépalos y pétalos"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Ejecutar 'python train_iris_model.py' para entrenar."
        )
    
    if model_info is None or model_info.get("model_type") != "iris_classifier":
        raise HTTPException(
            status_code=400,
            detail="El modelo cargado no es un clasificador Iris."
        )
    
    try:
        # Preparar datos de entrada
        input_array = np.array([[
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width
        ]])
        
        # Hacer predicción
        predictions = model.predict(input_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Obtener nombre de la especie
        species_names = model_info.get("classes", ["class_0", "class_1", "class_2"])
        predicted_species = species_names[predicted_class]
        
        # Preparar respuesta
        result = IrisResponse(
            predicted_class=predicted_class,
            predicted_species=predicted_species,
            confidence=confidence,
            all_predictions=predictions[0].tolist(),
            input_data={
                "sepal_length": iris_data.sepal_length,
                "sepal_width": iris_data.sepal_width,
                "petal_length": iris_data.petal_length,
                "petal_width": iris_data.petal_width
            },
            timestamp=datetime.now().isoformat()
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Clasificar imagen subida (endpoint original)"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Ejecutar entrenamiento primero."
        )
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="El archivo debe ser una imagen"
        )
    
    try:
        # Leer imagen
        image_bytes = await file.read()
        
        # Preprocesar
        processed_image = preprocess_image(image_bytes)
        
        # Hacer predicción
        predictions = model.predict(processed_image)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Preparar respuesta
        result = {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": predictions[0].tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

# Mantener endpoint /predict para compatibilidad
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """Endpoint legacy - redirige a predict_image"""
    return await predict_image(file)

@app.post("/retrain")
async def retrain_model():
    """Endpoint para reentrenar el modelo"""
    try:
        # Aquí ejecutarías el script de entrenamiento
        # Por ahora solo simulamos
        return {
            "message": "Reentrenamiento iniciado",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en reentrenamiento: {e}")
