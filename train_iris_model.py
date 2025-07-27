#!/usr/bin/env python3
"""
Script para entrenar un modelo simple de clasificación Iris
Este es el "Hello World" del Machine Learning
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_iris_data():
    """Cargar y explorar el dataset Iris"""
    print("🌸 Cargando dataset Iris...")
    
    # Cargar datos
    iris = datasets.load_iris()
    X = iris.data  # características: [sepal_length, sepal_width, petal_length, petal_width]
    y = iris.target  # especies: [0=setosa, 1=versicolor, 2=virginica]
    
    # Nombres de las características y especies
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🏷️  Clases: {target_names}")
    print(f"📏 Características: {feature_names}")
    
    # Crear DataFrame para mejor visualización
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    print("\n📋 Primeras 5 muestras:")
    print(df.head())
    
    print("\n📈 Estadísticas básicas:")
    print(df.describe())
    
    return X, y, feature_names, target_names, df

def create_visualizations(df, save_path="models/"):
    """Crear visualizaciones del dataset"""
    print("\n📊 Creando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribución de especies
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    df['species'].value_counts().plot(kind='bar')
    plt.title('Distribución de Especies')
    plt.ylabel('Cantidad')
    
    # 2. Scatter plot: Sepal
    plt.subplot(2, 2, 2)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], 
                   label=species, alpha=0.7)
    plt.xlabel('Largo del Sépalo (cm)')
    plt.ylabel('Ancho del Sépalo (cm)')
    plt.title('Sépalos por Especie')
    plt.legend()
    
    # 3. Scatter plot: Petal
    plt.subplot(2, 2, 3)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], 
                   label=species, alpha=0.7)
    plt.xlabel('Largo del Pétalo (cm)')
    plt.ylabel('Ancho del Pétalo (cm)')
    plt.title('Pétalos por Especie')
    plt.legend()
    
    # 4. Boxplot de todas las características
    plt.subplot(2, 2, 4)
    df_melted = df.melt(id_vars=['species'], var_name='characteristic', value_name='value')
    sns.boxplot(data=df_melted, x='characteristic', y='value', hue='species')
    plt.xticks(rotation=45)
    plt.title('Distribución de Características')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}iris_analysis.png", dpi=300, bbox_inches='tight')
    print(f"📊 Gráficos guardados en: {save_path}iris_analysis.png")
    plt.close()

def train_sklearn_model(X, y, target_names):
    """Entrenar modelo con scikit-learn"""
    print("\n🔬 Entrenando modelo con scikit-learn...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Datos de entrenamiento: {X_train.shape}")
    print(f"📊 Datos de prueba: {X_test.shape}")
    
    # Entrenar modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Precisión del modelo: {accuracy:.2%}")
    
    # Reporte detallado
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔢 Matriz de confusión:")
    print(cm)
    
    return model, accuracy

def train_tensorflow_model(X, y, target_names):
    """Entrenar modelo con TensorFlow/Keras para compatibilidad con la API"""
    print("\n🧠 Entrenando modelo con TensorFlow...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar datos
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Crear modelo simple de red neuronal
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(3, activation='softmax')  # 3 clases
    ])
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🏗️  Arquitectura del modelo:")
    model.summary()
    
    # Entrenar modelo
    print("\n🏃‍♂️ Entrenando...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    # Evaluar modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Precisión en datos de prueba: {test_accuracy:.2%}")
    
    # Predicciones de ejemplo
    predictions = model.predict(X_test[:5])
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\n🔮 Ejemplos de predicciones:")
    for i in range(5):
        print(f"Muestra {i+1}:")
        print(f"  Real: {target_names[y_test[i]]}")
        print(f"  Predicho: {target_names[predicted_classes[i]]}")
        print(f"  Confianza: {np.max(predictions[i]):.2%}")
        print()
    
    return model, test_accuracy, history

def save_models(sklearn_model, tf_model, save_path="models/"):
    """Guardar ambos modelos"""
    print("\n💾 Guardando modelos...")
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    # Guardar modelo scikit-learn
    with open(f"{save_path}iris_sklearn_model.pkl", 'wb') as f:
        pickle.dump(sklearn_model, f)
    print(f"✅ Modelo scikit-learn guardado: {save_path}iris_sklearn_model.pkl")
    
    # Guardar modelo TensorFlow (compatible con la API)
    tf_model.save(f"{save_path}iris_classifier.h5")
    print(f"✅ Modelo TensorFlow guardado: {save_path}iris_classifier.h5")
    
    # Guardar información del modelo
    model_info = {
        "model_type": "iris_classifier",
        "classes": ["setosa", "versicolor", "virginica"],
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "input_shape": (4,),
        "output_classes": 3,
        "description": "Clasificador de especies de flores Iris"
    }
    
    import json
    with open(f"{save_path}model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Información del modelo guardada: {save_path}model_info.json")

def create_test_data(save_path="models/"):
    """Crear datos de prueba para la API"""
    print("\n🧪 Creando datos de prueba...")
    
    # Crear algunos ejemplos de datos para probar la API
    test_examples = [
        {
            "name": "setosa_example",
            "data": [5.1, 3.5, 1.4, 0.2],
            "expected_class": 0,
            "expected_name": "setosa"
        },
        {
            "name": "versicolor_example", 
            "data": [6.2, 2.9, 4.3, 1.3],
            "expected_class": 1,
            "expected_name": "versicolor"
        },
        {
            "name": "virginica_example",
            "data": [7.2, 3.0, 5.8, 1.6],
            "expected_class": 2,
            "expected_name": "virginica"
        }
    ]
    
    import json
    with open(f"{save_path}test_examples.json", 'w') as f:
        json.dump(test_examples, f, indent=2)
    print(f"✅ Ejemplos de prueba guardados: {save_path}test_examples.json")

def main():
    """Función principal"""
    print("🌸 Entrenando el modelo Iris - Hello World del ML!")
    print("="*60)
    
    # Cargar datos
    X, y, feature_names, target_names, df = load_iris_data()
    
    # Crear visualizaciones
    create_visualizations(df)
    
    # Entrenar modelo scikit-learn
    sklearn_model, sklearn_accuracy = train_sklearn_model(X, y, target_names)
    
    # Entrenar modelo TensorFlow
    tf_model, tf_accuracy, history = train_tensorflow_model(X, y, target_names)
    
    # Guardar modelos
    save_models(sklearn_model, tf_model)
    
    # Crear datos de prueba
    create_test_data()
    
    print("\n" + "="*60)
    print("🎉 ¡Entrenamiento completado!")
    print(f"📊 Precisión scikit-learn: {sklearn_accuracy:.2%}")
    print(f"🧠 Precisión TensorFlow: {tf_accuracy:.2%}")
    print("\n📋 Archivos generados:")
    print("  - models/iris_classifier.h5 (modelo para la API)")
    print("  - models/iris_sklearn_model.pkl (modelo scikit-learn)")
    print("  - models/model_info.json (información del modelo)")
    print("  - models/test_examples.json (ejemplos de prueba)")
    print("  - models/iris_analysis.png (visualizaciones)")
    print("\n🚀 ¡Ya puedes usar la API para hacer predicciones!")

if __name__ == "__main__":
    main()
