#!/usr/bin/env python3
"""
Entrenamiento paralelo con Ray - Demostración educativa
Compara entrenamiento secuencial vs paralelo con métricas visuales
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import json
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ray

# Configurar Ray para uso local
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

def load_iris_data():
    """Cargar dataset Iris"""
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.target_names

@ray.remote
def train_single_model(X_train, y_train, X_test, y_test, params, model_id):
    """Entrenar un modelo individual con parámetros específicos"""
    print(f"🌱 Entrenando modelo {model_id} con parámetros: {params}")
    
    start_time = time.time()
    
    # Crear y entrenar modelo
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    return {
        'model_id': model_id,
        'params': params,
        'accuracy': accuracy,
        'training_time': training_time
    }

@ray.remote
def cross_validate_model(X, y, params, cv_folds=5):
    """Validación cruzada paralela"""
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_folds)
    return {
        'params': params,
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'scores': scores.tolist()
    }

def train_sequential(X, y):
    """Entrenamiento secuencial (baseline)"""
    print("🐌 Iniciando entrenamiento SECUENCIAL...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Diferentes configuraciones de hiperparámetros
    param_grid = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 10}
    ]
    
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    results = []
    
    # Entrenar modelos secuencialmente
    for i, params in enumerate(param_grid):
        print(f"  📊 Entrenando modelo {i+1}/4...")
        
        model_start = time.time()
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_time = time.time() - model_start
        
        results.append({
            'model_id': i+1,
            'params': params,
            'accuracy': accuracy,
            'training_time': model_time
        })
    
    total_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    return {
        'method': 'sequential',
        'total_time': total_time,
        'memory_used': final_memory - initial_memory,
        'results': results,
        'cpu_count': 1
    }

def train_parallel(X, y):
    """Entrenamiento paralelo con Ray"""
    print("🚀 Iniciando entrenamiento PARALELO con Ray...")
    
    # Inicializar Ray con configuración robusta sin dashboard
    if not ray.is_initialized():
        try:
            ray.init(
                num_cpus=psutil.cpu_count(),
                include_dashboard=False,
                dashboard_host="127.0.0.1",
                dashboard_port=None,
                log_to_driver=False,
                configure_logging=False
            )
            print(f"  ✅ Ray inicializado correctamente con {ray.cluster_resources().get('CPU', 0)} CPUs")
        except Exception as e:
            print(f"  ⚠️  Error al inicializar Ray: {e}")
            print("  🔄 Intentando con configuración mínima...")
            ray.init(
                num_cpus=min(4, psutil.cpu_count()),  # Limitar CPUs en caso de problemas
                include_dashboard=False,
                log_to_driver=False
            )
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Diferentes configuraciones de hiperparámetros
    param_grid = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 10}
    ]
    
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Crear tareas remotas para entrenamiento paralelo
    print(f"  🔄 Distribuyendo {len(param_grid)} modelos en {ray.cluster_resources()['CPU']} CPUs...")
    
    tasks = []
    for i, params in enumerate(param_grid):
        task = train_single_model.remote(X_train, y_train, X_test, y_test, params, i+1)
        tasks.append(task)
    
    # Ejecutar todas las tareas en paralelo
    print("  ⚡ Ejecutando entrenamiento paralelo...")
    try:
        results = ray.get(tasks)
    except Exception as e:
        print(f"  ❌ Error durante el entrenamiento paralelo: {e}")
        print("  🔄 Cerrando Ray e intentando secuencial como fallback...")
        ray.shutdown()
        print("  ⚠️  Ejecutando versión secuencial como alternativa...")
        return train_sequential(X, y)
    
    total_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    return {
        'method': 'parallel',
        'total_time': total_time,
        'memory_used': final_memory - initial_memory,
        'results': results,
        'cpu_count': ray.cluster_resources()['CPU']
    }

def create_performance_comparison(sequential_results, parallel_results, save_path="models/"):
    """Crear visualizaciones de comparación de rendimiento"""
    print("\n📊 Generando visualizaciones de comparación...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de tiempo total
    methods = ['Secuencial', 'Paralelo']
    times = [sequential_results['total_time'], parallel_results['total_time']]
    speedup = sequential_results['total_time'] / parallel_results['total_time']
    
    bars1 = ax1.bar(methods, times, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.set_title(f'Tiempo de Entrenamiento\nSpeedup: {speedup:.2f}x')
    
    # Agregar valores en las barras
    for bar, time_val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 2. Uso de memoria
    memory_usage = [sequential_results['memory_used'], parallel_results['memory_used']]
    bars2 = ax2.bar(methods, memory_usage, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('Memoria (MB)')
    ax2.set_title('Uso de Memoria')
    
    for bar, mem_val in zip(bars2, memory_usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mem_val:.1f} MB', ha='center', va='bottom')
    
    # 3. Accuracy por modelo
    seq_accuracies = [r['accuracy'] for r in sequential_results['results']]
    par_accuracies = [r['accuracy'] for r in parallel_results['results']]
    
    x = np.arange(len(seq_accuracies))
    width = 0.35
    
    ax3.bar(x - width/2, seq_accuracies, width, label='Secuencial', color='#ff7f0e', alpha=0.8)
    ax3.bar(x + width/2, par_accuracies, width, label='Paralelo', color='#2ca02c', alpha=0.8)
    
    ax3.set_xlabel('Modelo')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy por Modelo')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'M{i+1}' for i in range(len(seq_accuracies))])
    ax3.legend()
    
    # 4. Eficiencia de recursos
    cpu_counts = [sequential_results['cpu_count'], parallel_results['cpu_count']]
    efficiency = [
        1.0,  # Secuencial siempre 100% eficiente con 1 CPU
        speedup / parallel_results['cpu_count']  # Eficiencia paralela
    ]
    
    bars4 = ax4.bar(methods, efficiency, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
    ax4.set_ylabel('Eficiencia')
    ax4.set_title('Eficiencia de Paralelización')
    ax4.set_ylim(0, 1.1)
    
    for bar, eff_val in zip(bars4, efficiency):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{eff_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 Gráficos guardados en: {save_path}performance_comparison.png")
    plt.close()

def save_benchmark_results(sequential_results, parallel_results, save_path="models/"):
    """Guardar métricas numéricas"""
    speedup = sequential_results['total_time'] / parallel_results['total_time']
    efficiency = speedup / parallel_results['cpu_count']
    
    benchmark = {
        'summary': {
            'speedup': round(speedup, 2),
            'efficiency': round(efficiency, 2),
            'time_saved': round(sequential_results['total_time'] - parallel_results['total_time'], 2),
            'cpu_cores_used': int(parallel_results['cpu_count'])
        },
        'sequential': sequential_results,
        'parallel': parallel_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs(save_path, exist_ok=True)
    
    with open(f"{save_path}benchmark_results.json", 'w') as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"📋 Métricas guardadas en: {save_path}benchmark_results.json")
    return benchmark

def print_summary(benchmark):
    """Imprimir resumen de resultados"""
    print("\n" + "="*60)
    print("🎯 RESUMEN DE COMPARACIÓN DE RENDIMIENTO")
    print("="*60)
    print(f"🚀 Speedup obtenido: {benchmark['summary']['speedup']}x")
    print(f"⚡ Eficiencia: {benchmark['summary']['efficiency']:.2f}")
    print(f"⏱️  Tiempo ahorrado: {benchmark['summary']['time_saved']:.2f} segundos")
    print(f"🔧 CPUs utilizadas: {benchmark['summary']['cpu_cores_used']}")
    print("\n📊 Tiempos detallados:")
    print(f"  • Secuencial: {benchmark['sequential']['total_time']:.2f}s")
    print(f"  • Paralelo:   {benchmark['parallel']['total_time']:.2f}s")
    print("\n💾 Memoria utilizada:")
    print(f"  • Secuencial: {benchmark['sequential']['memory_used']:.1f} MB")
    print(f"  • Paralelo:   {benchmark['parallel']['memory_used']:.1f} MB")

def main():
    """Función principal"""
    print("🌸 Comparación de Entrenamiento: Secuencial vs Paralelo con Ray")
    print("="*70)
    
    # Cargar datos
    X, y, target_names = load_iris_data()
    print(f"📊 Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"🏷️  Clases: {target_names}")
    
    # Entrenamiento secuencial
    sequential_results = train_sequential(X, y)
    
    # Entrenamiento paralelo
    parallel_results = train_parallel(X, y)
    
    # Limpiar Ray
    ray.shutdown()
    
    # Crear visualizaciones
    create_performance_comparison(sequential_results, parallel_results)
    
    # Guardar resultados
    benchmark = save_benchmark_results(sequential_results, parallel_results)
    
    # Mostrar resumen
    print_summary(benchmark)
    
    print("\n✅ ¡Comparación completada! Revisa los archivos generados:")
    print("  📊 models/performance_comparison.png")
    print("  📋 models/benchmark_results.json")

if __name__ == "__main__":
    main()