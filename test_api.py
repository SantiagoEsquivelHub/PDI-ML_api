#!/usr/bin/env python3
"""
Script para probar la API ML con el modelo Iris
"""

import requests
import json

# Configuración
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Probar endpoint de salud"""
    print("🏥 Probando endpoint de salud...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API funcionando correctamente")
            print(f"   Modelo cargado: {data['model_ready']}")
            print(f"   Tipo de modelo: {data['model_type']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando con la API: {e}")
        return False

def test_iris_prediction():
    """Probar predicción con datos Iris"""
    print("\n🌸 Probando predicción Iris...")
    
    # Ejemplos de datos para cada especie
    test_cases = [
        {
            "name": "Setosa",
            "data": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            "expected": "setosa"
        },
        {
            "name": "Versicolor", 
            "data": {
                "sepal_length": 6.2,
                "sepal_width": 2.9,
                "petal_length": 4.3,
                "petal_width": 1.3
            },
            "expected": "versicolor"
        },
        {
            "name": "Virginica",
            "data": {
                "sepal_length": 7.2,
                "sepal_width": 3.0,
                "petal_length": 5.8,
                "petal_width": 1.6
            },
            "expected": "virginica"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Prueba {i+1}: {test_case['name']}")
        print(f"   Datos: {test_case['data']}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/iris",
                json=test_case["data"]
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_species = result["predicted_species"]
                confidence = result["confidence"]
                
                print(f"   ✅ Predicción: {predicted_species}")
                print(f"   📊 Confianza: {confidence:.2%}")
                
                if predicted_species == test_case["expected"]:
                    print(f"   🎯 ¡Correcto! (esperado: {test_case['expected']})")
                    success_count += 1
                else:
                    print(f"   ❌ Incorrecto (esperado: {test_case['expected']})")
                    
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Resultados: {success_count}/{len(test_cases)} predicciones correctas")
    return success_count == len(test_cases)

def test_interactive():
    """Modo interactivo para probar con datos personalizados"""
    print("\n🎮 Modo interactivo - Ingresa tus propios datos:")
    print("Medidas típicas:")
    print("  - Setosa: sépalos pequeños, pétalos muy pequeños")
    print("  - Versicolor: tamaño medio")
    print("  - Virginica: sépalos y pétalos grandes")
    
    try:
        sepal_length = float(input("🌿 Largo del sépalo (cm, ej: 5.1): "))
        sepal_width = float(input("🌿 Ancho del sépalo (cm, ej: 3.5): "))
        petal_length = float(input("🌸 Largo del pétalo (cm, ej: 1.4): "))
        petal_width = float(input("🌸 Ancho del pétalo (cm, ej: 0.2): "))
        
        data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        print(f"\n🔮 Enviando datos: {data}")
        
        response = requests.post(f"{API_BASE_URL}/predict/iris", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n🎉 Resultado:")
            print(f"   🌸 Especie: {result['predicted_species']}")
            print(f"   📊 Confianza: {result['confidence']:.2%}")
            print(f"   🔢 Clase: {result['predicted_class']}")
            print(f"   📈 Todas las probabilidades:")
            for i, prob in enumerate(result['all_predictions']):
                species = ["setosa", "versicolor", "virginica"][i]
                print(f"      {species}: {prob:.2%}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except ValueError:
        print("❌ Por favor ingresa números válidos")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_curl_examples():
    """Mostrar ejemplos de comandos curl"""
    print("\n📋 Ejemplos de comandos curl:")
    print("\n1. Verificar salud de la API:")
    print(f"curl {API_BASE_URL}/health")
    
    print("\n2. Predicción Iris (setosa):")
    print(f"""curl -X POST "{API_BASE_URL}/predict/iris" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }}'""")
    
    print("\n3. Predicción Iris (versicolor):")
    print(f"""curl -X POST "{API_BASE_URL}/predict/iris" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "sepal_length": 6.2,
       "sepal_width": 2.9,
       "petal_length": 4.3,
       "petal_width": 1.3
     }}'""")

def main():
    """Función principal"""
    print("🌸 Tester para API ML - Clasificador Iris")
    print("=" * 50)
    
    # Probar conexión
    if not test_health():
        print("\n❌ No se puede conectar con la API.")
        print("💡 Asegúrate de que el servidor esté ejecutándose:")
        print("   uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Probar predicciones automáticas
    test_iris_prediction()
    
    while True:
        print("\n" + "=" * 50)
        print("🎮 Opciones:")
        print("1. 🧪 Ejecutar pruebas automáticas")
        print("2. 🎯 Modo interactivo")
        print("3. 📋 Mostrar ejemplos curl")
        print("4. 🚪 Salir")
        
        choice = input("\nElige una opción (1-4): ").strip()
        
        if choice == "1":
            test_iris_prediction()
        elif choice == "2":
            test_interactive()
        elif choice == "3":
            show_curl_examples()
        elif choice == "4":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()
