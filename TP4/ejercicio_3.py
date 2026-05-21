import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from datos.cargador_datos import CargadorDatos
from algoritmos.algoritmo_genetico_f2 import AlgoritmoGeneticoF2
from utils.metricas import Metricas
from utils.visualizacion import Visualizacion
from utils.herramientas import Herramientas


def ejercicio_3():
    print("EJERCICIO 3: Optimización con Selección de Variables")
    
    cargador = CargadorDatos(test_size=0.25, random_state=42)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargador.cargar_wine()
    
    cantidad_caracteristicas = cargador.obtener_cantidad_caracteristicas()
    cantidad_clases = cargador.obtener_cantidad_clases()
    
    configuraciones = {
        'C1': (1.0, 0.0, 0.0),
        'C2': (0.8, 0.1, 0.1),
        'C3': (0.6, 0.2, 0.2),
        'C4': (0.4, 0.5, 0.1),
        'C5': (0.4, 0.1, 0.5)
    }
    
    resultados_configuraciones = {}
    
    for nombre_config, (alfa, beta, gamma) in configuraciones.items():
        print(f"\nEjecutando AG con configuración {nombre_config}")
        print(f"  α={alfa}, β={beta}, γ={gamma}")
        
        ag = AlgoritmoGeneticoF2(
            cantidad_caracteristicas=cantidad_caracteristicas,
            cantidad_clases=cantidad_clases,
            tamaño_poblacion=20,
            generaciones=30,
            probabilidad_mutacion=0.05,
            tamaño_torneo=2,
            alfa=alfa,
            beta=beta,
            gamma=gamma,
            seed=42
        )
        
        mejor_cromosoma, mejor_fitness = ag.ejecutar(X_entrenamiento, y_entrenamiento)
        
        caracteristicas_activas, arquitectura_optima = ag.obtener_mejor_caracteristicas_y_arquitectura()
        cantidad_caracteristicas_seleccionadas = np.sum(caracteristicas_activas)
        
        print(f"Mejor fitness: {mejor_fitness:.4f}")
        print(f"Arquitectura: {arquitectura_optima}")
        print(f"Características seleccionadas: {cantidad_caracteristicas_seleccionadas}/{cantidad_caracteristicas}")
        print(f"Índices de características: {np.where(caracteristicas_activas)[0].tolist()}")
        
        X_prueba_filtrado = X_prueba[:, caracteristicas_activas]
        
        modelo_optimo = MLPClassifier(
            hidden_layer_sizes=arquitectura_optima,
            activation='tanh',
            solver='sgd',
            learning_rate_init=0.01,
            max_iter=300,
            random_state=42
        )
        
        X_entrenamiento_filtrado = X_entrenamiento[:, caracteristicas_activas]
        modelo_optimo.fit(X_entrenamiento_filtrado, y_entrenamiento)
        
        y_predicho = modelo_optimo.predict(X_prueba_filtrado)
        accuracy_test = Metricas.calcular_accuracy(y_prueba, y_predicho)
        
        cantidad_parametros = calcular_parametros_arquitectura(
            cantidad_caracteristicas_seleccionadas, cantidad_clases, arquitectura_optima
        )
        
        print(f"Accuracy en conjunto de prueba: {accuracy_test:.4f}")
        print(f"Parámetros entrenables: {cantidad_parametros}")
        
        matriz_confusion = Metricas.obtener_matriz_confusion(y_prueba, y_predicho)
        print("Matriz de Confusión:")
        print(matriz_confusion)
        
        historial = ag.obtener_historial()
        
        resultados_configuraciones[nombre_config] = {
            'arquitectura': arquitectura_optima,
            'mejor_fitness': mejor_fitness,
            'accuracy_test': accuracy_test,
            'parametros': cantidad_parametros,
            'caracteristicas_seleccionadas': cantidad_caracteristicas_seleccionadas,
            'indices_caracteristicas': np.where(caracteristicas_activas)[0].tolist(),
            'historial': historial,
            'matriz_confusion': matriz_confusion,
            'y_predicho': y_predicho
        }
        
        fig_evolucion = Visualizacion.graficar_evolucion_fitness(
            historial['mejor_fitness'],
            historial['fitness_promedio'],
            titulo=f"Evolución Fitness - {nombre_config}"
        )
        
        Visualizacion.guardar_figura(
            fig_evolucion, 
            f"./resultados/evolucion_fitness_{nombre_config}.png"
        )
        
        fig_confusion = Visualizacion.graficar_matriz_confusion(
            y_prueba, y_predicho,
            titulo=f"Matriz de Confusión - {nombre_config}"
        )
        
        Visualizacion.guardar_figura(
            fig_confusion,
            f"./resultados/matriz_confusion_{nombre_config}.png"
        )
    
    print("TABLA COMPARATIVA - EJERCICIO 3")
    
    tabla_comparativa = crear_tabla_comparativa_ej3(resultados_configuraciones)
    print(tabla_comparativa.to_string(index=False))
    
    tabla_comparativa.to_csv("./resultados/tabla_comparativa_ejercicio_3.csv", index=False)
    
    print("\nArchivos generados:")
    for config in configuraciones.keys():
        print(f"  - evolucion_fitness_{config}.png")
        print(f"  - matriz_confusion_{config}.png")
    print("  - tabla_comparativa_ejercicio_3.csv")
    print()
    
    return resultados_configuraciones, tabla_comparativa


def calcular_parametros_arquitectura(cantidad_caracteristicas, cantidad_clases, arquitectura):
    """Calcula cantidad de parámetros para una arquitectura dada"""
    parametros = 0
    neuronas_anteriores = cantidad_caracteristicas
    
    for neuronas in arquitectura:
        parametros += (neuronas_anteriores * neuronas) + neuronas
        neuronas_anteriores = neuronas
    
    parametros += (neuronas_anteriores * cantidad_clases) + cantidad_clases
    return parametros


def crear_tabla_comparativa_ej3(resultados_configuraciones):
    """Crea tabla comparativa para diferentes configuraciones"""
    datos_tabla = []
    
    for config, resultado in resultados_configuraciones.items():
        datos_tabla.append({
            'Configuración': config,
            'Mejor Fitness': resultado['mejor_fitness'],
            'Arquitectura': str(resultado['arquitectura']),
            'Accuracy Test': resultado['accuracy_test'],
            'Parámetros': resultado['parametros'],
            'Variables Seleccionadas': resultado['caracteristicas_seleccionadas']
        })
    
    return pd.DataFrame(datos_tabla)


if __name__ == "__main__":
    ejercicio_3()
