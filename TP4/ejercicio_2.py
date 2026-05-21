import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from datos.cargador_datos import CargadorDatos
from algoritmos.algoritmo_genetico_f1 import AlgoritmoGeneticoF1
from utils.metricas import Metricas
from utils.visualizacion import Visualizacion
from utils.herramientas import Herramientas


def ejercicio_2():
    print("EJERCICIO 2: Optimización de Arquitectura con Algoritmo Genético")
    
    cargador = CargadorDatos(test_size=0.25, random_state=42)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargador.cargar_wine()
    
    cantidad_caracteristicas = cargador.obtener_cantidad_caracteristicas()
    cantidad_clases = cargador.obtener_cantidad_clases()
    
    valores_beta = [0.0, 0.1, 0.3, 0.5]
    resultados_beta = {}
    
    for beta in valores_beta:
        alfa = 1.0 - beta
        
        print(f"\nEjecutando AG con β={beta}, α={alfa}")
        
        ag = AlgoritmoGeneticoF1(
            cantidad_caracteristicas=cantidad_caracteristicas,
            cantidad_clases=cantidad_clases,
            tamaño_poblacion=20,
            generaciones=30,
            probabilidad_mutacion=0.05,
            tamaño_torneo=2,
            alfa=alfa,
            beta=beta,
            seed=42
        )
        
        mejor_cromosoma, mejor_fitness = ag.ejecutar(X_entrenamiento, y_entrenamiento)
        
        arquitectura_optima = ag.obtener_mejor_arquitectura()
        print(f"Mejor arquitectura encontrada: {arquitectura_optima}")
        print(f"Mejor fitness: {mejor_fitness:.4f}")
        
        modelo_optimo = MLPClassifier(
            hidden_layer_sizes=arquitectura_optima,
            activation='tanh',
            solver='sgd',
            learning_rate_init=0.01,
            max_iter=300,
            random_state=42
        )
        
        modelo_optimo.fit(X_entrenamiento, y_entrenamiento)
        
        y_predicho = modelo_optimo.predict(X_prueba)
        accuracy_test = Metricas.calcular_accuracy(y_prueba, y_predicho)
        
        cantidad_parametros = calcular_parametros_arquitectura(
            cantidad_caracteristicas, cantidad_clases, arquitectura_optima
        )
        
        print(f"Accuracy en conjunto de prueba: {accuracy_test:.4f}")
        print(f"Parámetros entrenables: {cantidad_parametros}")
        
        matriz_confusion = Metricas.obtener_matriz_confusion(y_prueba, y_predicho)
        print("Matriz de Confusión:")
        print(matriz_confusion)
        
        historial = ag.obtener_historial()
        
        resultados_beta[beta] = {
            'arquitectura': arquitectura_optima,
            'mejor_fitness': mejor_fitness,
            'accuracy_test': accuracy_test,
            'parametros': cantidad_parametros,
            'historial': historial,
            'matriz_confusion': matriz_confusion,
            'y_predicho': y_predicho
        }
        
        fig_evolucion = Visualizacion.graficar_evolucion_fitness(
            historial['mejor_fitness'],
            historial['fitness_promedio'],
            titulo=f"Evolución Fitness - β={beta}"
        )
        
        Visualizacion.guardar_figura(
            fig_evolucion, 
            f"./resultados/evolucion_fitness_beta_{beta}.png"
        )
        
        fig_confusion = Visualizacion.graficar_matriz_confusion(
            y_prueba, y_predicho,
            titulo=f"Matriz de Confusión - β={beta}"
        )
        
        Visualizacion.guardar_figura(
            fig_confusion,
            f"./resultados/matriz_confusion_beta_{beta}.png"
        )
    

    print("TABLA COMPARATIVA")
    
    tabla_comparativa = Herramientas.crear_tabla_comparativa(resultados_beta)
    print(tabla_comparativa.to_string(index=False))
    
    tabla_comparativa.to_csv("./resultados/tabla_comparativa_ejercicio_2.csv", index=False)
    
    print("\nArchivos generados:")
    for beta in valores_beta:
        print(f"  - evolucion_fitness_beta_{beta}.png")
        print(f"  - matriz_confusion_beta_{beta}.png")
    print("  - tabla_comparativa_ejercicio_2.csv")
    
    return resultados_beta, tabla_comparativa


def calcular_parametros_arquitectura(cantidad_caracteristicas, cantidad_clases, arquitectura):
    """Calcula cantidad de parámetros para una arquitectura dada"""
    parametros = 0
    neuronas_anteriores = cantidad_caracteristicas
    
    for neuronas in arquitectura:
        parametros += (neuronas_anteriores * neuronas) + neuronas
        neuronas_anteriores = neuronas
    
    parametros += (neuronas_anteriores * cantidad_clases) + cantidad_clases
    return parametros


if __name__ == "__main__":
    ejercicio_2()
