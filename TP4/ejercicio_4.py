import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from datos.cargador_datos import CargadorDatos
from algoritmos.algoritmo_genetico_f1 import AlgoritmoGeneticoF1
from algoritmos.algoritmo_genetico_f2 import AlgoritmoGeneticoF2
from utils.metricas import Metricas
from utils.visualizacion import Visualizacion


def ejercicio_4():
    print("EJERCICIO 4: Análisis de Estabilidad - 10 Corridas F1 y F2")
    
    cargador = CargadorDatos(test_size=0.25, random_state=42)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargador.cargar_wine()
    
    cantidad_caracteristicas = cargador.obtener_cantidad_caracteristicas()
    cantidad_clases = cargador.obtener_cantidad_clases()
    
    semillas = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    
    print("\nEjecutando 10 corridas de F1 (β=0.0)")
 
    
    resultados_f1 = ejecutar_multiples_corridas_f1(
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba,
        cantidad_caracteristicas, cantidad_clases, semillas
    )
    
    print("\nEjecutando 10 corridas de F2 (α=0.6, β=0.2, γ=0.2)")

    
    resultados_f2 = ejecutar_multiples_corridas_f2(
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba,
        cantidad_caracteristicas, cantidad_clases, semillas
    )
    

    print("ANÁLISIS ESTADÍSTICO")
    
    estadisticas_f1 = calcular_estadisticas(resultados_f1, 'F1')
    estadisticas_f2 = calcular_estadisticas(resultados_f2, 'F2')
    
    print("\nEstadísticas F1:")
    print(estadisticas_f1.to_string(index=False))
    
    print("\nEstadísticas F2:")
    print(estadisticas_f2.to_string(index=False))
    
    estadisticas_f1.to_csv("./resultados/estadisticas_f1.csv", index=False)
    estadisticas_f2.to_csv("./resultados/estadisticas_f2.csv", index=False)
    
    df_resultados_f1 = pd.DataFrame(resultados_f1)
    df_resultados_f2 = pd.DataFrame(resultados_f2)
    
    df_resultados_f1.to_csv("./resultados/resultados_corridas_f1.csv", index=False)
    df_resultados_f2.to_csv("./resultados/resultados_corridas_f2.csv", index=False)
    
    print("\nArchivos generados:")
    print("  - estadisticas_f1.csv")
    print("  - estadisticas_f2.csv")
    print("  - resultados_corridas_f1.csv")
    print("  - resultados_corridas_f2.csv")
    
    print("\nDetalles F1:")
    print(f"  Accuracy promedio: {np.mean(df_resultados_f1['accuracy_test']):.4f}")
    print(f"  Accuracy std dev: {np.std(df_resultados_f1['accuracy_test']):.4f}")
    print(f"  Accuracy min: {np.min(df_resultados_f1['accuracy_test']):.4f}")
    print(f"  Accuracy max: {np.max(df_resultados_f1['accuracy_test']):.4f}")
    
    print("\nDetalles F2:")
    print(f"  Accuracy promedio: {np.mean(df_resultados_f2['accuracy_test']):.4f}")
    print(f"  Accuracy std dev: {np.std(df_resultados_f2['accuracy_test']):.4f}")
    print(f"  Accuracy min: {np.min(df_resultados_f2['accuracy_test']):.4f}")
    print(f"  Accuracy max: {np.max(df_resultados_f2['accuracy_test']):.4f}")
    
    fig_boxplot = Visualizacion.graficar_boxplot_comparacion(
        df_resultados_f1['accuracy_test'].values,
        df_resultados_f2['accuracy_test'].values,
        'Comparación Accuracy: F1 vs F2'
    )
    Visualizacion.guardar_figura(fig_boxplot, "./resultados/boxplot_f1_vs_f2.png")
    
    print()
    return resultados_f1, resultados_f2


def ejecutar_multiples_corridas_f1(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba,
                                    cantidad_caracteristicas, cantidad_clases, semillas):
    """Ejecuta 10 corridas de F1 con semillas distintas"""
    resultados = []
    
    for i, semilla in enumerate(semillas):
        print(f"  Corrida {i+1}/10 (semilla={semilla})")
        
        ag = AlgoritmoGeneticoF1(
            cantidad_caracteristicas=cantidad_caracteristicas,
            cantidad_clases=cantidad_clases,
            tamaño_poblacion=20,
            generaciones=30,
            probabilidad_mutacion=0.05,
            tamaño_torneo=2,
            alfa=1.0,
            beta=0.0,
            seed=semilla
        )
        
        mejor_cromosoma, mejor_fitness = ag.ejecutar(X_entrenamiento, y_entrenamiento)
        arquitectura = ag.obtener_mejor_arquitectura()
        
        modelo = MLPClassifier(
            hidden_layer_sizes=arquitectura,
            activation='tanh',
            solver='sgd',
            learning_rate_init=0.01,
            max_iter=300,
            random_state=semilla
        )
        
        modelo.fit(X_entrenamiento, y_entrenamiento)
        
        y_predicho = modelo.predict(X_prueba)
        accuracy = Metricas.calcular_accuracy(y_prueba, y_predicho)
        
        cantidad_parametros = calcular_parametros_arquitectura_f1(
            cantidad_caracteristicas, cantidad_clases, arquitectura
        )
        
        resultados.append({
            'semilla': semilla,
            'mejor_fitness': mejor_fitness,
            'accuracy_test': accuracy,
            'parametros': cantidad_parametros,
            'arquitectura': str(arquitectura)
        })
    
    return resultados


def ejecutar_multiples_corridas_f2(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba,
                                    cantidad_caracteristicas, cantidad_clases, semillas):
    """Ejecuta 10 corridas de F2 con semillas distintas"""
    resultados = []
    
    for i, semilla in enumerate(semillas):
        print(f"  Corrida {i+1}/10 (semilla={semilla})")
        
        ag = AlgoritmoGeneticoF2(
            cantidad_caracteristicas=cantidad_caracteristicas,
            cantidad_clases=cantidad_clases,
            tamaño_poblacion=20,
            generaciones=30,
            probabilidad_mutacion=0.05,
            tamaño_torneo=2,
            alfa=0.6,
            beta=0.2,
            gamma=0.2,
            seed=semilla
        )
        
        mejor_cromosoma, mejor_fitness = ag.ejecutar(X_entrenamiento, y_entrenamiento)
        caracteristicas_activas, arquitectura = ag.obtener_mejor_caracteristicas_y_arquitectura()
        
        cantidad_caracteristicas_seleccionadas = np.sum(caracteristicas_activas)
        
        X_prueba_filtrado = X_prueba[:, caracteristicas_activas]
        X_entrenamiento_filtrado = X_entrenamiento[:, caracteristicas_activas]
        
        modelo = MLPClassifier(
            hidden_layer_sizes=arquitectura,
            activation='tanh',
            solver='sgd',
            learning_rate_init=0.01,
            max_iter=300,
            random_state=semilla
        )
        
        modelo.fit(X_entrenamiento_filtrado, y_entrenamiento)
        
        y_predicho = modelo.predict(X_prueba_filtrado)
        accuracy = Metricas.calcular_accuracy(y_prueba, y_predicho)
        
        cantidad_parametros = calcular_parametros_arquitectura_f2(
            cantidad_caracteristicas_seleccionadas, cantidad_clases, arquitectura
        )
        
        resultados.append({
            'semilla': semilla,
            'mejor_fitness': mejor_fitness,
            'accuracy_test': accuracy,
            'parametros': cantidad_parametros,
            'arquitectura': str(arquitectura),
            'variables_seleccionadas': cantidad_caracteristicas_seleccionadas
        })
    
    return resultados


def calcular_estadisticas(resultados, nombre_algoritmo):
    """Calcula estadísticas de las corridas"""
    accuracies = [r['accuracy_test'] for r in resultados]
    parametros = [r['parametros'] for r in resultados]
    
    estadisticas = {
        'Algoritmo': [nombre_algoritmo],
        'Accuracy Media': [np.mean(accuracies)],
        'Accuracy Std': [np.std(accuracies)],
        'Accuracy Min': [np.min(accuracies)],
        'Accuracy Max': [np.max(accuracies)],
        'Accuracy Mediana': [np.median(accuracies)],
        'Parámetros Media': [np.mean(parametros)],
        'Parámetros Std': [np.std(parametros)]
    }
    
    return pd.DataFrame(estadisticas)


def calcular_parametros_arquitectura_f1(cantidad_caracteristicas, cantidad_clases, arquitectura):
    """Calcula cantidad de parámetros para F1"""
    parametros = 0
    neuronas_anteriores = cantidad_caracteristicas
    
    for neuronas in arquitectura:
        parametros += (neuronas_anteriores * neuronas) + neuronas
        neuronas_anteriores = neuronas
    
    parametros += (neuronas_anteriores * cantidad_clases) + cantidad_clases
    return parametros


def calcular_parametros_arquitectura_f2(cantidad_caracteristicas, cantidad_clases, arquitectura):
    """Calcula cantidad de parámetros para F2"""
    return calcular_parametros_arquitectura_f1(cantidad_caracteristicas, cantidad_clases, arquitectura)


if __name__ == "__main__":
    ejercicio_4()
