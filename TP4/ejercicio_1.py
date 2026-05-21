import numpy as np
import pandas as pd
from datos.cargador_datos import CargadorDatos
from algoritmos.mlp_modelo import ModeloMLP
from utils.metricas import Metricas
from utils.visualizacion import Visualizacion
from utils.herramientas import Herramientas


def ejercicio_1():
    print("EJERCICIO 1: Entrenamiento MLP con hiperparámetros fijos")
    
    cargador = CargadorDatos(test_size=0.25, random_state=42)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargador.cargar_wine()
    
    print(f"Datos cargados: {X_entrenamiento.shape[0]} muestras de entrenamiento")
    print(f"                {X_prueba.shape[0]} muestras de prueba")
    print(f"                {X_entrenamiento.shape[1]} características")
    print(f"                {cargador.obtener_cantidad_clases()} clases")
    print()
    
    modelo = ModeloMLP(
        capas_ocultas=(16, 8),
        activacion="tanh",
        solver="sgd",
        tasa_aprendizaje=0.01,
        iteraciones_max=300,
        random_state=42
    )
    
    modelo.entrenar(X_entrenamiento, y_entrenamiento)
    
    cantidad_parametros = modelo.obtener_cantidad_parametros()
    print(f"Cantidad de parámetros entrenables: {cantidad_parametros}")
    print()
    
    y_predicho = modelo.predecir(X_prueba)
    accuracy = Metricas.calcular_accuracy(y_prueba, y_predicho)
    
    print(f"Accuracy en conjunto de prueba: {accuracy:.4f}")
    print()
    
    print("Matriz de Confusión:")
    matriz_confusion = Metricas.obtener_matriz_confusion(y_prueba, y_predicho)
    print(matriz_confusion)
    print()
    
    print("Reporte de Clasificación:")
    reporte = Metricas.obtener_reporte_clasificacion(y_prueba, y_predicho)
    print(reporte)
    print()
    
    df_resultados = Herramientas.crear_dataframe_resultados(
        accuracy=accuracy,
        cantidad_parametros=cantidad_parametros,
        nombre_modelo="MLP (16,8)"
    )
    
    print("Tabla de Resultados:")
    print(df_resultados.to_string(index=False))
    print()
    
    fig_confusion = Visualizacion.graficar_matriz_confusion(
        y_prueba, y_predicho, 
        titulo="Matriz de Confusión - Ejercicio 1"
    )
    
    Visualizacion.guardar_figura(fig_confusion, "./resultados/matriz_confusion_ej1.png")
    
    df_resultados.to_csv("./resultados/resultados_ejercicio_1.csv", index=False)
    
    print("Archivos guardados:")
    print("  - matriz_confusion_ej1.png")
    print("  - resultados_ejercicio_1.csv")
    print()
    
    return df_resultados, accuracy, cantidad_parametros


if __name__ == "__main__":
    ejercicio_1()
