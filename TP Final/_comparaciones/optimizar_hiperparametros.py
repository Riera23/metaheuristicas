import os
import sys
import numpy as np
import random

# Configurar el entorno para importar los módulos del proyecto de forma robusta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABC_DIR = os.path.join(BASE_DIR, 'ABC')
sys.path.append(BASE_DIR)
sys.path.append(ABC_DIR)

# Importar datos del mercado y clases de optimización
from ABC.datos_mercado import RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, N_ACTIVOS
from _comparaciones.func.optimizador_nsga2 import OptimizadorNSGAII
from _comparaciones.func.optimizador_abc import OptimizadorABC

def main():
    """
    Script para ejecutar la búsqueda aleatoria de hiperparámetros
    tanto para NSGA-II como para ABC, utilizando las clases base implementadas.
    """
    print("=== INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS ===")
    
    # 1. Configuración de búsqueda para NSGA-II
    # Se busca maximizar el Ratio de Sharpe promedio del frente de Pareto.
    print("\n--- Optimizando NSGA-II ---")
    espacio_nsga2 = {
        'tam_poblacion': [20, 50, 100],
        'n_generaciones': [50, 100],
        'prob_crossover': (0.4, 0.9),
        'prob_mutacion': (0.01, 0.2),
        'eta_c': (10, 30),
        'eta_m': (10, 30)
    }
    
    opt_nsga = OptimizadorNSGAII(
        n_activos=N_ACTIVOS,
        retornos=RETORNOS_ESPERADOS,
        covarianzas=MATRIZ_COVARIANZAS,
        espacio_busqueda=espacio_nsga2,
        n_iteraciones=10
    )
    mejor_config_nsga, mejor_score_nsga = opt_nsga.optimizar()

    # 2. Configuración de búsqueda para ABC
    # Se utiliza un lambda = 0.5 (balance riesgo/retorno) para evaluar la calidad del algoritmo.
    print("\n--- Optimizando ABC (Lambda = 0.5) ---")
    espacio_abc = {
        'tamano_poblacion': [10, 20, 50],
        'max_ciclos': [50, 100, 200],
        'limite': [10, 50, 100]
    }
    
    opt_abc = OptimizadorABC(
        n_activos=N_ACTIVOS,
        retornos=RETORNOS_ESPERADOS,
        covarianzas=MATRIZ_COVARIANZAS,
        lmbda=0.5,
        espacio_busqueda=espacio_abc,
        n_iteraciones=10
    )
    mejor_config_abc, mejor_score_abc = opt_abc.optimizar()

    print("\n" + "="*40)
    print("MEJORES RESULTADOS ENCONTRADOS:")
    print(f"NSGA-II -> Score: {mejor_score_nsga:.6f} | Config: {mejor_config_nsga}")
    print(f"ABC     -> Score: {mejor_score_abc:.6f} | Config: {mejor_config_abc}")
    print("="*40)

if __name__ == "__main__":
    main()