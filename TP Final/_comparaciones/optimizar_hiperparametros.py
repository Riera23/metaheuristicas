import os
import sys
import numpy as np
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABC_DIR = os.path.join(BASE_DIR, 'ABC')
sys.path.append(BASE_DIR)
sys.path.append(ABC_DIR)

from ABC.datos_mercado import RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, N_ACTIVOS
from _comparaciones.func.optimizador_nsga2 import OptimizadorNSGAII

def main():
    print("=== INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS ===")
    
    espacio_nsga2 = {
        'tam_poblacion': 50,
        'n_generaciones': 100,
        'prob_crossover': (0.4, 0.9),
        'prob_mutacion': (0.01, 0.2),
        'eta_c': (10, 30),
        'eta_m': (10, 30)
    }
    
    punto_referencia = np.array([0.0, 0.0])

    opt_nsga = OptimizadorNSGAII(
        n_activos=N_ACTIVOS,
        retornos=RETORNOS_ESPERADOS,
        covarianzas=MATRIZ_COVARIANZAS,
        espacio_busqueda=espacio_nsga2,
        punto_ref=punto_referencia,
        n_iteraciones=10
    )
    mejor_config_nsga, mejor_score_nsga = opt_nsga.optimizar()


    print("MEJORES RESULTADOS ENCONTRADOS:")
    print(f"NSGA-II -> Score: {mejor_score_nsga:.6f} | Config: {mejor_config_nsga}")
    print("="*40)

if __name__ == "__main__":
    main()