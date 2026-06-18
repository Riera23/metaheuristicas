import sys
import os
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABC_DIR = os.path.join(BASE_DIR, 'ABC')
sys.path.append(BASE_DIR)
sys.path.append(ABC_DIR)

from ABC.datos_mercado import RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, N_ACTIVOS
from ABC.algoritmo_abc import AlgoritmoABC
from NSGA_II.nsga_ii import NSGAII

from func.funciones import calcular_hipervolumen_2d, exportar_carteras, exportar_hipervolumen, exportar_test_wilcoxon, generar_graficos, wilcoxon
from ABC.utils.metricas import MetricasFinancieras

def main():
    carpeta_salida = os.path.join(BASE_DIR, '_comparaciones')
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Configuración de Hiperparámetros Optimizados (obtenidos del optimizador)
    params_nsga = {'tam_poblacion': 50, 'n_generaciones': 99, 'prob_crossover': 0.76, 'prob_mutacion': 0.005, 'eta_c': 17, 'eta_m': 19}
    params_abc = {'tamano_poblacion': 20, 'limite': 50}
    LAMBDA_OPTIMO = 0.5
    
    # Presupuesto total de evaluaciones (E = Pop * Generaciones para NSGA / E = 2 * Pop * Ciclos para ABC)
    # NSGA: 50 * (99 + 1) = 5000 evaluaciones
    PRESUPUESTO_TOTAL = 5000
    
    np.random.seed(42)
    semillas = np.random.randint(1, 10000, 30).tolist()
    lambdas_barrido = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Estructuras para resultados estadísticos
    nsga_results_por_semilla = {} 
    abc_fronts_por_semilla = {}
    abc_pesos_por_semilla = {}
    dist_sharpe_nsga = []
    dist_hv_nsga = []
    dist_fitness_nsga_caso2, dist_fitness_abc_caso2 = [], []

    print(f"=== Iniciando Comparación Estadística ({len(semillas)} semillas) ===")

    for i, semilla in enumerate(semillas):
        print(f"Iteración {i+1}/{len(semillas)} - Semilla: {semilla}")
        np.random.seed(semilla)
        random.seed(semilla)

        # --- EJECUCIÓN NSGA-II (Una sola vez por semilla) ---
        nsga = NSGAII(
            n_activos=N_ACTIVOS, 
            retornos_esperados=RETORNOS_ESPERADOS, 
            matriz_covarianza=MATRIZ_COVARIANZAS,
            **params_nsga
        )
        pob_nsga, obj_nsga = nsga.ejecutar()
        nsga_results_por_semilla[semilla] = (obj_nsga, pob_nsga)
        
        # Guardar para estadísticas de Sharpe
        sharpes_nsga = (-obj_nsga[:, 1]) / np.sqrt(np.clip(obj_nsga[:, 0], 1e-10, None))
        dist_sharpe_nsga.append(np.max(sharpes_nsga))

        # ABC Barrido: Repartimos el presupuesto entre los lambdas
        # E_por_lambda = 5000 / 11 ~ 454 -> Ciclos = 454 / (2 * 20) ~ 11 ciclos
        evals_por_lambda = PRESUPUESTO_TOTAL // len(lambdas_barrido)
        ciclos_abc = evals_por_lambda // (2 * params_abc['tamano_poblacion'])
        
        frente_abc_muestreado = []
        pesos_abc_muestreado = []
        for lmb in lambdas_barrido:
            abc_frente = AlgoritmoABC(n_activos=N_ACTIVOS, max_ciclos=ciclos_abc, **params_abc)
            sol = abc_frente.ejecutar(RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, lmb)
            var = np.dot(sol.T, np.dot(MATRIZ_COVARIANZAS, sol))
            ret = np.sum(sol * RETORNOS_ESPERADOS)
            frente_abc_muestreado.append([var, -ret])
            pesos_abc_muestreado.append(sol)
        
        abc_fronts_por_semilla[semilla] = frente_abc_muestreado
        abc_pesos_por_semilla[semilla] = pesos_abc_muestreado

        # --- CASO 2: Mejor vs Mejor (Fitness Escalarizado) ---
        # ABC con Lambda Óptimo y presupuesto completo
        ciclos_abc_full = PRESUPUESTO_TOTAL // (2 * params_abc['tamano_poblacion'])
        abc_opt = AlgoritmoABC(n_activos=N_ACTIVOS, max_ciclos=ciclos_abc_full, **params_abc)
        sol_abc_opt = abc_opt.ejecutar(RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, LAMBDA_OPTIMO)
        
        f_obj_abc = MetricasFinancieras.funcion_objetivo(sol_abc_opt, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, LAMBDA_OPTIMO)
        fit_abc = MetricasFinancieras.calcular_fitness(f_obj_abc)
        dist_fitness_abc_caso2.append(fit_abc)

        # Buscar el mejor individuo de NSGA-II para ese mismo Lambda
        # Calculamos el fitness escalar de cada punto en el frente de Pareto de NSGA
        fitness_frente_nsga = []
        for obj in obj_nsga:
            # obj[0] es varianza, -obj[1] es retorno
            f_obj_n = LAMBDA_OPTIMO * obj[0] - (1 - LAMBDA_OPTIMO) * (-obj[1])
            fitness_frente_nsga.append(MetricasFinancieras.calcular_fitness(f_obj_n))
        
        dist_fitness_nsga_caso2.append(np.max(fitness_frente_nsga))

    # --- Cálculo de Hipervolumen con Punto de Referencia Robusto ---
    print("\nDeterminando punto de referencia global y recalculando hipervolúmenes...")
    todos_los_objetivos = []
    for obj, _ in nsga_results_por_semilla.values(): todos_los_objetivos.extend(obj)
    for frente in abc_fronts_por_semilla.values(): todos_los_objetivos.extend(frente)
    todos_los_objetivos = np.array(todos_los_objetivos)
    
    # El punto de referencia debe ser mayor al peor valor encontrado (nadir global)
    nadir_f1 = np.max(todos_los_objetivos[:, 0])
    nadir_f2 = np.max(todos_los_objetivos[:, 1])
    pto_ref_robusto = [nadir_f1 * 1.1, nadir_f2 + 0.001]

    dist_hv_nsga = [calcular_hipervolumen_2d(nsga_results_por_semilla[s][0], pto_ref_robusto) for s in semillas]
    dist_hv_abc = [calcular_hipervolumen_2d(np.array(abc_fronts_por_semilla[s]), pto_ref_robusto) for s in semillas]

    # Identificar mejores frentes globales según el nuevo HV
    idx_mejor_nsga = np.argmax(dist_hv_nsga)
    mejor_obj_nsga, mejor_pob_nsga = nsga_results_por_semilla[semillas[idx_mejor_nsga]]
    
    idx_mejor_abc = np.argmax(dist_hv_abc)
    mejores_obj_abc = np.array(abc_fronts_por_semilla[semillas[idx_mejor_abc]])
    mejores_pesos_abc = np.array(abc_pesos_por_semilla[semillas[idx_mejor_abc]])

    # --- Procesamiento de Resultados y Tests Estadísticos ---
    print("\n=== Analizando Resultados ===")
    
    # Test Caso 1: Hipervolumen
    _, p_val_hv = wilcoxon(dist_hv_nsga, dist_hv_abc)
    # Test Caso 2: Fitness mejor vs mejor
    _, p_val_fit = wilcoxon(dist_fitness_nsga_caso2, dist_fitness_abc_caso2)

    # Exportar resultados estadísticos
    df_stats = pd.DataFrame({
        'Comparación': ['Caso 1: Hipervolumen (Frentes)', 'Caso 2: Fitness (Mejor vs Mejor)'],
        'Media NSGA-II': [np.mean(dist_hv_nsga), np.mean(dist_fitness_nsga_caso2)],
        'Media ABC': [np.mean(dist_hv_abc), np.mean(dist_fitness_abc_caso2)],
        'P-Valor (Wilcoxon)': [p_val_hv, p_val_fit],
        'Sig. (0.05)': [p_val_hv < 0.05, p_val_fit < 0.05]
    })
    df_stats.to_csv(os.path.join(carpeta_salida, 'comparacion_estadistica_final.csv'), index=False)
    
    print(df_stats.to_string())

    # Exportación de archivos detallados
    print("\nExportando archivos de resultados...")
    exportar_carteras(mejor_pob_nsga, mejor_obj_nsga, mejores_pesos_abc, mejores_obj_abc, lambdas_barrido, carpeta_salida)
    exportar_hipervolumen(mejor_obj_nsga, mejores_obj_abc, pto_ref_robusto, carpeta_salida)
    exportar_test_wilcoxon(semillas, abc_fronts_por_semilla, pto_ref_robusto, dist_hv_nsga, dist_sharpe_nsga, carpeta_salida)
    generar_graficos(mejor_obj_nsga, mejores_obj_abc, carpeta_salida)

    print(f"Proceso finalizado. Resultados en: {carpeta_salida}")

if __name__ == '__main__':
    main()