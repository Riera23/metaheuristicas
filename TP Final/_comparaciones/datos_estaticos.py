import sys
import os
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABC_DIR = os.path.join(BASE_DIR, 'ABC')
sys.path.append(BASE_DIR)
sys.path.append(ABC_DIR)

from ABC.datos_mercado import RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, N_ACTIVOS
from ABC.algoritmo_abc import AlgoritmoABC
from NSGA_II.nsga_ii import NSGAII

from func.funciones import calcular_hipervolumen_2d, exportar_carteras, exportar_hipervolumen, exportar_test_wilcoxon, generar_graficos


def main():
    carpeta_salida = os.path.join(BASE_DIR, '_comparaciones')
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # semillas fijas para garantizar reproducibilidad
    np.random.seed(42)
    semillas = np.random.randint(1, 10000, 30).tolist()


    # NSGA
    print("Ejecución de las corridas para NSGA")
    
    # guardamos el mejor frente
    mejor_hv_nsga = -1
    mejor_obj_nsga = None
    mejor_pob_nsga = None 
    pto_ref_temp = [0.01, 0.01] 
    
    # listas para test estadisticos
    dist_hv_nsga, dist_sharpe_nsga = [], []
    
    for semilla in semillas:
        np.random.seed(semilla)
        random.seed(semilla)
        
        nsga = NSGAII(n_activos=N_ACTIVOS, retornos_esperados=RETORNOS_ESPERADOS, 
                      matriz_covarianza=MATRIZ_COVARIANZAS, tam_poblacion=50, n_generaciones=100) 
        pob_nsga, obj_nsga = nsga.ejecutar()
        
        hv_actual = calcular_hipervolumen_2d(obj_nsga, pto_ref_temp)
        
        # guardamos el hipervolumen para test estadistico
        dist_hv_nsga.append(hv_actual)
        sharpes_nsga = (-obj_nsga[:, 1] - 0.0) / np.sqrt(obj_nsga[:, 0])
        dist_sharpe_nsga.append(np.max(sharpes_nsga))

        # guardamos el mejor hipervoluemn
        if hv_actual > mejor_hv_nsga:
            mejor_hv_nsga = hv_actual
            mejor_obj_nsga = obj_nsga
            mejor_pob_nsga = pob_nsga


    # ABC para cada Lambda
    print("Ejecución de las 10 corridas para ABC")
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mejores_obj_abc = []
    mejores_pesos_abc = []
    
    # para test estadisticos
    abc_fronts_por_semilla = {s: [] for s in semillas}
    
    # por cada lambda usamos las semillas
    for lmbda in lambdas:
        print(f"  Optimizando para λ = {lmbda}...")
        mejor_valor_escalarizado = float('inf')
        mejor_obj_lmbda = None
        mejor_peso_lmbda = None
        
        for semilla in semillas:
            np.random.seed(semilla)
            random.seed(semilla)
            
            # ejecucion del algoritmo
            abc = AlgoritmoABC(n_activos=N_ACTIVOS, tamano_poblacion=20, max_ciclos=100)
            mejor_peso = abc.ejecutar(RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, lmbda)
            
            varianza = np.dot(mejor_peso.T, np.dot(MATRIZ_COVARIANZAS, mejor_peso))
            retorno = np.sum(mejor_peso * RETORNOS_ESPERADOS)

            # guardamos para test estadistico
            abc_fronts_por_semilla[semilla].append([varianza, -retorno])
            
            # guardamos la mejor funcion fitness escalar
            valor_escalarizado = lmbda * varianza - (1 - lmbda) * retorno
            
            if valor_escalarizado < mejor_valor_escalarizado:
                mejor_valor_escalarizado = valor_escalarizado
                mejor_obj_lmbda = [varianza, -retorno] # [f1, f2] para comparar luego
                mejor_peso_lmbda = mejor_peso
                
        mejores_obj_abc.append(mejor_obj_lmbda)
        mejores_pesos_abc.append(mejor_peso_lmbda)
        
    mejores_obj_abc = np.array(mejores_obj_abc)
    mejores_pesos_abc = np.array(mejores_pesos_abc)


    # Exportamos los resultados
    print("Exportación de carteras")
    exportar_carteras(mejor_pob_nsga, mejor_obj_nsga, mejores_pesos_abc, mejores_obj_abc, lambdas, carpeta_salida)
    
    print("Calculo del hipervolumen final")
    exportar_hipervolumen(mejor_obj_nsga, mejores_obj_abc, carpeta_salida)
    
    print("Exportacion de test de Wilcoxon")
    exportar_test_wilcoxon(semillas, abc_fronts_por_semilla, pto_ref_temp, dist_hv_nsga, dist_sharpe_nsga, carpeta_salida)
    
    print("Generación de gráficos")
    generar_graficos(mejor_obj_nsga, mejores_obj_abc, carpeta_salida)

if __name__ == '__main__':
    main()