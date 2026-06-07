import random
import numpy as np
from datos_mercado import N_ACTIVOS, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS
from algoritmos.algoritmo_abc import AlgoritmoABC
from utils.metricas import MetricasFinancieras
from utils.visualizacion import Visualizacion

def evaluacionABC():
    print("Iniciando evaluacion del barrido de aversión al riesgo (10 semillas aleatorias fijas por cada Lambda)...")
    
    valores_lambda = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    resultados = []
    num_semillas = 10
    
    # 1. Generamos las 10 semillas aleatorias únicas entre 0 y 1000 UNA SOLA VEZ
    semillas_fijas = [42, 128, 314, 876, 555, 902, 23, 777, 404, 619]
    
    # Instanciamos el motor del algoritmo una sola vez
    abc = AlgoritmoABC(n_activos=N_ACTIVOS, tamano_poblacion=20, max_ciclos=500, limite=50)
    
    for lmbda in valores_lambda:
        print(f"  Optimizando para λ = {lmbda}...")
        mejor_peso_lambda = None
        mejor_fitness_lambda = -float('inf')
        
        # Bucle iterando sobre nuestra lista de semillas aleatorias fijas
        for semilla in semillas_fijas:
            # Fijamos la semilla para reproducibilidad
            np.random.seed(semilla)
            random.seed(semilla)
            
            # Ejecutamos el algoritmo
            peso_candidato = abc.ejecutar(RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, lmbda)
            
            # Evaluamos qué tan bueno es este candidato matemáticamente
            f_obj = MetricasFinancieras.funcion_objetivo(peso_candidato, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, lmbda)
            fitness_candidato = MetricasFinancieras.calcular_fitness(f_obj)
            
            # Nos quedamos con el mejor absoluto de las 10 corridas
            if fitness_candidato > mejor_fitness_lambda:
                mejor_fitness_lambda = fitness_candidato
                mejor_peso_lambda = peso_candidato
        
        # Una vez que tenemos el ganador definitivo para este lambda, calculamos sus métricas limpias
        retorno, riesgo = MetricasFinancieras.calcular_retorno_riesgo(mejor_peso_lambda, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS)
        
        # Guardamos el resultado en la lista final
        resultados.append({
            'lambda': lmbda,
            'retorno': retorno,
            'riesgo': riesgo,
            'pesos': mejor_peso_lambda
        })
    
    
    # Renderizamos la tabla en consola
    Visualizacion.formatear_resultados_tabla_4(resultados)
    
    # Exportamos CSV y Gráficos
    Visualizacion.guardar_resultados_csv(resultados, "resultados_tabla4.csv")
    Visualizacion.graficar_frontera_pareto(resultados, "figura_2_abc.png")

if __name__ == "__main__":
    evaluacionABC()