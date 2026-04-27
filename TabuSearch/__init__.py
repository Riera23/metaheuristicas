#-----------------------------------------------------------------------------------------------
# Algoritmo Tabú Search para optimización de funciones unimodales en 1D

# ESTA FUNCION NO TIENE CRITERIO DE DIVERSIFICACIÓN NI INTESIFICACIÓN
# ADEMÁS TIENE COMO CRITERIO DE PARADA QUE SI TODOS LOS VECINOS SON TABÚ, SE DETIENE EL ALGORITMO
# Parámetros:
# - f_objetivo: función a minimizar (unimodal)
# - k_tenure: longitud de la memoria tabú
# - dominio: intervalo [a, b] donde se busca el mínimo
# - N_vecinos: número de vecinos generados en cada iteración
# - N_max: número máximo de iteraciones
# - sigma: desviación estándar para la generación de vecinos
# - tabu_tol: tolerancia para considerar un vecino como tabú

import numpy as np
def tabu_search(f_objetivo,k_tenure, dominio, N_vecinos, N_max=1000, sigma=50, tabu_tol=1.0):
    # Inicialización aleatoria
    x_actual = np.random.uniform(dominio[0], dominio[1])
    f_actual = f_objetivo(x_actual)
    
    x_best = x_actual
    f_best = f_actual
    best_iteration = 0
    
    tabu_list = []
    
    # Métricas solicitadas
    convergence = []
    tabu_ratios = []
    evaluations = 1 # Evaluaciones de la funcion objetivo
    iteration = 0
    stop = False
    
    while iteration < N_max and not stop:
        vecinos = x_actual + np.random.normal(0, sigma, N_vecinos) # generamos 20 vecinos del x_ctual
        vecinos = np.clip(vecinos, dominio[0], dominio[1]) # para estar dentro del dominio
        
        f_vecinos = f_objetivo(vecinos)
        evaluations += N_vecinos
        
        # Chequeamos cuantos y que porcentaje de vecinos son tabu
        vecino_es_tabu = np.zeros(N_vecinos, dtype=bool)
        if len(tabu_list) > 0:
            for i, v in enumerate(vecinos):
                if np.any(np.abs(np.array(tabu_list) - v) < tabu_tol):
                    vecino_es_tabu[i] = True
        tabu_ratios.append(np.mean(vecino_es_tabu))
        
        # Selección del mejor vecino (que no sea tabú, o que cumpla criterio de aspiración)
        mejor_vecino_idx = -1
        mejor_vecino_f = float('inf')
        for i in range(N_vecinos):
            if not vecino_es_tabu[i]:  #or f_vecinos[i] < f_best (criterio de aspiración)
                if f_vecinos[i] < mejor_vecino_f:
                    mejor_vecino_f = f_vecinos[i]
                    mejor_vecino_idx = i           
        # Si todos los vecinos son tabú, se detiene el algoritmo
        if mejor_vecino_idx == -1:
            stop = True
            # OPCION 2: Si TODOS los vecinos son tabú elegimos el mejor absoluto para no estancarnos:
            #mejor_vecino_idx = np.argmin(f_vecinos)
            #mejor_vecino_f = f_vecinos[mejor_vecino_idx]
            
        # Actualizamos el estado actual
        x_actual = vecinos[mejor_vecino_idx]
        f_actual = mejor_vecino_f
        
        # Actualizamos la memoria Tabú
        tabu_list.append(x_actual)
        if len(tabu_list) > k_tenure:
            tabu_list.pop(0) # Eliminamos el más antiguo => FIFO o cola
            
        # Actualizamos el mejor global
        if f_actual < f_best:
            f_best = f_actual
            x_best = x_actual
            best_iteration = iteration + 1

        convergence.append(f_best)

        iteration += 1
        
    return {
        'k (Tenure)': k_tenure,
        'Mejor Fitness': f_best,
        'Posición x': x_best,
        'Iteración Mejor': best_iteration,
        'Evaluaciones': evaluations,
        'Proporción Tabú [%]': round(np.mean(tabu_ratios) * 100, 2),
        'Convergencia': convergence
    }

#-----------------------------------------------------------------------------------------------
# Algoritmo Tabú Search