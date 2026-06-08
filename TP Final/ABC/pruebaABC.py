from datos_mercado import N_ACTIVOS, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS
from algoritmo_abc import AlgoritmoABC
from utils.metricas import MetricasFinancieras
from utils.visualizacion import Visualizacion

def pruebaABC():
    print("Iniciando optimización para un inversor moderado (λ = 0.5)...")
    
    # 1. Definir el nivel de aversión al riesgo
    lmbda = 0.5
    
    # 2. Instanciar el algoritmo ABC
    abc = AlgoritmoABC(n_activos=N_ACTIVOS, tamano_poblacion=20, max_ciclos=500, limite=50)
    
    # 3. Ejecutar la búsqueda
    mejor_peso = abc.ejecutar(RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS, lmbda)
    
    # 4. Calcular métricas finales limpias
    retorno, riesgo = MetricasFinancieras.calcular_retorno_riesgo(mejor_peso, RETORNOS_ESPERADOS, MATRIZ_COVARIANZAS)
    
    # 5. Empaquetar y visualizar
    resultado = [{
        'lambda': lmbda,
        'retorno': retorno,
        'riesgo': riesgo,
        'pesos': mejor_peso
    }]
    
    Visualizacion.formatear_resultados_tabla_4(resultado)