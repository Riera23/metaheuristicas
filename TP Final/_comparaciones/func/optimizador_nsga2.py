import numpy as np
try:
    from .optimizador_base import OptimizadorBase
except ImportError:
    from optimizador_base import OptimizadorBase
from NSGA_II.nsga_ii import NSGAII

class OptimizadorNSGAII(OptimizadorBase):
    def __init__(self, n_activos, retornos, covarianzas, espacio_busqueda, n_iteraciones=10):
        super().__init__(espacio_busqueda, n_iteraciones)
        self.n_activos = n_activos
        self.retornos = retornos
        self.covarianzas = covarianzas

    def evaluar_configuracion(self, config):
        # Instanciar el algoritmo con la configuración de hiperparámetros
        algoritmo = NSGAII(
            n_activos=self.n_activos,
            retornos_esperados=self.retornos,
            matriz_covarianza=self.covarianzas,
            **config
        )
        
        # Ejecutar el algoritmo para obtener el frente de Pareto final
        _, objetivos = algoritmo.ejecutar()
        
        # Extraer métricas: objetivos[:, 0] es varianza, objetivos[:, 1] es -retorno
        riesgos = np.clip(objetivos[:, 0], 1e-10, None)
        retornos = -objetivos[:, 1]
        
        # Métrica de calidad: Promedio del Ratio de Sharpe del frente
        sharpe_ratios = retornos / np.sqrt(riesgos)
        score = np.mean(sharpe_ratios)
        
        return float(score)