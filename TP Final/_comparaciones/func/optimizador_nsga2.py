import numpy as np
try:
    from .optimizador_base import OptimizadorBase
except ImportError:
    from optimizador_base import OptimizadorBase
from NSGA_II.nsga_ii import NSGAII
from func.funciones import calcular_hipervolumen_2d

class OptimizadorNSGAII(OptimizadorBase):
    def __init__(self, n_activos, retornos, covarianzas, espacio_busqueda, punto_ref, n_iteraciones=10):
        super().__init__(espacio_busqueda, n_iteraciones)
        self.n_activos = n_activos
        self.retornos = retornos
        self.covarianzas = covarianzas
        self.punto_ref = punto_ref  # Guardamos el punto de referencia para el HV

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
        
        # El score ahora es directamente el Hipervolumen del frente obtenido
        score = calcular_hipervolumen_2d(objetivos, self.punto_ref)
        
        return float(score)