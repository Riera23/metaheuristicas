try:
    from .optimizador_base import OptimizadorBase
except ImportError:
    from optimizador_base import OptimizadorBase
from ABC.algoritmo_abc import AlgoritmoABC
from utils.metricas import MetricasFinancieras

class OptimizadorABC(OptimizadorBase):
    def __init__(self, n_activos, retornos, covarianzas, lmbda, espacio_busqueda, n_iteraciones=10):
        super().__init__(espacio_busqueda, n_iteraciones)
        self.n_activos = n_activos
        self.retornos = retornos
        self.covarianzas = covarianzas
        self.lmbda = lmbda

    def evaluar_configuracion(self, config):
        # Instanciar el algoritmo ABC
        algoritmo = AlgoritmoABC(
            n_activos=self.n_activos,
            **config
        )
        
        # Obtener el mejor portafolio
        mejor_solucion = algoritmo.ejecutar(self.retornos, self.covarianzas, self.lmbda)
        
        # Evaluar la calidad mediante la función de fitness original
        f_obj = MetricasFinancieras.funcion_objetivo(
            mejor_solucion, self.retornos, self.covarianzas, self.lmbda
        )
        score = MetricasFinancieras.calcular_fitness(f_obj)
        
        return float(score)