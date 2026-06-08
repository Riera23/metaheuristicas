import numpy as np

class MetricasFinancieras:
    
    @staticmethod
    def calcular_retorno_riesgo(pesos, retornos, covarianzas):
        """
        Calcula el retorno esperado y el riesgo (varianza) de un portafolio
        mediante operaciones matriciales.
        """
        retorno = np.dot(pesos, retornos)
        riesgo = np.dot(pesos.T, np.dot(covarianzas, pesos))
        return retorno, riesgo

    @staticmethod
    def funcion_objetivo(pesos, retornos, covarianzas, lmbda):
        """
        Calcula la función objetivo de Markowitz con relajación Lagrangiana.
        A minimizar: λ * Varianza - (1 - λ) * Retorno
        """
        retorno, riesgo = MetricasFinancieras.calcular_retorno_riesgo(pesos, retornos, covarianzas)
        return (lmbda * riesgo) - ((1 - lmbda) * retorno)

    @staticmethod
    def calcular_fitness(f_obj):
        """
        Convierte la función objetivo (que se busca minimizar) a un 
        valor de fitness (que el algoritmo ABC busca maximizar).
        """
        if f_obj >= 0:
            return 1 / (1 + f_obj)
        else:
            return 1 + abs(f_obj)