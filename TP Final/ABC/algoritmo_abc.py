import random
import numpy as np
from utils.metricas import MetricasFinancieras

class AlgoritmoABC:
    
    def __init__(self, n_activos, tamano_poblacion=20, max_ciclos=500, limite=50, 
                 peso_minimo=0.0, peso_maximo=1.0):
        self.n_activos = n_activos
        self.P_tamano = tamano_poblacion
        self.Nmax = max_ciclos
        self.limite = limite
        self.peso_minimo = peso_minimo
        self.peso_maximo = peso_maximo

    def _inicializar_poblacion(self, retornos, covarianzas, lmbda):
        # Crea la población inicial de fuentes de alimento (portafolios)
        portfolios = np.zeros((self.P_tamano, self.n_activos))
        fitness = np.zeros(self.P_tamano)
        paciencia = np.zeros(self.P_tamano)

        for i in range(self.P_tamano):
            portfolios[i] = np.random.uniform(self.peso_minimo, self.peso_maximo, self.n_activos)
            suma_pesos = sum(portfolios[i])
            if suma_pesos > 0: 
                portfolios[i] = portfolios[i] / suma_pesos
            
            f_obj = MetricasFinancieras.funcion_objetivo(portfolios[i], retornos, covarianzas, lmbda)
            fitness[i] = MetricasFinancieras.calcular_fitness(f_obj)
            
        return portfolios, fitness, paciencia

    def ejecutar(self, retornos, covarianzas, lmbda):
        """
        Ejecuta las 3 fases del algoritmo ABC y retorna el mejor portafolio encontrado.
        """
        portfolios, fitness, paciencia = self._inicializar_poblacion(retornos, covarianzas, lmbda)
        
        mejor_solucion = np.copy(portfolios[0])
        mejor_fitness = fitness[0]

        for ciclo in range(self.Nmax):
            
            # --- FASE 1: Abejas Empleadas ---
            for i in range(self.P_tamano):
                j = random.choice([x for x in range(self.P_tamano) if x != i])
                k = random.randint(0, self.n_activos - 1)
                phi = random.uniform(-1, 1)
                
                pesos_i = np.copy(portfolios[i])
                pesos_i[k] = portfolios[i][k] + phi * (portfolios[i][k] - portfolios[j][k])
                pesos_i[k] = np.clip(pesos_i[k], self.peso_minimo, self.peso_maximo)
                
                if sum(pesos_i) > 0: 
                    pesos_i = pesos_i / sum(pesos_i) 
                
                f_obj = MetricasFinancieras.funcion_objetivo(pesos_i, retornos, covarianzas, lmbda)
                fit_i = MetricasFinancieras.calcular_fitness(f_obj)
                
                if fit_i > fitness[i]:
                    portfolios[i], fitness[i], paciencia[i] = pesos_i, fit_i, 0
                else: 
                    paciencia[i] += 1
                    
            # FASE 2: Abejas Observadoras
            probabilidades = fitness / sum(fitness)
            t = 0
            i = 0
            while t < self.P_tamano:
                if random.random() < probabilidades[i]:
                    t += 1
                    j = random.choice([x for x in range(self.P_tamano) if x != i])
                    k = random.randint(0, self.n_activos - 1)
                    phi = random.uniform(-1, 1)
                    
                    pesos_i = np.copy(portfolios[i])
                    pesos_i[k] = portfolios[i][k] + phi * (portfolios[i][k] - portfolios[j][k])
                    pesos_i[k] = np.clip(pesos_i[k], self.peso_minimo, self.peso_maximo)
                    
                    if sum(pesos_i) > 0: 
                        pesos_i = pesos_i / sum(pesos_i)
                    
                    f_obj = MetricasFinancieras.funcion_objetivo(pesos_i, retornos, covarianzas, lmbda)
                    fit_i = MetricasFinancieras.calcular_fitness(f_obj)
                    
                    if fit_i > fitness[i]:
                        portfolios[i], fitness[i], paciencia[i] = pesos_i, fit_i, 0
                    else: 
                        paciencia[i] += 1
                i = (i + 1) % self.P_tamano
                
            # Etilismo
            for i in range(self.P_tamano):
                if fitness[i] > mejor_fitness:
                    mejor_fitness, mejor_solucion = fitness[i], np.copy(portfolios[i])
                    
            # FASE 3: Abejas Exploradoras
            for i in range(self.P_tamano):
                if paciencia[i] >= self.limite:
                    portfolios[i] = np.random.uniform(self.peso_minimo, self.peso_maximo, self.n_activos)
                    if sum(portfolios[i]) > 0: 
                        portfolios[i] = portfolios[i] / sum(portfolios[i])
                    
                    f_obj = MetricasFinancieras.funcion_objetivo(portfolios[i], retornos, covarianzas, lmbda)
                    fitness[i] = MetricasFinancieras.calcular_fitness(f_obj)
                    paciencia[i] = 0

        return mejor_solucion