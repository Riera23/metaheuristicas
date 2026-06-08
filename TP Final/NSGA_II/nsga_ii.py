import numpy as np

class NSGAII:
    def __init__(self, n_activos, retornos_esperados, matriz_covarianza, 
                 tam_poblacion=100, n_generaciones=200, 
                 prob_crossover=0.4, prob_mutacion=0.001,
                 eta_c=20, eta_m=20):
        self.n_activos = n_activos
        self.retornos = retornos_esperados
        self.covarianza = matriz_covarianza
        self.tam_poblacion = tam_poblacion
        self.n_generaciones = n_generaciones
        self.pc = prob_crossover
        self.pm = prob_mutacion
        self.eta_c = eta_c # Índice de distribución para SBX. Si es alto los hijos se parecen más a los padres
        self.eta_m = eta_m # Índice de distribución para Mutación Polinomial.

    # normalizar
    def reparar_solucion(self, x):
        """Garantiza que las proporciones sean >= 0 y sumen 1."""
        x = np.maximum(x, 0)
        suma = np.sum(x)
        if suma == 0:
            x = np.ones(self.n_activos) / self.n_activos
        else:
            x = x / suma
        return x

    # calcular fitness
    def calcular_objetivos(self, x):
        """Calcula f1 (riesgo) y f2 (retorno negativo)."""
        f1 = np.dot(x.T, np.dot(self.covarianza, x))  # minimizar riesgo
        f2 = -np.sum(x * self.retornos)               # maximizar retorno --> minimizar negativo de retorno
        return np.array([f1, f2])

    def domina(self, obj1, obj2):
        """Verifica si la solución 1 domina a la solución 2."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def ordenamiento_no_dominado(self, objetivos):
        """Realiza el Fast Non-dominated Sort."""
        n = objetivos.shape[0]
        dominados_por = [[] for _ in range(n)]
        contador_dominancia = np.zeros(n)
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self.domina(objetivos[p], objetivos[q]):
                    dominados_por[p].append(q)
                elif self.domina(objetivos[q], objetivos[p]):
                    contador_dominancia[p] += 1
            
            if contador_dominancia[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            siguiente_front = []
            for p in fronts[i]:
                for q in dominados_por[p]:
                    contador_dominancia[q] -= 1
                    if contador_dominancia[q] == 0:
                        siguiente_front.append(q)
            i += 1
            fronts.append(siguiente_front)
        
        return fronts[:-1]

    def calcular_distancia_crowding(self, objetivos, indices):
        """Calcula la distancia de crowding para un frente específico."""
        n = len(indices)
        distancias = np.zeros(n)
        if n <= 2:
            distancias[:] = np.inf
            return distancias

        for m in range(2): # 2 objetivos
            valores_obj = objetivos[indices, m]
            idx_ordenados = np.argsort(valores_obj)
            
            distancias[idx_ordenados[0]] = np.inf
            distancias[idx_ordenados[-1]] = np.inf
            
            rango = np.max(valores_obj) - np.min(valores_obj)
            if rango == 0: continue
            
            for i in range(1, n - 1):
                distancias[idx_ordenados[i]] += (valores_obj[idx_ordenados[i+1]] - valores_obj[idx_ordenados[i-1]]) / rango
        
        return distancias

    def seleccion_torneo(self, poblacion, objetivos, fronts, distancias):
        """Selección por torneo binario considerando rango y distancia."""
        idx1, idx2 = np.random.choice(len(poblacion), 2, replace=False)
        
        # Encontrar rango (frente) de cada uno
        rango1 = next(i for i, f in enumerate(fronts) if idx1 in f)
        rango2 = next(i for i, f in enumerate(fronts) if idx2 in f)
        
        if rango1 < rango2:
            return poblacion[idx1]
        elif rango2 < rango1:
            return poblacion[idx2]
        else:
            # Si están en el mismo frente, desempatar por distancia de crowding
            if distancias[idx1] > distancias[idx2]:
                return poblacion[idx1]
            else:
                return poblacion[idx2]

    def crossover_sbx(self, p1, p2):
        """Simulated Binary Crossover."""
        if np.random.rand() > self.pc:
            return p1.copy(), p2.copy()
        
        h1, h2 = np.zeros_like(p1), np.zeros_like(p2)
        for i in range(self.n_activos):
            if np.random.rand() <= 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (self.eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1))
                
                h1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                h2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            else:
                h1[i], h2[i] = p1[i], p2[i]
        return h1, h2

    def mutacion_polinomial(self, x):
        """Mutación Polinomial."""
        for i in range(self.n_activos):
            if np.random.rand() <= self.pm:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (self.eta_m + 1))
                
                x[i] = x[i] + delta # Asumiendo límites [0, 1], la perturbación es directa
        return x

    def ejecutar(self):
        """Ejecución principal del algoritmo NSGA-II."""
        # Inicialización
        poblacion = np.random.rand(self.tam_poblacion, self.n_activos)
        poblacion = np.array([self.reparar_solucion(ind) for ind in poblacion])
        
        for gen in range(self.n_generaciones):
            # Evaluar población actual
            objetivos = np.array([self.calcular_objetivos(ind) for ind in poblacion])
            
            # Ranking y Distancia
            fronts = self.ordenamiento_no_dominado(objetivos)
            distancias_totales = np.zeros(self.tam_poblacion)
            for f in fronts:
                distancias_frente = self.calcular_distancia_crowding(objetivos, f)
                for i, idx_pob in enumerate(f):
                    distancias_totales[idx_pob] = distancias_frente[i]
            
            # Crear descendencia (Qt)
            descendencia = []
            while len(descendencia) < self.tam_poblacion:
                p1 = self.seleccion_torneo(poblacion, objetivos, fronts, distancias_totales)
                p2 = self.seleccion_torneo(poblacion, objetivos, fronts, distancias_totales)
                
                h1, h2 = self.crossover_sbx(p1, p2)
                h1 = self.mutacion_polinomial(h1)
                h2 = self.mutacion_polinomial(h2)
                
                descendencia.append(self.reparar_solucion(h1))
                if len(descendencia) < self.tam_poblacion:
                    descendencia.append(self.reparar_solucion(h2))
            
            descendencia = np.array(descendencia)
            objetivos_desc = np.array([self.calcular_objetivos(ind) for ind in descendencia])
            
            # Combinar P_t y Q_t (Rt)
            poblacion_combinada = np.vstack((poblacion, descendencia))
            objetivos_combinados = np.vstack((objetivos, objetivos_desc))
            
            # Nuevo ordenamiento de Rt
            fronts_combinados = self.ordenamiento_no_dominado(objetivos_combinados)
            
            # Selección para la siguiente generación P_{t+1}
            nueva_poblacion = []
            indices_seleccionados = []
            
            for f in fronts_combinados:
                if len(nueva_poblacion) + len(f) <= self.tam_poblacion:
                    nueva_poblacion.extend(poblacion_combinada[f])
                    indices_seleccionados.extend(f)
                else:
                    # Llenar lo que falta usando distancia de crowding
                    distancias_f = self.calcular_distancia_crowding(objetivos_combinados, f)
                    idx_ordenados_dist = np.argsort(distancias_f)[::-1] # Mayor distancia primero
                    
                    espacio_restante = self.tam_poblacion - len(nueva_poblacion)
                    for i in range(espacio_restante):
                        idx_f = idx_ordenados_dist[i]
                        nueva_poblacion.append(poblacion_combinada[f[idx_f]])
                    break
            
            poblacion = np.array(nueva_poblacion)

        # Retornar el frente de Pareto final
        objetivos_finales = np.array([self.calcular_objetivos(ind) for ind in poblacion])
        fronts_finales = self.ordenamiento_no_dominado(objetivos_finales)
        pareto_front_indices = fronts_finales[0]
        
        return poblacion[pareto_front_indices], objetivos_finales[pareto_front_indices]
