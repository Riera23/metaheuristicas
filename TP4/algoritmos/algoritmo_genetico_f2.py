import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from .operadores_geneticos import OperadoresGeneticos


class AlgoritmoGeneticoF2:
    
    def __init__(self, cantidad_caracteristicas, cantidad_clases, 
                 tamaño_poblacion=20, generaciones=30, 
                 probabilidad_mutacion=0.05, tamaño_torneo=2,
                 alfa=0.6, beta=0.2, gamma=0.2, seed=42):
        self.cantidad_caracteristicas = cantidad_caracteristicas
        self.cantidad_clases = cantidad_clases
        self.tamaño_poblacion = tamaño_poblacion
        self.generaciones = generaciones
        self.probabilidad_mutacion = probabilidad_mutacion
        self.tamaño_torneo = tamaño_torneo
        self.alfa = alfa
        self.beta = beta
        self.gamma = gamma
        self.seed = seed
        
        np.random.seed(seed)
        
        self.mejor_cromosoma = None
        self.mejor_fitness = -np.inf
        self.historial_mejor_fitness = []
        self.historial_fitness_promedio = []
        self.memoria_fitness = {} 
    
    def codificar_cromosoma(self, caracteristicas_activas, num_neuronas_capa2=0, 
                           num_neuronas_capa3=0, activa_capa2=0, activa_capa3=0):
        """
        Codifica cromosoma de 36 bits:
        - Bits 0-12: selección de 13 características
        - Bits 13-35: arquitectura (23 bits)
        """
        cromosoma = np.zeros(36, dtype=int)
        
        for i in range(len(caracteristicas_activas)):
            cromosoma[i] = caracteristicas_activas[i]
        
        bits_capa2 = format(num_neuronas_capa2 % 128, '07b')
        for i, bit in enumerate(bits_capa2):
            cromosoma[13 + i] = int(bit)
        cromosoma[20] = activa_capa2
        
        bits_capa3 = format(num_neuronas_capa3 % 128, '07b')
        for i, bit in enumerate(bits_capa3):
            cromosoma[21 + i] = int(bit)
        cromosoma[28] = activa_capa3
        
        return cromosoma
    
    def decodificar_cromosoma(self, cromosoma):
        """Decodifica cromosoma en características y arquitectura"""
        # Máscara de características (13 bits)
        caracteristicas_activas = cromosoma[0:13].astype(bool)
        
        # Capa 1: Bits 13 a 19 (siempre activa)
        bits_capa1 = ''.join(map(str, cromosoma[13:20]))
        num_neuronas_capa1 = int(bits_capa1, 2) + 1
        
        # Capa 2: Bit 20 (activación) y Bits 21 a 27 (neuronas)
        activa_capa2 = cromosoma[20]
        bits_capa2 = ''.join(map(str, cromosoma[21:28]))
        num_neuronas_capa2 = int(bits_capa2, 2) + 1
        
        # Capa 3: Bit 28 (activación) y Bits 29 a 35 (neuronas)
        activa_capa3 = cromosoma[28]
        bits_capa3 = ''.join(map(str, cromosoma[29:36]))
        num_neuronas_capa3 = int(bits_capa3, 2) + 1
        
        # Armamos la arquitectura
        capas = [num_neuronas_capa1]
        
        if activa_capa2 == 1:
            capas.append(num_neuronas_capa2)
        if activa_capa3 == 1:
            capas.append(num_neuronas_capa3)
            
        return caracteristicas_activas, tuple(capas)
    
    def crear_poblacion_inicial(self):
        """Crea población inicial aleatoria"""
        poblacion = []
        for _ in range(self.tamaño_poblacion):
            cromosoma = np.random.randint(0, 2, size=36)
            if np.sum(cromosoma[0:13]) == 0:
                cromosoma[np.random.randint(0, 13)] = 1
            poblacion.append(cromosoma)
        return poblacion
    
    def calcular_fitness(self, cromosoma, X_entrenamiento, y_entrenamiento):
        """
        Calcula fitness F2(I) = α*AccuracyCV + β*(1-P(I)/Pmax) + γ*(1-K/N)
        """
        # Chequeo de Caché
        llave_cromosoma = str(cromosoma)
        if llave_cromosoma in self.memoria_fitness:
            return self.memoria_fitness[llave_cromosoma]

        try:
            caracteristicas_activas, arquitectura = self.decodificar_cromosoma(cromosoma)
            
            if np.sum(caracteristicas_activas) == 0 or len(arquitectura) < 2:
                return -np.inf
            
            cantidad_caracteristicas_seleccionadas = np.sum(caracteristicas_activas)
            
            if cantidad_caracteristicas_seleccionadas < 1:
                return -np.inf
            
            X_filtrado = X_entrenamiento[:, caracteristicas_activas]
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            accuracies = []
            
            for train_idx, val_idx in skf.split(X_filtrado, y_entrenamiento):
                X_train = X_filtrado[train_idx]
                y_train = y_entrenamiento[train_idx]
                X_val = X_filtrado[val_idx]
                y_val = y_entrenamiento[val_idx]
                
                modelo = MLPClassifier(
                    hidden_layer_sizes=arquitectura,
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    max_iter=300,
                    random_state=self.seed
                )
                modelo.fit(X_train, y_train)
                score = modelo.score(X_val, y_val)
                accuracies.append(score)
            
            accuracy_cv = np.mean(accuracies)
            
            cantidad_parametros = self._calcular_parametros_arquitectura(
                arquitectura, cantidad_caracteristicas_seleccionadas
            )
            # El TP fija P_max en 35203
            cantidad_parametros_max = 35203
            
            factor_complejidad = 1.0 - (cantidad_parametros / cantidad_parametros_max)
            factor_reduccion = 1.0 - (cantidad_caracteristicas_seleccionadas / self.cantidad_caracteristicas)
            
            fitness = (self.alfa * accuracy_cv) + \
                      (self.beta * factor_complejidad) + \
                      (self.gamma * factor_reduccion)
            
            # Guardamos en caché
            self.memoria_fitness[llave_cromosoma] = fitness
            return fitness
            
        except Exception:
            return -np.inf
    
    def _calcular_parametros_arquitectura(self, arquitectura, cantidad_caracteristicas_entrada):
        """Calcula cantidad de parámetros para una arquitectura"""
        parametros = 0
        neuronas_anteriores = cantidad_caracteristicas_entrada
        
        for neuronas in arquitectura:
            parametros += (neuronas_anteriores * neuronas) + neuronas
            neuronas_anteriores = neuronas
        
        parametros += (neuronas_anteriores * self.cantidad_clases) + self.cantidad_clases
        return parametros
    
    def ejecutar(self, X_entrenamiento, y_entrenamiento):
        """Ejecuta el algoritmo genético completo"""
        poblacion = self.crear_poblacion_inicial()
        
        for generacion in range(self.generaciones):
            fitness_valores = np.array([
                self.calcular_fitness(cromosoma, X_entrenamiento, y_entrenamiento)
                for cromosoma in poblacion
            ])
            
            mejor_idx = np.argmax(fitness_valores)
            mejor_fitness_generacion = fitness_valores[mejor_idx]
            
            if mejor_fitness_generacion > self.mejor_fitness:
                self.mejor_fitness = mejor_fitness_generacion
                self.mejor_cromosoma = poblacion[mejor_idx].copy()
            
            self.historial_mejor_fitness.append(self.mejor_fitness)
            self.historial_fitness_promedio.append(np.mean(fitness_valores))
            
            nueva_poblacion = []
            
            indice_elite = np.argmax(fitness_valores)
            nueva_poblacion.append(poblacion[indice_elite].copy())
            
            while len(nueva_poblacion) < self.tamaño_poblacion:
                padre1, padre2 = OperadoresGeneticos.seleccionar_padres(
                    poblacion, fitness_valores, self.tamaño_torneo
                )
                
                hijo1, hijo2 = OperadoresGeneticos.crossover_un_punto(padre1, padre2)
                
                hijo1 = OperadoresGeneticos.mutacion_binaria(hijo1, self.probabilidad_mutacion)
                hijo2 = OperadoresGeneticos.mutacion_binaria(hijo2, self.probabilidad_mutacion)
                
                if np.sum(hijo1[0:13]) == 0:
                    hijo1[np.random.randint(0, 13)] = 1
                if np.sum(hijo2[0:13]) == 0:
                    hijo2[np.random.randint(0, 13)] = 1
                
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < self.tamaño_poblacion:
                    nueva_poblacion.append(hijo2)
            
            poblacion = nueva_poblacion[:self.tamaño_poblacion]
        
        return self.mejor_cromosoma, self.mejor_fitness
    
    def obtener_mejor_caracteristicas_y_arquitectura(self):
        """Retorna características y arquitectura óptimas"""
        if self.mejor_cromosoma is None:
            return None, None
        return self.decodificar_cromosoma(self.mejor_cromosoma)
    
    def obtener_historial(self):
        """Retorna historial de ejecución"""
        return {
            'mejor_fitness': self.historial_mejor_fitness,
            'fitness_promedio': self.historial_fitness_promedio
        }
