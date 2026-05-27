import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from .operadores_geneticos import OperadoresGeneticos


class AlgoritmoGeneticoF1:
    
    def __init__(self, cantidad_caracteristicas, cantidad_clases, 
                 tamaño_poblacion=20, generaciones=30, 
                 probabilidad_mutacion=0.05, tamaño_torneo=2,
                 alfa=1.0, beta=0.0, seed=42):
        self.cantidad_caracteristicas = cantidad_caracteristicas
        self.cantidad_clases = cantidad_clases
        self.tamaño_poblacion = tamaño_poblacion
        self.generaciones = generaciones
        self.probabilidad_mutacion = probabilidad_mutacion
        self.tamaño_torneo = tamaño_torneo
        self.alfa = alfa
        self.beta = beta
        self.seed = seed
        
        np.random.seed(seed)
        
        self.mejor_cromosoma = None
        self.mejor_fitness = -np.inf
        self.historial_mejor_fitness = []
        self.historial_fitness_promedio = []
        self.memoria_fitness = {}
    
    def codificar_arquitectura(self, num_neuronas_capa2=0, num_neuronas_capa3=0, 
                               activa_capa2=0, activa_capa3=0):
        """
        Codifica una arquitectura en cromosoma de 23 bits:
        - Bits 0-6: neuronas capa 2 (0-127)
        - Bits 7: activa/desactiva capa 2
        - Bits 8-14: neuronas capa 3 (0-127)
        - Bits 15: activa/desactiva capa 3
        """
        cromosoma = np.zeros(23, dtype=int)
        
        bits_capa2 = format(num_neuronas_capa2 % 128, '07b')
        for i, bit in enumerate(bits_capa2):
            cromosoma[i] = int(bit)
        cromosoma[7] = activa_capa2
        
        bits_capa3 = format(num_neuronas_capa3 % 128, '07b')
        for i, bit in enumerate(bits_capa3):
            cromosoma[8 + i] = int(bit)
        cromosoma[15] = activa_capa3
        
        return cromosoma
    
    def decodificar_arquitectura(self, cromosoma):
        """Decodifica cromosoma a arquitectura (tupla de capas ocultas)"""
        # Capa 1: Bits 0 a 6 (Siempre activa)
        bits_capa1 = ''.join(map(str, cromosoma[0:7]))
        num_neuronas_capa1 = int(bits_capa1, 2) + 1  # Según TP: h_l = bin(...) + 1
        
        # Capa 2: Bit 7 (activación) y Bits 8 a 14 (neuronas)
        activa_capa2 = cromosoma[7]
        bits_capa2 = ''.join(map(str, cromosoma[8:15]))
        num_neuronas_capa2 = int(bits_capa2, 2) + 1
        
        # Capa 3: Bit 15 (activación) y Bits 16 a 22 (neuronas)
        activa_capa3 = cromosoma[15]
        bits_capa3 = ''.join(map(str, cromosoma[16:23]))
        num_neuronas_capa3 = int(bits_capa3, 2) + 1
        
        # Armamos la arquitectura
        capas = [num_neuronas_capa1] # La capa 1 está siempre presente
        
        if activa_capa2 == 1:
            capas.append(num_neuronas_capa2)
        if activa_capa3 == 1:
            capas.append(num_neuronas_capa3)
            
        return tuple(capas)
    
    def crear_poblacion_inicial(self):
        """Crea población inicial aleatoria"""
        poblacion = []
        for _ in range(self.tamaño_poblacion):
            cromosoma = np.random.randint(0, 2, size=23)
            poblacion.append(cromosoma)
        return poblacion
    
    def calcular_fitness(self, cromosoma, X_entrenamiento, y_entrenamiento):
        """
        Calcula fitness F1(I) = α * AccuracyCV + β * (1 - P(I)/Pmax)
        """
        # Chequeo de Caché: Si ya lo calculamos antes, lo devolvemos directo
        llave_cromosoma = str(cromosoma)
        if llave_cromosoma in self.memoria_fitness:
            return self.memoria_fitness[llave_cromosoma]

        try:
            arquitectura = self.decodificar_arquitectura(cromosoma)
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            accuracies = []
            
            for train_idx, val_idx in skf.split(X_entrenamiento, y_entrenamiento):
                X_train = X_entrenamiento[train_idx]
                y_train = y_entrenamiento[train_idx]
                X_val = X_entrenamiento[val_idx]
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
            
            cantidad_parametros = self._calcular_parametros_arquitectura(arquitectura)
            cantidad_parametros_max = 35203 # Según TP Pmax es 35203
            
            factor_complejidad = 1.0 - (cantidad_parametros / cantidad_parametros_max)
            
            fitness = (self.alfa * accuracy_cv) + (self.beta * factor_complejidad)
            
            # Guardamos en la memoria antes de retornar
            self.memoria_fitness[llave_cromosoma] = fitness
            return fitness
            
        except Exception:
            return -np.inf
    
    def _calcular_parametros_arquitectura(self, arquitectura):
        """Calcula cantidad de parámetros para una arquitectura"""
        parametros = 0
        neuronas_anteriores = self.cantidad_caracteristicas
        
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
                
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < self.tamaño_poblacion:
                    nueva_poblacion.append(hijo2)
            
            poblacion = nueva_poblacion[:self.tamaño_poblacion]
        
        return self.mejor_cromosoma, self.mejor_fitness
    
    def obtener_mejor_arquitectura(self):
        """Retorna la mejor arquitectura encontrada"""
        if self.mejor_cromosoma is None:
            return None
        return self.decodificar_arquitectura(self.mejor_cromosoma)
    
    def obtener_historial(self):
        """Retorna historial de ejecución"""
        return {
            'mejor_fitness': self.historial_mejor_fitness,
            'fitness_promedio': self.historial_fitness_promedio
        }
