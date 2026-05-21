import numpy as np


class OperadoresGeneticos:
    
    @staticmethod
    def seleccion_torneo(poblacion, fitness_valores, tamaño_torneo=2):
        """
        Selecciona un individuo mediante torneo.
        Población es lista de cromosomas (arrays), fitness_valores son scores.
        """
        tamano_poblacion = len(poblacion)
        indices_torneo = np.random.choice(tamano_poblacion, tamaño_torneo, replace=False)
        indice_ganador = indices_torneo[np.argmax(fitness_valores[indices_torneo])]
        return poblacion[indice_ganador].copy()
    
    @staticmethod
    def seleccionar_padres(poblacion, fitness_valores, tamaño_torneo=2):
        """Selecciona dos padres mediante torneo"""
        padre1 = OperadoresGeneticos.seleccion_torneo(poblacion, fitness_valores, tamaño_torneo)
        padre2 = OperadoresGeneticos.seleccion_torneo(poblacion, fitness_valores, tamaño_torneo)
        return padre1, padre2
    
    @staticmethod
    def crossover_un_punto(padre1, padre2):
        """Crossover de un punto"""
        longitud = len(padre1)
        punto_corte = np.random.randint(1, longitud)
        
        hijo1 = np.concatenate([padre1[:punto_corte], padre2[punto_corte:]])
        hijo2 = np.concatenate([padre2[:punto_corte], padre1[punto_corte:]])
        
        return hijo1, hijo2
    
    @staticmethod
    def mutacion_binaria(cromosoma, probabilidad_mutacion=0.01):
        """Mutación binaria: invierte bits con probabilidad dada"""
        cromosoma_mutado = cromosoma.copy()
        for i in range(len(cromosoma_mutado)):
            if np.random.random() < probabilidad_mutacion:
                cromosoma_mutado[i] = 1 - cromosoma_mutado[i]
        return cromosoma_mutado
