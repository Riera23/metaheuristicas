import numpy as np
from sklearn.neural_network import MLPClassifier


class ModeloMLP:
    
    def __init__(self, capas_ocultas=(16, 8), activacion="tanh", 
                 solver="sgd", tasa_aprendizaje=0.01, iteraciones_max=300, 
                 random_state=42):
        self.capas_ocultas = capas_ocultas
        self.activacion = activacion
        self.solver = solver
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones_max = iteraciones_max
        self.random_state = random_state
        self.modelo = None
        self.cantidad_parametros = None
    
    def crear_modelo(self):
        self.modelo = MLPClassifier(
            hidden_layer_sizes=self.capas_ocultas,
            activation=self.activacion,
            solver=self.solver,
            learning_rate_init=self.tasa_aprendizaje,
            max_iter=self.iteraciones_max,
            random_state=self.random_state
        )
        return self.modelo
    
    def entrenar(self, X_entrenamiento, y_entrenamiento):
        if self.modelo is None:
            self.crear_modelo()
        self.modelo.fit(X_entrenamiento, y_entrenamiento)
        self._calcular_parametros(X_entrenamiento.shape[1])
    
    def predecir(self, X):
        return self.modelo.predict(X)
    
    def puntuacion(self, X, y):
        return self.modelo.score(X, y)
    
    def _calcular_parametros(self, cantidad_caracteristicas_entrada):
        """Calcula analíticamente la cantidad de parámetros entrenables"""
        cantidad_parametros = 0
        neuronas_anteriores = cantidad_caracteristicas_entrada
        
        for neuronas_capa in self.capas_ocultas:
            cantidad_parametros += (neuronas_anteriores * neuronas_capa) + neuronas_capa
            neuronas_anteriores = neuronas_capa
        
        cantidad_clases = len(self.modelo.classes_)
        cantidad_parametros += (neuronas_anteriores * cantidad_clases) + cantidad_clases
        
        self.cantidad_parametros = cantidad_parametros
        return cantidad_parametros
    
    def obtener_cantidad_parametros(self):
        return self.cantidad_parametros
    
    def obtener_pesos(self):
        return self.modelo.coefs_
    
    def obtener_sesgos(self):
        return self.modelo.intercepts_
