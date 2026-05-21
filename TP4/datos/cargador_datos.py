import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CargadorDatos:
    
    def __init__(self, test_size=0.25, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.escalador = StandardScaler()
        self.X_entrenamiento = None
        self.X_prueba = None
        self.y_entrenamiento = None
        self.y_prueba = None
        self.nombres_caracteristicas = None
    
    def cargar_wine(self):
        datos = load_wine()
        X = datos.data
        y = datos.target
        self.nombres_caracteristicas = datos.feature_names
        
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        X_entrenamiento_escalado = self.escalador.fit_transform(X_entrenamiento)
        X_prueba_escalado = self.escalador.transform(X_prueba)
        
        self.X_entrenamiento = X_entrenamiento_escalado
        self.X_prueba = X_prueba_escalado
        self.y_entrenamiento = y_entrenamiento
        self.y_prueba = y_prueba
        
        return self.X_entrenamiento, self.X_prueba, self.y_entrenamiento, self.y_prueba
    
    def obtener_datos_entrenamiento(self):
        return self.X_entrenamiento, self.y_entrenamiento
    
    def obtener_datos_prueba(self):
        return self.X_prueba, self.y_prueba
    
    def obtener_cantidad_caracteristicas(self):
        return self.X_entrenamiento.shape[1]
    
    def obtener_cantidad_clases(self):
        return len(np.unique(self.y_entrenamiento))
