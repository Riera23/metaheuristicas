from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np


class Metricas:
    
    @staticmethod
    def calcular_accuracy(y_verdadero, y_predicho):
        """Calcula accuracy"""
        return accuracy_score(y_verdadero, y_predicho)
    
    @staticmethod
    def obtener_matriz_confusion(y_verdadero, y_predicho):
        """Obtiene matriz de confusión"""
        return confusion_matrix(y_verdadero, y_predicho)
    
    @staticmethod
    def obtener_reporte_clasificacion(y_verdadero, y_predicho):
        """Obtiene reporte de clasificación"""
        return classification_report(y_verdadero, y_predicho)
    
    @staticmethod
    def calcular_precisiones_clases(y_verdadero, y_predicho):
        """Calcula precisión, recall y f1 por clase"""
        reporte = classification_report(y_verdadero, y_predicho, output_dict=True)
        return reporte
