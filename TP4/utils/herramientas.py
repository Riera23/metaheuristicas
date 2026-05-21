import numpy as np
import pandas as pd


class Herramientas:
    
    @staticmethod
    def crear_dataframe_resultados(accuracy, cantidad_parametros, nombre_modelo="Modelo"):
        """Crea DataFrame con resultados"""
        datos = {
            'Modelo': [nombre_modelo],
            'Accuracy': [accuracy],
            'Parámetros Entrenables': [cantidad_parametros]
        }
        return pd.DataFrame(datos)
    
    @staticmethod
    def combinar_resultados(lista_dataframes):
        """Combina múltiples DataFrames en uno"""
        return pd.concat(lista_dataframes, ignore_index=True)
    
    @staticmethod
    def guardar_resultados_csv(dataframe, nombre_archivo):
        """Guarda DataFrame en CSV"""
        dataframe.to_csv(nombre_archivo, index=False)
    
    @staticmethod
    def crear_tabla_comparativa(resultados_beta):
        """Crea tabla comparativa para diferentes valores de β"""
        datos_tabla = []
        
        for beta, resultado in resultados_beta.items():
            datos_tabla.append({
                'β': beta,
                'α': 1.0 - beta,
                'Mejor Fitness': resultado['mejor_fitness'],
                'Arquitectura': str(resultado['arquitectura']),
                'Accuracy Test': resultado['accuracy_test'],
                'Parámetros': resultado['parametros']
            })
        
        return pd.DataFrame(datos_tabla)
