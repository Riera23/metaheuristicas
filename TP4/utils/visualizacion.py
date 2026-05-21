import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Visualizacion:
    
    @staticmethod
    def graficar_matriz_confusion(y_verdadero, y_predicho, titulo="Matriz de Confusión"):
        """Genera gráfico de matriz de confusión"""
        cm = confusion_matrix(y_verdadero, y_predicho)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(titulo)
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def graficar_evolucion_fitness(historial_mejor, historial_promedio, 
                                   titulo="Evolución del Fitness"):
        """Grafica la evolución del fitness en el AG"""
        generaciones = range(len(historial_mejor))
        
        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, historial_mejor, label='Mejor Fitness', marker='o')
        plt.plot(generaciones, historial_promedio, label='Fitness Promedio', marker='s')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def graficar_comparacion_beta(resultados_beta):
        """Grafica comparación de fitness para diferentes valores de β"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for beta, datos in resultados_beta.items():
            historial = datos['historial']
            ax.plot(range(len(historial['mejor_fitness'])), 
                   historial['mejor_fitness'], 
                   label=f'β={beta}', marker='o')
        
        ax.set_xlabel('Generación')
        ax.set_ylabel('Mejor Fitness')
        ax.set_title('Comparación del Fitness Mejor para diferentes β')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def graficar_boxplot_comparacion(datos_f1, datos_f2, titulo="Comparación"):
        """Grafica boxplot comparativo de dos conjuntos de datos"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        datos_boxplot = [datos_f1, datos_f2]
        ax.boxplot(datos_boxplot, labels=['F1', 'F2'])
        
        ax.set_ylabel('Accuracy')
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def guardar_figura(fig, nombre_archivo):
        """Guarda figura en archivo"""
        fig.savefig(nombre_archivo, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def mostrar_figura(fig):
        """Muestra figura"""
        plt.show()
