import sys
sys.path.insert(0, '/home/bartolome-augusto-riera/Documents/metaheuristicas/TP4')

from ejercicio_1 import ejercicio_1
from ejercicio_2 import ejercicio_2
from ejercicio_3 import ejercicio_3
from ejercicio_4 import ejercicio_4


def main():
    print("\nOPTIMIZACIÓN DE REDES NEURONALES CON ALGORITMOS GENÉTICOS")
    print("Dataset: Wine Recognition")
    print("\n")
    
    print("\n*** EJECUTANDO EJERCICIO 1 ***\n")
    ejercicio_1()
    
    print("\n*** EJECUTANDO EJERCICIO 2 ***\n")
    ejercicio_2()
    
    print("\n*** EJECUTANDO EJERCICIO 3 ***\n")
    ejercicio_3()
    
    print("\n*** EJECUTANDO EJERCICIO 4 ***\n")
    ejercicio_4()
    
    print("COMPLETADO")
    print("\nResultados guardados en la carpeta ./resultados:")
    print("  - resultados_ejercicio_1.csv")
    print("  - tabla_comparativa_ejercicio_2.csv")
    print("  - tabla_comparativa_ejercicio_3.csv")
    print("  - estadisticas_f1.csv")
    print("  - estadisticas_f2.csv")
    print("  - resultados_corridas_f1.csv")
    print("  - resultados_corridas_f2.csv")
    print("  - Gráficos: matriz_confusion_*.png, evolucion_fitness_*.png, boxplot_*.png")


if __name__ == "__main__":
    main()
