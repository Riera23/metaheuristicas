import numpy as np

# DATOS REALES DEL PAPER (Bolsa de Comercio de Santiago)
N_ACTIVOS = 6

# datos de la tabla 1 del paper
RETORNOS_ESPERADOS = np.array([
    0.0002,   # COPEC
    0.0004,   # CTC-A
   -0.0003,   # CAP
   -0.00028,  # COLBUN
   -0.0001,   # ENDESA
    0.0001    # ENTEL
])

# datos de la tabla 2 del paper
MATRIZ_COVARIANZAS = np.array([
    # COPEC    CTC-A     CAP     COLBUN  ENDESA    ENTEL
    [0.00038, 0.00020, 0.00017, 0.00014, 0.00019, 0.00017], # COPEC
    [0.00020, 0.00043, 0.00015, 0.00013, 0.00021, 0.00014], # CTC-A
    [0.00017, 0.00015, 0.00034, 0.00011, 0.00014, 0.00014], # CAP
    [0.00014, 0.00013, 0.00011, 0.00044, 0.00014, 0.00011], # COLBUN
    [0.00019, 0.00021, 0.00014, 0.00014, 0.00040, 0.00014], # ENDESA
    [0.00017, 0.00014, 0.00014, 0.00011, 0.00014, 0.00046]  # ENTEL
])