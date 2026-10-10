import numpy as np
import pandas as pd

# ==========================================
# 1. FUNCIÓN MATEMÁTICA (TMM)
# ==========================================
def calcular_transmitancia(grosores, indices, longitudes_onda, n_0=1.0, n_s=1.0):
    """
    Calcula el espectro de transmitancia de un filtro de capas delgadas usando 
    el Método de la Matriz de Transferencia.
    """
    T_espectro = np.zeros(len(longitudes_onda))
    
    for i, lam in enumerate(longitudes_onda):
        M_total = np.identity(2, dtype=complex)
        
        for d, n in zip(grosores, indices):
            delta = (2 * np.pi / lam) * n * d
            
            M_j = np.array([
                [np.cos(delta), (1j / n) * np.sin(delta)],
                [1j * n * np.sin(delta), np.cos(delta)]
            ])
            
            M_total = np.dot(M_total, M_j)
            
        m11, m12 = M_total[0, 0], M_total[0, 1]
        m21, m22 = M_total[1, 0], M_total[1, 1]
        
        numerador_t = 2 * n_0
        denominador_t = (n_0 * m11) + (n_0 * n_s * m12) + m21 + (n_s * m22)
        t = numerador_t / denominador_t
        
        T_espectro[i] = (n_s / n_0) * np.abs(t)**2
        
    return T_espectro

# ==========================================
# 2. PARÁMETROS DEL DATASET
# ==========================================
NUM_MUESTRAS = 100000
NUM_CAPAS = 64

GROSOR_MIN = 50.0
GROSOR_MAX = 250.0

longitudes_onda = np.linspace(400, 800, 200)
indices_refraccion = np.array([1.45, 2.3] * (NUM_CAPAS // 2))

# ==========================================
# 3. GENERACIÓN MASIVA
# ==========================================
X_grosores = np.zeros((NUM_MUESTRAS, NUM_CAPAS))
Y_espectros = np.zeros((NUM_MUESTRAS, len(longitudes_onda)))

print(f"Generando {NUM_MUESTRAS} muestras...")

for i in range(NUM_MUESTRAS):
    # Generar grosores
    grosores_aleatorios = np.random.uniform(GROSOR_MIN, GROSOR_MAX, NUM_CAPAS)
    
    # Calcular el espectro (ahora la función sí existe en este mismo archivo)
    espectro = calcular_transmitancia(grosores_aleatorios, indices_refraccion, longitudes_onda)
    
    X_grosores[i, :] = grosores_aleatorios
    Y_espectros[i, :] = espectro
    
    if (i + 1) % 1000 == 0:
        print(f"Progreso: {i + 1} / {NUM_MUESTRAS}")

# ==========================================
# 4. EXPORTACIÓN A CSV
# ==========================================
print("Estructurando datos para exportación...")

nombres_columnas_X = [f"d_{j+1}" for j in range(NUM_CAPAS)]
nombres_columnas_Y = [f"T_{int(lam)}" for lam in longitudes_onda]

df_grosores = pd.DataFrame(X_grosores, columns=nombres_columnas_X)
df_espectros = pd.DataFrame(Y_espectros, columns=nombres_columnas_Y)

df_final = pd.concat([df_grosores, df_espectros], axis=1)

nombre_archivo = "dataset_filtros_10k.csv"
df_final.to_csv(nombre_archivo, index=False)

print(f"¡Terminado! Dataset guardado como '{nombre_archivo}'.")