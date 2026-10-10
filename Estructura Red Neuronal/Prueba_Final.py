import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. EL MOTOR FÍSICO (TMM ORIGINAL)
# ==========================================
def calcular_transmitancia(grosores, indices, longitudes_onda, n_0=1.0, n_s=1.0):
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
        
        t = (2 * n_0) / ((n_0 * m11) + (n_0 * n_s * m12) + m21 + (n_s * m22))
        T_espectro[i] = (n_s / n_0) * np.abs(t)**2
    return T_espectro

# ==========================================
# 2. CARGAR LA RED INVERSA
# ==========================================
class RedInversa(nn.Module):
    def __init__(self):
        super(RedInversa, self).__init__()
        self.modelo = nn.Sequential(
            nn.Linear(200, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.Sigmoid()
        )
    def forward(self, x):
        return self.modelo(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
red_inversa = RedInversa().to(device)
# Cargar los pesos que entrenaste con el código anterior
red_inversa.load_state_dict(torch.load("red_inversa.pth", weights_only=True))
red_inversa.eval()

# ==========================================
# 3. CREAR UN ESPECTRO INVENTADO (OBJETIVO)
# ==========================================
longitudes_onda = np.linspace(400, 800, 200)

# Vamos a inventar un filtro pasa-bandas con forma Gaussiana centrado en 600 nm (naranja)
centro = 600.0
ancho = 30.0 # Qué tan ancho es el pico
espectro_objetivo = np.exp(-0.5 * ((longitudes_onda - centro) / ancho)**2)

# Convertir el objetivo a tensor de PyTorch para pasarlo por la red
tensor_objetivo = torch.tensor(espectro_objetivo, dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# 4. INFERENCIA Y DESNORMALIZACIÓN
# ==========================================
with torch.no_grad():
    grosores_predichos_norm = red_inversa(tensor_objetivo).cpu().numpy()[0]

# EL PASO CRÍTICO: Devolver los grosores a la escala física [50, 250] nm
grosor_min, grosor_max = 50.0, 250.0
grosores_fisicos = (grosores_predichos_norm * (grosor_max - grosor_min)) + grosor_min

print("\n--- GEOMETRÍA DISEÑADA POR LA IA ---")
for i, d in enumerate(grosores_fisicos):
    print(f"Capa {i+1}: {d:.2f} nm")

# ==========================================
# 5. VALIDACIÓN FÍSICA Y GRÁFICA
# ==========================================
NUM_CAPAS = 64
indices_refraccion = np.array([1.45, 2.3] * (NUM_CAPAS // 2))

# Simulamos la geometría que nos dio la red usando el código original
espectro_validado = calcular_transmitancia(grosores_fisicos, indices_refraccion, longitudes_onda)

# Graficar Resultados
plt.figure(figsize=(10, 6))
plt.plot(longitudes_onda, espectro_objetivo, 'k--', linewidth=2, label="Objetivo Ideal (Gaussiana)")
plt.plot(longitudes_onda, espectro_validado, 'r-', linewidth=2, label="Simulación del Diseño IA")
plt.title("Validación de la Red Tándem: Diseño Inverso de Filtro Multicapa", fontsize=14)
plt.xlabel("Longitud de onda (nm)", fontsize=12)
plt.ylabel("Transmitancia", fontsize=12)
plt.ylim(0, 1.1)
plt.xlim(400, 800)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()