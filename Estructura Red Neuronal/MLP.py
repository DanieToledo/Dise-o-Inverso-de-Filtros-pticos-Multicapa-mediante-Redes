import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

class FiltrosDataset(Dataset):
    def __init__(self, csv_file):
        # 1. Leer el archivo CSV
        df = pd.read_csv(csv_file)
        
        # 2. Rebanar los datos
        # Las primeras 8 columnas son los grosores (X), las 200 restantes son el espectro (Y)
        x_data_cruda = df.iloc[:, :8].values
        y_data_cruda = df.iloc[:, 8:].values
        
        # 3. NORMALIZACIÓN (El paso más crítico)
        # Escalamos los grosores físicamente posibles (50 a 250 nm) al rango [0, 1]
        self.grosor_min = 50.0
        self.grosor_max = 250.0
        x_data_normalizada = (x_data_cruda - self.grosor_min) / (self.grosor_max - self.grosor_min)
        
        # El espectro ya es transmitancia (0 a 1), así que lo dejamos intacto
        
        # 4. Convertir a tensores de PyTorch
        # Es obligatorio usar float32 para los pesos de las redes neuronales
        self.x_tensor = torch.tensor(x_data_normalizada, dtype=torch.float32)
        self.y_tensor = torch.tensor(y_data_cruda, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x_tensor)
        
    def __getitem__(self, idx):
        # Devuelve el par: (Geometría, Espectro)
        return self.x_tensor[idx], self.y_tensor[idx]

# ==========================================
# PREPARACIÓN PARA EL ENTRENAMIENTO
# ==========================================
if __name__ == "__main__":
    # Instanciar el dataset
    dataset_completo = FiltrosDataset("dataset_filtros_10k.csv")
    
    # División: 80% Entrenamiento, 20% Validación
    train_size = int(0.8 * len(dataset_completo))
    val_size = len(dataset_completo) - train_size
    train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])
    
    # Crear los DataLoaders (batch_size de 128 es ideal para tu laptop)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"Muestras para entrenar: {len(train_dataset)}")
    print(f"Muestras para validar: {len(val_dataset)}")
    
    # Comprobación de dimensiones del primer lote
    x_batch, y_batch = next(iter(train_loader))
    print(f"Dimensiones de un lote de Geometría: {x_batch.shape} -> (Lote, 8 capas)")
    print(f"Dimensiones de un lote de Espectro: {y_batch.shape} -> (Lote, 200 longitudes de onda)")

    import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. ARQUITECTURA DE LA RED FORWARD
# ==========================================
class RedForward(nn.Module):
    def __init__(self):
        super(RedForward, self).__init__()
        
        self.modelo = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1), # Apaga el 10% de las neuronas al azar para forzar generalización
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 200),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.modelo(x)

# ==========================================
# 2. CONFIGURACIÓN DEL ENTRENAMIENTO (MODIFICADA)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simulador_forward = RedForward().to(device)
criterio = nn.MSELoss()

# Empezamos con un Learning Rate estándar
optimizador = optim.Adam(simulador_forward.parameters(), lr=0.001)

# NUEVO: Programador de tasa de aprendizaje
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=10)
# ==========================================
# 3. CICLO DE ENTRENAMIENTO (TRAINING LOOP)
# ==========================================
epocas = 300 
historial_train = []
historial_val = []

for epoca in range(epocas):
    # --- FASE DE ENTRENAMIENTO ---
    simulador_forward.train()
    loss_entrenamiento = 0.0
    
    for lotes_x, lotes_y in train_loader:
        lotes_x, lotes_y = lotes_x.to(device), lotes_y.to(device)
        
        optimizador.zero_grad()
        predicciones = simulador_forward(lotes_x)
        loss = criterio(predicciones, lotes_y)
        loss.backward()
        optimizador.step()
        
        loss_entrenamiento += loss.item()
        
    promedio_train_loss = loss_entrenamiento / len(train_loader)
    historial_train.append(promedio_train_loss)

   # --- FASE DE VALIDACIÓN ---
    simulador_forward.eval()
    loss_validacion = 0.0
    
    with torch.no_grad():
        for lotes_x, lotes_y in val_loader:
            lotes_x, lotes_y = lotes_x.to(device), lotes_y.to(device)
            predicciones = simulador_forward(lotes_x)
            loss = criterio(predicciones, lotes_y)
            loss_validacion += loss.item()
            
    promedio_val_loss = loss_validacion / len(val_loader)
    historial_val.append(promedio_val_loss)
    
    # Notificar al scheduler
    scheduler.step(promedio_val_loss)
    
    # Imprimir progreso cada 10 épocas
    if (epoca + 1) % 10 == 0:
        print(f"Época [{epoca+1}/{epocas}] | Loss Train: {promedio_train_loss:.6f} | Loss Val: {promedio_val_loss:.6f}")

# Guardar los pesos de la red pre-entrenada al final del ciclo
torch.save(simulador_forward.state_dict(), "simulador_forward.pth")
print("Entrenamiento finalizado. Modelo guardado.")