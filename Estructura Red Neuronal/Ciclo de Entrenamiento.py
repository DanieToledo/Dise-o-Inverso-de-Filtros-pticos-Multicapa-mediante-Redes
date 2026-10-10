import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

# ==========================================
# 1. CARGA DE DATOS (DATASET Y DATALOADERS)
# ==========================================
class FiltrosDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        x_data_cruda = df.iloc[:, :64].values
        y_data_cruda = df.iloc[:, 64:].values
        
        # Normalización crítica
        self.grosor_min = 50.0
        self.grosor_max = 250.0
        x_data_normalizada = (x_data_cruda - self.grosor_min) / (self.grosor_max - self.grosor_min)
        
        self.x_tensor = torch.tensor(x_data_normalizada, dtype=torch.float32)
        self.y_tensor = torch.tensor(y_data_cruda, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x_tensor)
        
    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

# Asegúrate de que el nombre coincida con tu archivo generado
dataset_completo = FiltrosDataset("dataset_filtros_10k.csv") 
train_size = int(0.8 * len(dataset_completo))
val_size = len(dataset_completo) - train_size
train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# ==========================================
# 2. ARQUITECTURA FORWARD (SIMULADOR)
# ==========================================
class RedForward(nn.Module):
    def __init__(self):
        super(RedForward, self).__init__()
        self.modelo = nn.Sequential(
            nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 200), nn.Sigmoid()
        )
    def forward(self, x):
        return self.modelo(x)

# ==========================================
# 3. CONFIGURACIÓN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

simulador_forward = RedForward().to(device)
criterio = nn.MSELoss()
optimizador = optim.Adam(simulador_forward.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=10)

# ==========================================
# 4. CICLO DE ENTRENAMIENTO
# ==========================================
epocas = 300

for epoca in range(epocas):
    # FASE DE ENTRENAMIENTO
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
    
    # FASE DE VALIDACIÓN
    simulador_forward.eval()
    loss_validacion = 0.0
    
    with torch.no_grad():
        for lotes_x, lotes_y in val_loader:
            lotes_x, lotes_y = lotes_x.to(device), lotes_y.to(device)
            predicciones = simulador_forward(lotes_x)
            loss = criterio(predicciones, lotes_y)
            loss_validacion += loss.item()
            
    promedio_val_loss = loss_validacion / len(val_loader)
    scheduler.step(promedio_val_loss)
    
    if (epoca + 1) % 10 == 0:
        print(f"Época [{epoca+1}/{epocas}] | Loss Train: {promedio_train_loss:.6f} | Loss Val: {promedio_val_loss:.6f}")

torch.save(simulador_forward.state_dict(), "simulador_forward.pth")
print("Entrenamiento finalizado. Modelo guardado.")