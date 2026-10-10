import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

# ==========================================
# 1. CARGA DE DATOS 
# ==========================================
class FiltrosDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        x_data_cruda = df.iloc[:, :64].values
        y_data_cruda = df.iloc[:, 64:].values
        
        self.grosor_min = 50.0
        self.grosor_max = 250.0
        x_data_normalizada = (x_data_cruda - self.grosor_min) / (self.grosor_max - self.grosor_min)
        
        self.x_tensor = torch.tensor(x_data_normalizada, dtype=torch.float32)
        self.y_tensor = torch.tensor(y_data_cruda, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x_tensor)
        
    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

dataset_completo = FiltrosDataset("dataset_filtros_10k.csv") 
train_size = int(0.8 * len(dataset_completo))
val_size = len(dataset_completo) - train_size
train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# ==========================================
# 2. ARQUITECTURAS
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

class RedInversa(nn.Module):
    def __init__(self):
        super(RedInversa, self).__init__()
        # Entrada: 200 (espectro), Salida: 8 (grosores normalizados)
        self.modelo = nn.Sequential(
            nn.Linear(200, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.Sigmoid()
        )
    def forward(self, x):
        return self.modelo(x)

# ==========================================
# 3. ENSAMBLE DEL TÁNDEM
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Iniciando entrenamiento Tándem en: {device}")

# Cargar el motor físico (Simulador Forward)
simulador_forward = RedForward().to(device)
simulador_forward.load_state_dict(torch.load("simulador_forward.pth", weights_only=True))
simulador_forward.eval() 

# CONGELAR LOS PESOS DEL SIMULADOR
for param in simulador_forward.parameters():
    param.requires_grad = False

# Instanciar la red que vamos a entrenar
red_inversa = RedInversa().to(device)

criterio = nn.MSELoss()
# El optimizador SOLO vigila los parámetros de la Red Inversa
optimizador = optim.Adam(red_inversa.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=10)

# ==========================================
# 4. ENTRENAMIENTO TÁNDEM
# ==========================================
epocas = 200

for epoca in range(epocas):
    red_inversa.train()
    loss_entrenamiento = 0.0
    
    # En el tándem, solo nos importa el espectro deseado (lotes_y)
    for _, lotes_y in train_loader: 
        lotes_y = lotes_y.to(device)
        
        optimizador.zero_grad()
        
        # 1. La red inversa diseña una geometría para el espectro objetivo
        grosores_predichos = red_inversa(lotes_y)
        
        # 2. El simulador evalúa esa geometría y arroja el espectro resultante
        espectro_obtenido = simulador_forward(grosores_predichos)
        
        # 3. Calculamos el error entre el espectro obtenido y el objetivo original
        loss = criterio(espectro_obtenido, lotes_y)
        
        loss.backward()
        optimizador.step()
        
        loss_entrenamiento += loss.item()
        
    promedio_train_loss = loss_entrenamiento / len(train_loader)
    
    # --- FASE DE VALIDACIÓN ---
    red_inversa.eval()
    loss_validacion = 0.0
    
    with torch.no_grad():
        for _, lotes_y in val_loader:
            lotes_y = lotes_y.to(device)
            grosores_predichos = red_inversa(lotes_y)
            espectro_obtenido = simulador_forward(grosores_predichos)
            loss = criterio(espectro_obtenido, lotes_y)
            loss_validacion += loss.item()
            
    promedio_val_loss = loss_validacion / len(val_loader)
    scheduler.step(promedio_val_loss)
    
    if (epoca + 1) % 10 == 0:
        print(f"Época [{epoca+1}/{epocas}] | Loss Train: {promedio_train_loss:.6f} | Loss Val: {promedio_val_loss:.6f}")

torch.save(red_inversa.state_dict(), "red_inversa.pth")
print("Entrenamiento Tándem finalizado. Red Inversa guardada y lista para usarse.")