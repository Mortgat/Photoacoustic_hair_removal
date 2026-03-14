import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# ==========================================
# ⚙️ CONFIGURATION ET HYPERPARAMÈTRES
# ==========================================
fichier_volume = r"dicom_data/dicom/4261_fromdcm.npz"
fichier_poils = r"pipeline/body_hair_extraction_methods/masque_positifs_pu.npy"
fichier_sortie = r"pipeline/body_hair_extraction_methods/masque_predit_pu.npy"

PRIOR_PI = 0.05        # ~5% de poils dans le volume
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
PATCH_SIZE = 64        # Taille des cubes 3D d'entraînement (attention à la VRAM)
EPOCHS = 10
ITERATIONS_PAR_EPOCH = 100 # Nombre de patchs tirés au hasard par époque

# Activation explicite des optimisations GPU (NVIDIA CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"🚀 Matériel utilisé pour le Deep Learning : {device.type.upper()}")

# ==========================================
# 🧠 ARCHITECTURE : MINI U-NET 3D
# ==========================================
class MiniUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encodeur
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self.conv_block(16, 32)
        
        # Décodeur
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        
        # Sortie (1 canal, pas d'activation car géré par BCEWithLogitsLoss)
        self.out = nn.Conv3d(16, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        d1 = self.up1(e2)
        # Concaténation (Skip connection)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

# ==========================================
# ⚖️ FONCTION DE PERTE : PU LOSS
# ==========================================
class nnPULoss(nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.prior = prior
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, labels):
        pos_loss = self.loss_func(predictions, torch.ones_like(predictions))
        unl_loss = self.loss_func(predictions, torch.zeros_like(predictions))
        pos_as_neg_loss = self.loss_func(predictions, torch.zeros_like(predictions))

        pos_mask = (labels == 1).float()
        unl_mask = (labels == 0).float()
        
        n_pos = torch.clamp(pos_mask.sum(), min=1.0)
        n_unl = torch.clamp(unl_mask.sum(), min=1.0)

        risk_p = (pos_loss * pos_mask).sum() / n_pos
        risk_u = (unl_loss * unl_mask).sum() / n_unl
        risk_p_neg = (pos_as_neg_loss * pos_mask).sum() / n_pos

        risk_n = risk_u - self.prior * risk_p_neg
        
        if risk_n < 0:
            return -risk_n + self.prior * risk_p
        else:
            return risk_n + self.prior * risk_p

# ==========================================
# 📦 GESTION DES DONNÉES (PATCHS ALÉATOIRES)
# ==========================================
class RandomPatchDataset(Dataset):
    def __init__(self, volume, masque, patch_size, iterations):
        # Normalisation Z-score du volume pour le réseau de neurones
        self.volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
        self.masque = masque
        self.patch_size = patch_size
        self.iterations = iterations
        self.nx, self.ny, self.nz = volume.shape

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        x = np.random.randint(0, self.nx - self.patch_size)
        y = np.random.randint(0, self.ny - self.patch_size)
        z = np.random.randint(0, self.nz - self.patch_size)
        
        vol_patch = self.volume[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        mask_patch = self.masque[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        
        vol_tensor = torch.tensor(vol_patch, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_patch, dtype=torch.float32).unsqueeze(0)
        
        return vol_tensor, mask_tensor

# ==========================================
# 🚀 EXÉCUTION PRINCIPALE
# ==========================================
def run_pipeline():
    print("Chargement des matrices en RAM CPU...")
    volume_complet = np.load(fichier_volume)['volume']
    masque_positifs = np.load(fichier_poils)
    nx, ny, nz = volume_complet.shape

    dataset = RandomPatchDataset(volume_complet, masque_positifs, PATCH_SIZE, ITERATIONS_PAR_EPOCH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    modele = MiniUNet3D().to(device)
    optimizer = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)
    critere_pu = nnPULoss(prior=PRIOR_PI)

    print("--- 🏋️ ENTRAÎNEMENT DU RÉSEAU ---")
    modele.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_vol, batch_mask in dataloader:
            # Transfert strict vers la carte graphique
            batch_vol = batch_vol.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            predictions = modele(batch_vol)
            loss = critere_pu(predictions, batch_mask)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Époque {epoch+1}/{EPOCHS} - Loss moyenne: {total_loss/len(dataloader):.4f}")

    print("--- 🔬 INFÉRENCE (RECONSTRUCTION DU MASQUE COMPLET) ---")
    modele.eval()
    masque_predit = np.zeros_like(volume_complet, dtype=np.uint8)
    
    # Inférence tranche par tranche pour ne pas surcharger la VRAM
    # On découpe le volume entier en blocs selon l'axe Z
    vol_norm = (volume_complet - np.mean(volume_complet)) / (np.std(volume_complet) + 1e-8)
    
    with torch.no_grad():
        step = PATCH_SIZE
        for x in range(0, nx, step):
            for y in range(0, ny, step):
                for z in range(0, nz, step):
                    x_end, y_end, z_end = min(x+step, nx), min(y+step, ny), min(z+step, nz)
                    
                    # Extraction du bloc (padding automatique si on déborde)
                    bloc = np.zeros((step, step, step), dtype=np.float32)
                    v_shape = vol_norm[x:x_end, y:y_end, z:z_end].shape
                    bloc[:v_shape[0], :v_shape[1], :v_shape[2]] = vol_norm[x:x_end, y:y_end, z:z_end]
                    
                    bloc_tensor = torch.tensor(bloc).unsqueeze(0).unsqueeze(0).to(device)
                    pred = modele(bloc_tensor)
                    
                    # Sigmoid pour obtenir une probabilité, puis seuillage binaire à 0.5
                    pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
                    pred_binaire = (pred_prob > 0.5).astype(np.uint8)
                    
                    # On replace le bloc dans le masque final
                    masque_predit[x:x_end, y:y_end, z:z_end] = pred_binaire[:v_shape[0], :v_shape[1], :v_shape[2]]

    print(f"💾 Sauvegarde du masque réseau dans : {fichier_sortie}")
    np.save(fichier_sortie, masque_predit)
    print("✅ Pipeline PU Learning terminée !")

if __name__ == "__main__":
    run_pipeline()