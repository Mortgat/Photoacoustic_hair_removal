import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal.windows import hann

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_VOL = r"dicom_data/dicom/4261_fromdcm.npz"
DOSSIER_MASQUES = r"pipeline/2steps_pu_learning"
DOSSIER_MODELE = r"pipeline/modele_pu"
CHEMIN_POIDS = os.path.join(DOSSIER_MODELE, "unet_pu_weights.pth")

DZ_MM = 0.125
ZONE_CONFLIT_MAX_MM = 5.0
PATCH_XY = 64
PATCH_Z = 24

# ==========================================
# ARCHITECTURE U-NET 3D (Identique à l'entraînement)
# ==========================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), groups=8):
        super().__init__()
        pad = tuple(k // 2 for k in kernel_size)
        g = min(groups, out_c)
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(g, out_c), nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(g, out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class MiniUNet3DAniso(nn.Module):
    def __init__(self, in_channels=3, base=16): 
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base, kernel_size=(3, 3, 3))
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3)
        if d2.shape[-3:] != e2.shape[-3:]: d2 = F.interpolate(d2, size=e2.shape[-3:])
        d2 = self.dec2(torch.cat([e2, d2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]: d1 = F.interpolate(d1, size=e1.shape[-3:])
        return self.out(self.dec1(torch.cat([e1, d1], dim=1)))

# ==========================================
# INFÉRENCE GLISSANTE OPTIMISÉE POUR LA RAM
# ==========================================
def get_hann_window_3d(shape):
    wx = hann(shape[0], sym=True)
    wy = hann(shape[1], sym=True)
    wz = hann(shape[2], sym=True)
    return (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float32)

def sliding_window_inference_ram_safe(model, vol, dist, frangi, device, p_xy=PATCH_XY, p_z=PATCH_Z):
    print("\n--- INFÉRENCE GLISSANTE 3D (MODE SÉCURITÉ RAM) ---")
    model.eval()
    nx, ny, nz = vol.shape
    
    # Allocation stricte
    out_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    weight_map = np.zeros((nx, ny, nz), dtype=np.float32)
    window = get_hann_window_3d((p_xy, p_xy, p_z))
    
    step_xy, step_z = p_xy // 2, p_z // 2
    
    with torch.no_grad():
        for x in tqdm(range(0, max(1, nx - p_xy + 1), step_xy), desc="Balayage Axe X"):
            for y in range(0, max(1, ny - p_xy + 1), step_xy):
                for z in range(0, max(1, nz - p_z + 1), step_z):
                    
                    x0, y0, z0 = min(x, nx - p_xy), min(y, ny - p_xy), min(z, nz - p_z)
                    xs, ys, zs = slice(x0, x0+p_xy), slice(y0, y0+p_xy), slice(z0, z0+p_z)
                    
                    patch_in = np.stack([vol[xs, ys, zs], dist[xs, ys, zs], frangi[xs, ys, zs]], axis=0)
                    tensor_in = torch.from_numpy(patch_in).unsqueeze(0).float().to(device)
                    
                    probs = torch.sigmoid(model(tensor_in)).squeeze().cpu().numpy()
                    
                    out_mask[xs, ys, zs] += probs * window
                    weight_map[xs, ys, zs] += window

    print("Fusion et nettoyage mémoire...")
    
    # Opérations IN-PLACE pour éviter de doubler l'empreinte RAM
    weight_map[weight_map == 0] = 1.0
    out_mask /= weight_map 
    
    # Suppression explicite et purge du garbage collector
    del weight_map
    del window
    gc.collect()
    
    # Conversion finale
    out_mask_f16 = out_mask.astype(np.float16)
    del out_mask
    gc.collect()
    
    return out_mask_f16

# ==========================================
# EXÉCUTION
# ==========================================
def main():
    if not os.path.exists(CHEMIN_POIDS):
        print(f"Erreur : Modèle introuvable dans {CHEMIN_POIDS}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil d'inférence : {device}")
    
    print("1. Chargement et préparation des tenseurs...")
    vol = np.load(FICHIER_VOL)["volume"].astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8) 
    
    surf = np.load(os.path.join(DOSSIER_MASQUES, "surface_peau.npy"))
    frangi = np.load(os.path.join(DOSSIER_MASQUES, "carte_frangi.npy")).astype(np.float32)

    nz = vol.shape[2]
    surf_propre = np.nan_to_num(surf, nan=nz) 
    dist_mm = (np.arange(nz).reshape(1, 1, nz) - surf_propre[:, :, None]) * DZ_MM
    dist_norm = np.clip(dist_mm / ZONE_CONFLIT_MAX_MM, -1.0, 1.0).astype(np.float32)
    
    del surf_propre
    del dist_mm
    gc.collect()

    print("2. Chargement du modèle...")
    model = MiniUNet3DAniso(in_channels=3).to(device)
    model.load_state_dict(torch.load(CHEMIN_POIDS, map_location=device))
    
    torch.cuda.empty_cache()

    print("3. Début de l'inférence...")
    masque_predit = sliding_window_inference_ram_safe(model, vol, dist_norm, frangi, device)
    
    print("4. Sauvegarde du résultat...")
    chemin_prediction = os.path.join(DOSSIER_MASQUES, "prediction_pu.npy")
    np.save(chemin_prediction, masque_predit)
    
    print(f"✅ Terminé. Prédiction sauvegardée : {chemin_prediction}")

if __name__ == "__main__":
    main()