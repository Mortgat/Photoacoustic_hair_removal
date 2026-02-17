import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# Paramètres Physiques
RESOLUTION_Z_MM = 0.125
EPAISSEUR_PEAU_MM = 2.0  # La peau fait max 2mm
EPAISSEUR_PIXELS = int(EPAISSEUR_PEAU_MM / RESOLUTION_Z_MM)

# Taille du filtre de continuité (Lissage de la nappe)
# Plus c'est grand, plus la peau sera rigide (ignorant les poils et les trous)
SURFACE_SMOOTHING_SIZE = 25  # Rayon en pixels (ex: 25px autour)

# Seuil de détection "Premier Echo"
# On prendra X% du max pour dire "c'est de la matière"
SEUIL_DETECTION_RATIO = 0.15 

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v
    raise ValueError("Aucun volume 3D trouvé.")

def extract_robust_surface(volume):
    print("--- 🗺️ EXTRACTION SURFACE 2.5D (HEIGHTMAP) ---")
    
    nx, ny, nz = volume.shape
    
    # 1. GENERATION DE LA CARTE BRUTE (First Hit)
    print("   📍 Calcul du 'Premier Écho' (Z min)...")
    
    # Seuil absolu
    seuil_abs = np.max(volume) * SEUIL_DETECTION_RATIO
    
    # On binarise le volume temporairement
    mask_binaire = volume > seuil_abs
    
    # argmax renvoie l'index du PREMIER True le long de l'axe Z (axis=2)
    # Si la colonne est vide (tout False), argmax renvoie 0. On devra corriger ça.
    z_map_raw = np.argmax(mask_binaire, axis=2)
    
    # Correction des colonnes vides (là où il n'y a aucun signal)
    # On regarde si le max le long de Z est bien True. Sinon, c'est du vide.
    any_signal = np.max(mask_binaire, axis=2)
    
    # On remplace les 0 (vide) par la valeur moyenne de profondeur pour ne pas fausser le filtre médian
    z_mean = int(np.mean(z_map_raw[any_signal]))
    z_map_raw[~any_signal] = z_mean
    
    # 2. LISSAGE DE SURFACE (Continuité 3D)
    print(f"   🔨 Lissage de la nappe (Filtre Médian {SURFACE_SMOOTHING_SIZE}x{SURFACE_SMOOTHING_SIZE})...")
    # C'est ici que la magie opère : on force la continuité spatiale XY
    z_map_smooth = median_filter(z_map_raw, size=SURFACE_SMOOTHING_SIZE)
    
    # 3. CREATION DU MASQUE 3D (Extrusion)
    print(f"   🍰 Création du masque volumique (Épaisseur {EPAISSEUR_PIXELS} px)...")
    
    # On crée une grille d'indices Z pour tout le volume
    # Z_indices shape : (1, 1, 256) broadcasting
    Z_indices = np.arange(nz).reshape(1, 1, nz)
    
    # On broadcast la carte de hauteur : (2176, 1440, 1)
    Surface_limit = z_map_smooth[:, :, np.newaxis]
    
    # Condition : Etre SOUS la surface ET au-dessus de (Surface + Epaisseur)
    # Rappel : Z augmente avec la profondeur.
    # Donc : Z >= Surface  ET  Z <= Surface + Epaisseur
    
    mask_final = (Z_indices >= Surface_limit) & (Z_indices <= (Surface_limit + EPAISSEUR_PIXELS))
    
    # On applique aussi le seuil de luminosité pour ne pas garder du vide dans cette tranche
    mask_final = mask_final & mask_binaire

    visualize_results(volume, z_map_raw, z_map_smooth, mask_final)

def visualize_results(volume, z_raw, z_smooth, mask_3d):
    # On visualise la coupe centrale pour vérifier
    nx, ny, nz = volume.shape
    idx = nx // 2
    
    slice_orig = volume[idx, :, :].T
    slice_mask = mask_3d[idx, :, :].T
    line_raw = z_raw[idx, :]
    line_smooth = z_smooth[idx, :]
    
    plt.figure(figsize=(16, 10))
    
    # 1. Carte de Hauteur (Vue de dessus)
    plt.subplot(2, 2, 1)
    plt.title("Carte de Hauteur LISSÉE (Vue de dessus)")
    plt.imshow(z_smooth.T, cmap='jet', aspect='auto') # Transposé pour correspondre à (Y, X)
    plt.colorbar(label="Profondeur Z")
    
    # 2. Coupe Transversale : Tracking
    plt.subplot(2, 2, 2)
    plt.title(f"Coupe X={idx} : Surface Détectée")
    plt.imshow(slice_orig, cmap='gray', aspect='auto', vmax=np.percentile(slice_orig, 99))
    plt.plot(line_raw, color='red', linewidth=0.5, alpha=0.5, label="Premier Echo (Brut)")
    plt.plot(line_smooth, color='lime', linewidth=2, label="Surface Lissée")
    plt.plot(line_smooth + EPAISSEUR_PIXELS, color='lime', linestyle='--', linewidth=1, label="Limite Profondeur")
    plt.legend()
    
    # 3. Coupe Transversale : Masque Final
    plt.subplot(2, 2, 3)
    plt.title("Masque Final (Épaisseur contrainte)")
    plt.imshow(slice_orig, cmap='gray', aspect='auto', vmax=np.percentile(slice_orig, 99))
    overlay = np.zeros((nz, ny, 4))
    overlay[slice_mask == 1] = [0, 1, 0, 0.6]
    plt.imshow(overlay, aspect='auto')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    volume = load_volume(fichier_mat)
    extract_robust_surface(volume)