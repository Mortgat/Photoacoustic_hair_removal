import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm.mat"

# Dimensions physiques
RESOLUTION_Z_MM = 0.125
PROFONDEUR_POIL_MM = 1.0  # Profondeur du masque sous la peau

# Contrainte de Tracking (Physique)
# Combien de pixels la peau peut monter/descendre entre deux colonnes ?
MAX_JUMP = 2 

# Largeur de la bande centrale pour l'initialisation (en pixels)
# On prend une zone assez fine au sommet de l'arc pour que Z soit constant
CENTRAL_BAND_WIDTH = 50 

# Conversion pixels
MARGE_POIL_PIXELS = int(PROFONDEUR_POIL_MM / RESOLUTION_Z_MM)

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v
    raise ValueError("Aucun volume 3D trouvé.")

def get_robust_seed(slice_2d, seuil_surface):
    """
    Trouve la hauteur Z de départ en analysant une bande centrale.
    Utilise la MÉDIANE pour ignorer les poils (outliers).
    """
    rows_z, cols_y = slice_2d.shape # (256, 1440) car on a transposé pour l'analyse
    
    center_y = cols_y // 2
    start_band = center_y - (CENTRAL_BAND_WIDTH // 2)
    end_band = center_y + (CENTRAL_BAND_WIDTH // 2)
    
    z_candidates = []
    
    # On scanne chaque colonne de la bande centrale
    for y in range(start_band, end_band):
        col = slice_2d[:, y]
        # On cherche le premier pixel brillant (le plus haut, index faible)
        pixels_brillants = np.where(col > seuil_surface)[0]
        
        if len(pixels_brillants) > 0:
            z_candidates.append(pixels_brillants[0])
            
    if not z_candidates:
        return np.argmax(slice_2d[:, center_y]) # Fallback
        
    # LE COEUR DE LA ROBUSTESSE :
    # Si on a des poils, ils auront un Z plus petit (plus haut).
    # La peau aura un Z "moyen".
    # La médiane nous donne la peau à coup sûr.
    seed_z = int(np.median(z_candidates))
    
    print(f"   🎯 Seed Central : Z={seed_z} (Calculé sur {len(z_candidates)} colonnes)")
    return seed_z

def process_geometric_tracking(volume):
    print("--- 🕵️ TRACKING : BANDE CENTRALE ROBUSTE ---")
    
    # 1. Dimensions correctes (X, Y, Z)
    nx, ny, nz = volume.shape
    print(f"   Dimensions : X={nx} (Slices), Y={ny} (Largeur), Z={nz} (Profondeur)")
    
    # Stats pour seuils
    mean_val = volume.mean()
    std_val = volume.std()
    seuil_surface = mean_val + 4 * std_val
    seuil_poil = np.percentile(volume, 99.8)

    # Pour la démo, on travaille sur la coupe centrale (milieu de X)
    idx_slice = nx // 2
    
    # Extraction 2D : On veut Y en horizontal et Z en vertical
    # volume[idx, :, :] donne (1440, 256).
    # Pour avoir Z en vertical (lignes) et Y en horizontal (colonnes), on transpose (.T)
    slice_2d = volume[idx_slice, :, :].T 
    rows_z, cols_y = slice_2d.shape # (256, 1440)
    
    surface_line = np.zeros(cols_y, dtype=int)
    
    # --- ETAPE 1 : INITIALISATION (SEED) ---
    seed_z = get_robust_seed(slice_2d, seuil_surface)
    center_y = cols_y // 2
    
    # On initialise le centre
    surface_line[center_y] = seed_z
    
    # --- ETAPE 2 : TRACKING VERS LA DROITE ---
    current_z = seed_z
    for y in range(center_y + 1, cols_y):
        # Fenêtre de recherche : [Z actuel +/- MAX_JUMP]
        z_min = max(0, current_z - MAX_JUMP)
        z_max = min(rows_z, current_z + MAX_JUMP + 1)
        
        col_chunk = slice_2d[z_min:z_max, y]
        local_idx = np.argmax(col_chunk)
        
        # Si signal trop faible (vide), on garde la hauteur précédente (extrapolation)
        val_max = col_chunk[local_idx]
        if val_max < (seuil_surface * 0.5):
            new_z = current_z
        else:
            new_z = z_min + local_idx
            
        surface_line[y] = new_z
        current_z = new_z

    # --- ETAPE 3 : TRACKING VERS LA GAUCHE ---
    current_z = seed_z
    for y in range(center_y - 1, -1, -1):
        z_min = max(0, current_z - MAX_JUMP)
        z_max = min(rows_z, current_z + MAX_JUMP + 1)
        
        col_chunk = slice_2d[z_min:z_max, y]
        local_idx = np.argmax(col_chunk)
        
        val_max = col_chunk[local_idx]
        if val_max < (seuil_surface * 0.5):
            new_z = current_z
        else:
            new_z = z_min + local_idx
            
        surface_line[y] = new_z
        current_z = new_z

    # --- ETAPE 4 : MEDIAN FILTER FINAL ---
    # Indispensable pour lisser les micro-marches d'escalier du tracking
    print("✨ Lissage final (Median Filter)...")
    surface_smooth = median_filter(surface_line, size=15)

    # --- ETAPE 5 : CREATION MASQUE ---
    mask_2d = np.zeros_like(slice_2d)
    
    # Grilles d'indices pour vectorisation rapide
    Y_grid = np.arange(cols_y)
    Z_grid = np.arange(rows_z).reshape(-1, 1)
    
    # On broadcast la ligne de surface sur toute l'image
    Surf_grid = surface_smooth.reshape(1, -1)
    
    # Calcul distance
    dist = Z_grid - Surf_grid
    
    # Conditions
    # 1. Géométrie : entre -2mm (air) et +ProfondeurPoil
    # On prend large au dessus (-20px) pour attraper les poils sortants
    cond_geo = (dist > -20) & (dist < MARGE_POIL_PIXELS)
    
    # 2. Luminosité : Seulement ce qui brille fort (poils)
    cond_lum = slice_2d > seuil_poil
    
    mask_2d = cond_geo & cond_lum
    
    # --- VISUALISATION ---
    visualize(slice_2d, mask_2d, surface_smooth, seed_z, center_y)

def visualize(img, mask, surf, seed_z, seed_y):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Tracking (Point Cyan = Seed Médian)")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    plt.plot(surf, color='lime', linewidth=2, label="Surface (Median Filter)")
    # On affiche le point de départ
    plt.scatter([seed_y], [seed_z], color='cyan', s=100, marker='X', zorder=5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Masque Final")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
    overlay[mask == 1] = [1, 0, 0, 0.8] # Rouge
    plt.imshow(overlay, aspect='auto')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    volume = load_volume(fichier_mat)
    process_geometric_tracking(volume)