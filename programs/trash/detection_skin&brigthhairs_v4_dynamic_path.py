import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
MAX_JUMP = 2 # Contrainte physique stricte
RESOLUTION_Z_MM = 0.125
PROFONDEUR_POIL_MM = 1.0
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

def solve_viterbi_path(slice_2d):
    """
    Trouve le chemin optimal Z(y) qui maximise la luminosité totale
    tout en respectant |Z(y) - Z(y-1)| <= MAX_JUMP.
    """
    rows_z, cols_y = slice_2d.shape # (256, 1440)
    
    # Matrice de coût accumulé (Energy)
    E = np.zeros_like(slice_2d, dtype=np.float32)
    # Matrice de provenance (pour reconstruire le chemin)
    path_from = np.zeros_like(slice_2d, dtype=np.int32)
    
    # Init première colonne
    E[:, 0] = slice_2d[:, 0]
    
    print("   🧠 Calcul du chemin optimal (Passe avant)...")
    
    # PASSE AVANT (Forward)
    for y in range(1, cols_y):
        prev_col_E = E[:, y-1]
        
        # Pour chaque pixel Z de la colonne actuelle...
        # (On peut optimiser, mais la boucle est plus sûre pour la logique)
        for z in range(rows_z):
            # On regarde les voisins possibles dans la colonne d'avant
            z_prev_min = max(0, z - MAX_JUMP)
            z_prev_max = min(rows_z, z + MAX_JUMP + 1)
            
            # Quel voisin avait le meilleur score ?
            # On slice le tableau E précédent
            candidates = prev_col_E[z_prev_min:z_prev_max]
            best_local_idx = np.argmax(candidates)
            best_prev_z = z_prev_min + best_local_idx
            
            # Mise à jour du score
            E[z, y] = slice_2d[z, y] + prev_col_E[best_prev_z]
            path_from[z, y] = best_prev_z

    print("   🔙 Reconstruction (Passe arrière)...")
    surface_line = np.zeros(cols_y, dtype=int)
    
    # Trouver la fin du meilleur chemin
    best_end_z = np.argmax(E[:, cols_y - 1])
    surface_line[cols_y - 1] = best_end_z
    
    # Remonter le temps
    curr_z = best_end_z
    for y in range(cols_y - 1, 0, -1):
        prev_z = path_from[curr_z, y]
        surface_line[y-1] = prev_z
        curr_z = prev_z
        
    return surface_line

def process_dynamic_path(volume):
    print("--- 🛣️ TRACKING : CHEMIN DYNAMIQUE OPTIMAL ---")
    
    nx, ny, nz = volume.shape
    idx_slice = nx // 2
    slice_2d = volume[idx_slice, :, :].T # (256, 1440)
    
    seuil_poil = np.percentile(volume, 99.8)
    
    # 1. Calculer le chemin
    surface_brute = solve_viterbi_path(slice_2d)
    
    # 2. Lissage (Toujours utile)
    print("✨ Median Filter...")
    surface_smooth = median_filter(surface_brute, size=15)
    
    # 3. Masque
    mask_2d = np.zeros_like(slice_2d)
    rows_z, cols_y = slice_2d.shape
    
    Z_grid = np.arange(rows_z).reshape(-1, 1)
    Surf_grid = surface_smooth.reshape(1, -1)
    
    dist = Z_grid - Surf_grid
    cond_geo = (dist > -20) & (dist < MARGE_POIL_PIXELS)
    cond_lum = slice_2d > seuil_poil
    mask_2d = cond_geo & cond_lum
    
    visualize(slice_2d, mask_2d, surface_smooth, surface_brute)

def visualize(img, mask, surf_smooth, surf_brute):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Chemin Optimal")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    plt.plot(surf_brute, color='red', linewidth=0.5, alpha=0.5, label="Brut (DP)")
    plt.plot(surf_smooth, color='lime', linewidth=2, label="Lissé")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Masque")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
    overlay[mask == 1] = [1, 0, 0, 0.8]
    plt.imshow(overlay, aspect='auto')
    plt.show()

if __name__ == "__main__":
    volume = load_volume(fichier_mat)
    process_dynamic_path(volume)