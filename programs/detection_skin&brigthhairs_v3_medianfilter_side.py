import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm.mat"

# Contraintes Physiques
MAX_JUMP = 2         # La peau ne saute pas de plus de 2 pixels par colonne
STABILITY_CHECK = 10 # Nombre de pixels consécutifs plats requis pour valider le départ

# Paramètres Masque
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

def find_stable_seed_side(slice_2d, seuil_surface):
    """
    Scanne les colonnes Y de 0 à la fin.
    Cherche le PREMIER endroit où la surface est stable sur 10 pixels.
    """
    rows_z, cols_y = slice_2d.shape
    
    print("   🔭 Recherche latérale (Scan G->D)...")
    
    for y in range(cols_y - STABILITY_CHECK):
        col = slice_2d[:, y]
        # Pixels brillants
        candidats = np.where(col > seuil_surface)[0]
        
        if len(candidats) == 0:
            continue # Colonne vide (air)
            
        z_start = candidats[0] # Le plus haut pixel brillant
        
        # Test de stabilité : Est-ce que les 10 suivants sont à la même hauteur (+/- 2px) ?
        is_stable = True
        voisins_z = []
        for k in range(1, STABILITY_CHECK + 1):
            col_next = slice_2d[:, y + k]
            cand_next = np.where(col_next > seuil_surface)[0]
            
            if len(cand_next) == 0:
                is_stable = False
                break
            
            z_next = cand_next[0]
            if abs(z_next - z_start) > MAX_JUMP: # Ça saute trop -> c'est du bruit/poil
                is_stable = False
                break
            voisins_z.append(z_next)
            
        if is_stable:
            # On a trouvé le bord de la peau !
            # On prend la médiane des voisins pour être précis
            final_seed_z = int(np.median([z_start] + voisins_z))
            print(f"   ✅ BORD STABLE TROUVÉ : Y={y}, Z={final_seed_z}")
            return y, final_seed_z

    print("⚠️ ECHEC : Aucun bord stable trouvé. Repli sur le milieu (Risqué).")
    return cols_y // 2, np.argmax(slice_2d[:, cols_y//2])

def process_side_scan(volume):
    print("--- 🕵️ TRACKING : SIDE SCAN (CORRIGÉ) ---")
    
    nx, ny, nz = volume.shape # (2176, 1440, 256)
    
    # Slice centrale
    slice_2d = volume[nx // 2, :, :].T # (256, 1440) -> (Z, Y)
    rows_z, cols_y = slice_2d.shape
    
    # Stats seuils
    seuil_surface = volume.mean() + 4 * volume.std()
    seuil_poil = np.percentile(volume, 99.8)
    
    surface_line = np.zeros(cols_y, dtype=int)
    
    # 1. Trouver le point de départ stable
    start_y, start_z = find_stable_seed_side(slice_2d, seuil_surface)
    surface_line[start_y] = start_z
    
    # 2. Tracking vers la DROITE (Du seed vers la fin)
    current_z = start_z
    for y in range(start_y + 1, cols_y):
        z_min = max(0, current_z - MAX_JUMP)
        z_max = min(rows_z, current_z + MAX_JUMP + 1)
        
        col_chunk = slice_2d[z_min:z_max, y]
        local_idx = np.argmax(col_chunk)
        val = col_chunk[local_idx]
        
        if val < (seuil_surface * 0.5): # Perte de signal
            new_z = current_z
        else:
            new_z = z_min + local_idx
            
        surface_line[y] = new_z
        current_z = new_z
        
    # 3. Tracking vers la GAUCHE (Du seed vers le début, si seed > 0)
    current_z = start_z
    for y in range(start_y - 1, -1, -1):
        z_min = max(0, current_z - MAX_JUMP)
        z_max = min(rows_z, current_z + MAX_JUMP + 1)
        
        col_chunk = slice_2d[z_min:z_max, y]
        local_idx = np.argmax(col_chunk)
        val = col_chunk[local_idx]
        
        if val < (seuil_surface * 0.5):
            new_z = current_z
        else:
            new_z = z_min + local_idx
            
        surface_line[y] = new_z
        current_z = new_z

    # 4. Lissage & Masque
    print("✨ Median Filter...")
    surface_smooth = median_filter(surface_line, size=15)
    
    mask_2d = np.zeros_like(slice_2d)
    Y_grid = np.arange(cols_y)
    Z_grid = np.arange(rows_z).reshape(-1, 1)
    Surf_grid = surface_smooth.reshape(1, -1)
    
    dist = Z_grid - Surf_grid
    cond_geo = (dist > -20) & (dist < MARGE_POIL_PIXELS)
    cond_lum = slice_2d > seuil_poil
    mask_2d = cond_geo & cond_lum
    
    visualize(slice_2d, mask_2d, surface_smooth, start_y, start_z)

def visualize(img, mask, surf, sy, sz):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Side Scan Tracking")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    plt.plot(surf, color='lime', linewidth=2)
    plt.scatter([sy], [sz], color='cyan', s=100, marker='>', label="Start")
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
    process_side_scan(volume)