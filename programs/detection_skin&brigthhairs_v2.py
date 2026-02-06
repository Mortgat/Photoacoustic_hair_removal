import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter, grey_closing
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data\dicom\4261_fromdcm.mat"

RESOLUTION_Z_MM = 0.125
PROFONDEUR_ANALYSE_MM = 1.0 
MARGE_PIXELS = int(PROFONDEUR_ANALYSE_MM / RESOLUTION_Z_MM)

# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def detecter_surface_premier_pic(volume):
    """
    Méthode 'First-Peak' : Trouve le PREMIER obstacle significatif.
    Évite de plonger sur les vaisseaux profonds qui brillent fort.
    """
    nx, ny, nz = volume.shape
    surface_map = np.full((nx, ny), nz) # Par défaut au fond
    
    print("   🚀 Détection Surface : Stratégie 'Premier Pic'...")

    # Seuil de bruit : On ignore tout ce qui est en dessous de ça.
    seuil_bruit_absolu = np.mean(volume) + 3 * np.std(volume)
    
    for i in range(nx):
        slice_img = volume[i, :, :] # (NY, NZ)
        
        for y in range(ny):
            col = slice_img[y, :]
            
            # 1. Trouver TOUS les pics dans la colonne
            # height=seuil : On ne veut que les pics qui dépassent le bruit
            # distance=10 : Deux pics doivent être séparés d'au moins 10 pixels (évite le bruit local)
            peaks, properties = find_peaks(col, height=seuil_bruit_absolu, distance=10)
            
            if len(peaks) > 0:
                # 2. Prendre le PREMIER pic (le plus superficiel / petit index Z)
                # C'est la différence clé avec argmax !
                first_peak_idx = peaks[0]
                peak_height = properties['peak_heights'][0]
                
                # 3. Affiner : Trouver le flanc montant (Rising Edge) de CE pic là
                # On remonte vers 0 jusqu'à ce que le signal tombe sous 20% de la hauteur de ce pic
                seuil_local = peak_height * 0.2
                
                # On regarde la portion avant le pic
                pre_peak = col[:first_peak_idx]
                
                # On cherche où ça passe sous le seuil
                under_thresh = np.where(pre_peak < seuil_local)[0]
                
                if len(under_thresh) > 0:
                    z_surface = under_thresh[-1]
                    surface_map[i, y] = z_surface
                else:
                    # Si ça commence fort dès le pixel 0
                    surface_map[i, y] = 0
            else:
                # Aucun pic détecté (vide)
                surface_map[i, y] = nz

    return surface_map

def analyse_surface_et_poils():
    print(f"--- 🕵️ ANALYSE : PREMIER PIC + REBOUCHAGE ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier introuvable.")
        return

    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume = mat[key]
    nx, ny, nz = volume.shape 
    
    # --- 1. DETECTION SURFACE (PREMIER PIC) ---
    surface_brute = detecter_surface_premier_pic(volume)
    
    # --- 2. CONSOLIDATION (REBOUCHAGE DES TROUS) ---
    print("   🔧 Consolidation de la peau (Morphological Closing)...")
    
    # A. Closing : Si la ligne a un "trou" (dip) plus petit que 'size', elle est rebouchée.
    # size=(1, 30) -> On bouche les trous de 30 pixels de large sur l'axe Y.
    # Cela force la peau à être "tendue" au-dessus des trous.
    surface_bouchee = grey_closing(surface_brute, size=(1, 30))
    
    # B. Lissage final (Median) pour enlever les petits escaliers
    surface_peau_estimee = median_filter(surface_bouchee, size=10)

    # --- 3. EXTRACTION P ---
    print("✨ Extraction des candidats poils...")
    
    # Seuil Poil Absolu (Top 0.2%)
    seuil_poil_absolu = np.percentile(volume, 99.8)

    Z_indices = np.arange(nz).reshape(1, 1, nz)
    surface_broad = surface_peau_estimee[:, :, np.newaxis]
    dist_map = Z_indices - surface_broad
    
    cond_lum = volume > seuil_poil_absolu
    
    # Critère Géométrique :
    # On est très conservateur maintenant : 
    # Uniquement ce qui DÉPASSE (-30) ou ce qui est TRÈS superficiel (+MARGE)
    # Grâce au rebouchage, la peau ne devrait plus être SOUS les poils profonds.
    cond_geo = (dist_map > -30) & (dist_map < MARGE_PIXELS)
    
    masque_poils_learning = cond_lum & cond_geo
    
    nb_pixels = np.sum(masque_poils_learning)
    print(f"👉 Pixels 'Poils Sûrs' : {nb_pixels}")

    # ==========================================
    # VISUALISATION
    # ==========================================
    idx_coupe = nx // 2
    coupe_vol = volume[idx_coupe, :, :].T
    masque_coupe = masque_poils_learning[idx_coupe, :, :].T
    
    # On compare la version "Brute" (Premier Pic) et "Corrigée" (Rebouchée)
    ligne_brute = surface_brute[idx_coupe, :]
    ligne_corrigee = surface_peau_estimee[idx_coupe, :]

    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.title(f"1. Surface : Rouge=Premier Pic (Risque trous), Vert=Rebouché (Closing)")
    plt.imshow(coupe_vol, cmap='gray', aspect='auto', vmax=np.percentile(coupe_vol, 99))
    plt.plot(ligne_brute, color='red', linewidth=0.8, alpha=0.6, label='Premier Pic')
    plt.plot(ligne_corrigee, color='lime', linewidth=2, linestyle='--', label='Consolidée (Closing)')
    plt.legend()
    plt.ylabel("Z")

    plt.subplot(3, 1, 2)
    plt.title(f"2. Masque P (Doit être vide sous la peau)")
    plt.imshow(masque_coupe, cmap='gray', aspect='auto')
    plt.ylabel("Z")
    
    plt.subplot(3, 1, 3)
    plt.title("3. Superposition")
    plt.imshow(coupe_vol, cmap='gray', aspect='auto', vmax=np.percentile(coupe_vol, 99))
    overlay = np.zeros((masque_coupe.shape[0], masque_coupe.shape[1], 4))
    overlay[masque_coupe == 1] = [1, 0, 0, 0.8]
    plt.imshow(overlay, aspect='auto')
    plt.ylabel("Z")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyse_surface_et_poils()