import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_multiotsu

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v
    raise ValueError("Aucun volume 3D trouvé.")

def apply_smart_bandpass_color(volume):
    print("--- 🌈 DIAGNOSTIC COULEUR (MULTI-OTSU) ---")
    
    nx, ny, nz = volume.shape
    idx_slice = nx // 2
    slice_2d = volume[idx_slice, :, :].T 
    
    # 1. Calculs Otsu
    pixels = slice_2d.ravel()
    print("   📊 Calcul des seuils optimaux...")
    thresholds = threshold_multiotsu(slice_2d, classes=3)
    
    seuil_bas = thresholds[0]
    seuil_haut = thresholds[1]
    
    print(f"      - Seuil Bas (Vide): {seuil_bas:.2f}")
    print(f"      - Seuil Haut (Poils): {seuil_haut:.2f}")
    
    # 2. Création du masque (Passe-bande)
    mask_peau_smart = (slice_2d > seuil_bas) & (slice_2d < seuil_haut)
    
    # 3. Image Nettoyée (Valeurs réelles)
    img_cleaned = np.zeros_like(slice_2d)
    img_cleaned[mask_peau_smart] = slice_2d[mask_peau_smart]
    
    # ASTUCE : Pour l'affichage couleur, on masque les 0 (le vide)
    # pour qu'ils apparaissent blancs ou transparents, et ne faussent pas l'échelle de couleur
    img_masked_display = np.ma.masked_where(img_cleaned == 0, img_cleaned)
    
    # 4. Visualisation
    visualize_color_diagnostic(slice_2d, mask_peau_smart, img_masked_display, seuil_bas, seuil_haut, pixels)

def visualize_color_diagnostic(original, mask, img_masked, t_low, t_high, pixels):
    plt.figure(figsize=(18, 12))
    
    # A. Histogramme
    plt.subplot(2, 2, 1)
    plt.title("Histogramme & Plage Sélectionnée")
    plt.hist(pixels, bins=100, log=True, color='lightgray')
    plt.axvline(t_low, color='blue', linestyle='--', linewidth=2, label='Seuil Bas')
    plt.axvline(t_high, color='red', linestyle='--', linewidth=2, label='Seuil Haut')
    plt.axvspan(t_low, t_high, color='green', alpha=0.1, label='Zone Peau')
    plt.legend()
    
    # B. Masque Binaire
    plt.subplot(2, 2, 2)
    plt.title("Masque Binaire (Forme)")
    plt.imshow(mask, cmap='gray', aspect='auto')
    
    # C. ANALYSE COULEUR (ZOOM)
    # C'est ici qu'on regarde la différence Vaisseaux vs Peau
    plt.subplot(2, 1, 2) # Prend toute la largeur en bas
    plt.title(f"Analyse Intensité (Spectrale) : Entre {t_low:.1f} et {t_high:.1f}")
    
    # 'nipy_spectral' offre un contraste énorme entre les niveaux proches
    im = plt.imshow(img_masked, cmap='nipy_spectral', aspect='auto')
    plt.colorbar(im, label="Intensité du Pixel")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    volume = load_volume(fichier_mat)
    apply_smart_bandpass_color(volume)