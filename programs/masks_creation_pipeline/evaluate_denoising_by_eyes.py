import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Le fichier original
path_original = r"dicom_data/dicom/4261_fromdcm.mat"

# Le fichier résultat (généré par le script précédent)
path_denoised = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# ==========================================

def load_mat(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Fichier introuvable : {filepath}")
        return None
    data = scipy.io.loadmat(filepath)
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v
    return None

def show_comparison():
    print("Chargement des volumes...")
    vol_orig = load_mat(path_original)
    vol_deno = load_mat(path_denoised)

    if vol_orig is None or vol_deno is None:
        print("Erreur de chargement. Vérifie les noms de fichiers.")
        return

    # On prend une tranche au milieu
    nx = vol_orig.shape[0]
    idx = nx // 2
    
    # Extraction des images 2D
    img_orig = vol_orig[idx, :, :].T  # .T pour remettre l'image droite
    img_deno = vol_deno[idx, :, :].T

    # Réglage automatique du contraste (sinon l'image peut paraître noire)
    # On sature les 1% de pixels les plus brillants pour voir les détails sombres
    vmax = np.percentile(img_orig, 99)

    print(f"Affichage de la tranche {idx}/{nx}")

    plt.figure(figsize=(14, 8))

    # Image Originale
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_orig, cmap='gray', aspect='auto', vmax=vmax)
    plt.axis('off')

    # Image Denoised
    plt.subplot(1, 2, 2)
    plt.title("Denoised")
    plt.imshow(img_deno, cmap='gray', aspect='auto', vmax=vmax)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_comparison()