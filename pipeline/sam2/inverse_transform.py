import os
import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_MAT = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
FICHIER_METADATA = "transformation_metadata.json"
FICHIER_MASQUE_SAM = "masque_sam2_brut.npz"
FICHIER_SORTIE_MAT = "masque_peau_final.mat"
DOSSIER_VERIF = "verif_overlay"
# ==========================================

def appliquer_transformation_inverse():
    print("--- 🔄 INVERSION GÉOMÉTRIQUE ET GÉNÉRATION DE L'OVERLAY ---")

    # 1. Chargement
    print(f"📂 Chargement des données brutes, du masque SAM 2 et des métadonnées...")
    with open(FICHIER_METADATA, 'r') as f:
        meta = json.load(f)

    masque_1024 = np.load(FICHIER_MASQUE_SAM)['masque'] # Format (Frames, 1024, 1024)

    mat = scipy.io.loadmat(FICHIER_MAT)
    key = [k for k in mat.keys() if isinstance(mat[k], np.ndarray) and mat[k].ndim == 3 and not k.startswith('__')][0]
    volume_brut = mat[key]
    nx_brut, ny_brut, nz_brut = volume_brut.shape

    # 2. Retrait du Padding (Adapté à l'échelle 1024)
    print("✂️ Retrait du padding (Conversion des marges vers l'échelle 1024)...")
    w_crop = meta["format_sam2"]["shape_rognee"]["largeur_w"]
    h_crop = meta["format_sam2"]["shape_rognee"]["hauteur_h"]
    
    # On calcule le ratio qui avait été utilisé pour compresser en 1024
    taille_max = max(w_crop, h_crop)
    ratio_scale = 1024.0 / taille_max

    # On transpose les marges du JSON vers l'univers 1024x1024
    pad_top_1024 = round(meta["format_sam2"]["padding"]["top"] * ratio_scale)
    pad_left_1024 = round(meta["format_sam2"]["padding"]["left"] * ratio_scale)
    h_utile_1024 = round(h_crop * ratio_scale)
    w_utile_1024 = round(w_crop * ratio_scale)

    # On découpe la zone utile directement dans le 1024
    masque_sans_pad = masque_1024[:, pad_top_1024 : pad_top_1024 + h_utile_1024, pad_left_1024 : pad_left_1024 + w_utile_1024]

    # 3. Redimensionnement Exact
    print("📏 Redimensionnement vers la taille d'origine (Nearest Neighbor)...")
    # On calcule le ratio exact pour retomber sur (256, 2562)
    facteur_zoom_h = h_crop / masque_sans_pad.shape[1]
    facteur_zoom_w = w_crop / masque_sans_pad.shape[2]

    # Interpolation order=0 pour garder des zéros et des uns stricts
    masque_redimensionne = zoom(masque_sans_pad, (1.0, facteur_zoom_h, facteur_zoom_w), order=0)

    # 4. Reconstitution du Volume 3D
    print("🧩 Reconstitution du volume 3D...")
    # Le masque est actuellement (Y, Z, X). On transpose pour coller au DICOM (X, Y, Z).
    masque_3d = np.transpose(masque_redimensionne, (2, 0, 1))

    angle = meta["angle_rotation_applique"]
    x_min = meta["rognage_x"]["min"]

    if angle != 0:
        print(f"🔄 Application de la rotation initiale ({angle:.2f}°) sur le volume brut pour l'alignement...")
        volume_travail = rotate(volume_brut, angle, axes=(0, 1), reshape=True, order=1)
    else:
        volume_travail = volume_brut.copy()

    masque_final = np.zeros_like(volume_travail, dtype=bool)
    
    # ⚡ SÉCURITÉ GÉOMÉTRIQUE : On s'assure que X et Y collent parfaitement (gère l'image 2287 supprimée)
    nx_masque = masque_3d.shape[0]
    ny_masque = masque_3d.shape[1] # C'est ici que l'image supprimée est compensée
    nz_masque = masque_3d.shape[2]
    
    masque_final[x_min : x_min + nx_masque, :ny_masque, :nz_masque] = masque_3d

    # Si on a tourné le volume avant SAM 2, on tourne le masque dans le sens inverse
    if angle != 0:
        print(f"🔄 Inversion de la rotation de {-angle:.2f}°...")
        masque_final = rotate(masque_final.astype(np.uint8), -angle, axes=(0, 1), reshape=True, order=0)
        # On crop pour retomber strictement sur la taille DICOM originale
        masque_final = masque_final[:nx_brut, :ny_brut, :nz_brut]

    # 5. Sauvegarde
    print(f"💾 Sauvegarde du masque final {masque_final.shape} dans '{FICHIER_SORTIE_MAT}'...")
    scipy.io.savemat(FICHIER_SORTIE_MAT, {"masque_sam2_peau": masque_final.astype(np.uint8)})

    # 6. Génération des Images de Vérification
    print(f"📸 Génération des images d'overlay dans '{DOSSIER_VERIF}'...")
    os.makedirs(DOSSIER_VERIF, exist_ok=True)

    frames_a_tester = np.linspace(0, ny_brut - 1, 5, dtype=int)

    for y in frames_a_tester:
        slice_brute = volume_brut[:, y, :]
        slice_masque = masque_final[:, y, :]

        plt.figure(figsize=(10, 5), facecolor='black')
        plt.imshow(slice_brute.T, cmap='gray', aspect='auto') 
        
        overlay = np.zeros((nz_brut, nx_brut, 4))
        overlay[..., 0] = 1.0  
        overlay[..., 3] = (slice_masque.T) * 0.4  

        plt.imshow(overlay, aspect='auto')
        plt.title(f"Overlay Frame Y={y}", color='white')
        plt.axis('off')
        plt.tight_layout()
        
        chemin_img = os.path.join(DOSSIER_VERIF, f"overlay_Y_{y}.jpg")
        plt.savefig(chemin_img, dpi=200, facecolor='black')
        plt.close()

    print("✅ TOUT EST TERMINÉ ! La boucle est bouclée.")

if __name__ == "__main__":
    appliquer_transformation_inverse()