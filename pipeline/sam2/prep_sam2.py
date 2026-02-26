import scipy.io
import numpy as np
import cv2
import os
import time
import json
from scipy.ndimage import rotate
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION GÉNÉRALE
# ==========================================
FICHIER_MAT = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
DOSSIER_SORTIE = "frames_sam2"
FICHIER_METADATA = "transformation_metadata.json"

# --- PARAMÈTRES (VALIDÉS) ---
MULTIPLICATEUR_SEUIL_VIOLENT = 2
PERCENTILE_COUPURE_HAUTE = 96
MARGE_ROGNAGE_PIXELS = 0 
NB_WORKERS_ECRITURE = 32

# --- PARAMÈTRE SAM 2 ---
TAILLE_SAM2 = 1024
# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def calculer_angle_pca(volume, seuil):
    print("   🧠 Analyse de l'inclinaison du membre (PCA)...")
    masque_2d = np.any(volume > seuil, axis=2)
    y_coords, x_coords = np.nonzero(masque_2d)
    
    if len(x_coords) < 100:
        return 0.0
        
    points = np.column_stack((x_coords, y_coords))
    points_centres = points - np.mean(points, axis=0)
    cov_matrix = np.cov(points_centres, rowvar=False)
    
    valeurs_propres, vecteurs_propres = np.linalg.eigh(cov_matrix)
    vecteur_principal = vecteurs_propres[:, np.argmax(valeurs_propres)]
    
    angle_deg = np.degrees(np.arctan2(vecteur_principal[1], vecteur_principal[0]))
    
    if angle_deg > 45 and angle_deg < 135: angle_deg -= 90
    elif angle_deg < -45 and angle_deg > -135: angle_deg += 90
        
    print(f"   📐 Angle d'inclinaison détecté : {angle_deg:.2f}°")
    return angle_deg

def normaliser_tranche(tranche_2d, vmin, vmax):
    """Normalisation inversée : Le fond noir devient blanc (255), le signal devient noir (0)"""
    if vmax <= vmin: vmax = vmin + 1e-5 
    tranche_clipee = np.clip(tranche_2d, vmin, vmax)
    
    # ⚡ MODIFICATION ICI : Inversion mathématique de l'intensité
    tranche_normalisee = 255.0 - ((tranche_clipee - vmin) / (vmax - vmin) * 255.0)
    return tranche_normalisee.astype(np.uint8)

def formater_pour_sam2(image_8bit, pad_top, pad_bottom, pad_left, pad_right):
    """Ajoute les bandes blanches (padding) et redimensionne en 1024x1024"""
    # ⚡ MODIFICATION ICI : value=255 pour que le padding se fonde dans le nouveau fond blanc
    image_carree = cv2.copyMakeBorder(
        image_8bit, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, value=255
    )
    
    image_1024 = cv2.resize(image_carree, (TAILLE_SAM2, TAILLE_SAM2), interpolation=cv2.INTER_LANCZOS4)
    return np.stack((image_1024,) * 3, axis=-1)

def sauvegarder_image(args):
    cv2.imwrite(args[0], args[1])

def preparer_donnees_sam2():
    print(f"--- 🎬 PRÉPARATION VIDÉO (COULEURS INVERSÉES) ---")
    
    t0 = time.time()
    mat = scipy.io.loadmat(FICHIER_MAT)
    key = trouver_variable_volume(mat)
    volume = mat[key]
    shape_originale = volume.shape

    if not os.path.exists(DOSSIER_SORTIE): os.makedirs(DOSSIER_SORTIE)

    seuil_bruit = volume.mean() + MULTIPLICATEUR_SEUIL_VIOLENT * volume.std()
    pixels_utiles = volume[volume > seuil_bruit]
    vmax = np.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else np.max(volume)

    # --- 1. ROTATION ---
    angle_rotation = calculer_angle_pca(volume, seuil_bruit)
    if abs(angle_rotation) > 1.0:
        print(f"   🔄 Rotation du volume de {-angle_rotation:.2f}°...")
        volume = rotate(volume, -angle_rotation, axes=(0, 1), reshape=True, order=3)
    else:
        angle_rotation = 0.0

    # --- 2. ROGNAGE ---
    masque_global = volume > seuil_bruit
    signal_sur_x = np.any(masque_global, axis=(1, 2)) 
    x_indices = np.where(signal_sur_x)[0]
    
    if len(x_indices) > 0:
        x_min = max(0, x_indices[0] - MARGE_ROGNAGE_PIXELS)
        x_max = min(volume.shape[0], x_indices[-1] + MARGE_ROGNAGE_PIXELS)
        volume = volume[x_min:x_max, :, :]
    else:
        x_min, x_max = 0, volume.shape[0]

    nx, ny, nz = volume.shape
    
    # --- 3. CALCUL DU PADDING ---
    print("📐 Calcul de la géométrie du Carré Parfait...")
    H, W = nz, nx
    taille_max = max(H, W)
    
    pad_h = taille_max - H
    pad_w = taille_max - W
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # --- 4. SAUVEGARDE DU JSON ---
    print("📝 Sauvegarde du ticket d'inversion (JSON)...")
    metadata = {
        "shape_originale": shape_originale,
        "angle_rotation_applique": -angle_rotation,
        "rognage_x": {"min": int(x_min), "max": int(x_max)},
        "format_sam2": {
            "shape_rognee": {"largeur_w": int(W), "hauteur_h": int(H)},
            "padding": {
                "top": int(pad_top), 
                "bottom": int(pad_bottom), 
                "left": int(pad_left), 
                "right": int(pad_right)
            },
            "taille_finale": TAILLE_SAM2
        }
    }
    with open(FICHIER_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)

    # --- 5. GÉNÉRATION DES FRAMES ---
    print("⚙️ Génération des frames HD inversées...")
    images_a_sauvegarder = []
    
    for y in range(ny):
        coupe = volume[:, y, :].T 
        coupe_8bit = normaliser_tranche(coupe, seuil_bruit, vmax)
        coupe_sam2 = formater_pour_sam2(coupe_8bit, pad_top, pad_bottom, pad_left, pad_right)
        
        nom_fichier = f"{y:05d}.jpg"
        images_a_sauvegarder.append((os.path.join(DOSSIER_SORTIE, nom_fichier), coupe_sam2))
        
    print("💾 Sauvegarde sur le disque...")
    with ThreadPoolExecutor(max_workers=NB_WORKERS_ECRITURE) as executor:
        executor.map(sauvegarder_image, images_a_sauvegarder)
        
    print(f"✅ Vidéo prête en {time.time()-t0:.2f}s !")

if __name__ == "__main__":
    preparer_donnees_sam2()