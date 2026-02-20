import scipy.io
import numpy as np
import cv2
import os
import time
import json
from scipy.ndimage import rotate, binary_opening
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION GÉNÉRALE
# ==========================================
FICHIER_MAT = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
DOSSIER_SORTIE = "frames_sam2"
FICHIER_METADATA = "transformation_metadata.json"

# --- PARAMÈTRES DE BASE ---
MULTIPLICATEUR_SEUIL_VIOLENT = 10.0 
PERCENTILE_COUPURE_HAUTE = 99.5 
MARGE_ROGNAGE_PIXELS = 0 
NB_WORKERS_ECRITURE = 32

# --- NOUVEAUX PARAMÈTRES DE CONTRASTE INTELLIGENT ---
# 1.0 = Linéaire (sombre). 0.5 = Booste fortement les gris/nuances sombres de la peau.
CORRECTION_GAMMA = 0.5 

# Active l'égalisation locale d'histogramme (Très puissant pour faire ressortir les textures PA)
ACTIVER_CLAHE = True 
# ==========================================

# Initialisation du filtre CLAHE (si activé)
# clipLimit empêche le bruit de fond d'exploser. tileGridSize est la taille d'analyse locale.
clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

# ... (La fonction calculer_angle_pca reste inchangée, je ne la remets pas pour raccourcir) ...

def normaliser_tranche_rgb(tranche_2d, vmin, vmax):
    if vmax <= vmin: vmax = vmin + 1e-5 
    
    # 1. Clipping
    tranche_clipee = np.clip(tranche_2d, vmin, vmax)
    
    # 2. Normalisation entre 0.0 et 1.0
    tranche_norm = (tranche_clipee - vmin) / (vmax - vmin)
    
    # 3. CORRECTION GAMMA (La magie non-linéaire)
    # Ex: Si la peau vaut 0.1 (très sombre), avec Gamma 0.5, elle devient 0.1^0.5 = 0.31 (beaucoup plus clair !)
    tranche_gamma = np.power(tranche_norm, CORRECTION_GAMMA)
    
    # 4. Conversion 8-bit
    tranche_8bit = (tranche_gamma * 255.0).astype(np.uint8)
    
    # 5. FILTRE CLAHE (Optimisation des textures locales)
    if ACTIVER_CLAHE:
        tranche_8bit = clahe_filter.apply(tranche_8bit)
        
    # 6. RGB
    return np.stack((tranche_8bit,) * 3, axis=-1)

def sauvegarder_image(args):
    cv2.imwrite(args[0], args[1])

def preparer_donnees_sam2():
    print(f"--- 🎬 PRÉPARATION VIDÉO (CONTRASTE AVANCÉ & ROGNAGE ROBUSTE) ---")
    
    t0 = time.time()
    mat = scipy.io.loadmat(FICHIER_MAT)
    key = trouver_variable_volume(mat)
    volume = mat[key]
    shape_originale = volume.shape

    if not os.path.exists(DOSSIER_SORTIE): os.makedirs(DOSSIER_SORTIE)

    seuil_bruit = volume.mean() + MULTIPLICATEUR_SEUIL_VIOLENT * volume.std()
    pixels_utiles = volume[volume > seuil_bruit]
    vmax = np.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else np.max(volume)

    # --- ÉTAPE 1 : ROTATION (Simulée comme absente si tu l'as enlevée, mais la logique reste la même) ---
    angle_rotation = 0.0 # Remplace par l'appel à calculer_angle_pca si tu gardes la rotation
    
    # --- ÉTAPE 2 : ROGNAGE INTELLIGENT (Insensible aux pixels isolés) ---
    print("✂️ Rognage des bordures noires...")
    masque_global = volume > seuil_bruit
    
    # On regarde l'axe X (largeur). On compte les pixels allumés.
    signal_sur_x = np.sum(masque_global, axis=(1, 2)) 
    
    # CORRECTION DU VIDE : On ignore les tranches qui ont moins de 50 pixels allumés
    # Cela détruit instantanément l'influence des poussières/bruits isolés aux extrémités
    masque_x_solide = signal_sur_x > 50 
    
    x_indices = np.where(masque_x_solide)[0]
    
    if len(x_indices) > 0:
        x_min = max(0, x_indices[0] - MARGE_ROGNAGE_PIXELS)
        x_max = min(volume.shape[0], x_indices[-1] + MARGE_ROGNAGE_PIXELS)
        
        volume = volume[x_min:x_max, :, :]
        print(f"   🎯 Volume rogné sur X : de {signal_sur_x.shape[0]} px à {volume.shape[0]} px de large")
    else:
        x_min, x_max = 0, volume.shape[0]

    # --- ÉTAPE 3 : METADATA ---
    metadata = {
        "shape_originale": shape_originale,
        "angle_rotation_applique": -angle_rotation,
        "rognage_x": {"min": int(x_min), "max": int(x_max)}
    }
    with open(FICHIER_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)

    # --- ÉTAPE 4 : JPEGS ---
    print("⚙️ Génération des frames avec Gamma et CLAHE...")
    nx, ny, nz = volume.shape
    images_a_sauvegarder = []
    
    for y in range(ny):
        coupe = volume[:, y, :].T 
        coupe_rgb = normaliser_tranche_rgb(coupe, seuil_bruit, vmax)
        nom_fichier = f"{y:05d}.jpg"
        images_a_sauvegarder.append((os.path.join(DOSSIER_SORTIE, nom_fichier), coupe_rgb))
        
    print("💾 Sauvegarde sur le disque...")
    with ThreadPoolExecutor(max_workers=NB_WORKERS_ECRITURE) as executor:
        executor.map(sauvegarder_image, images_a_sauvegarder)
        
    print(f"✅ Prêt en {time.time()-t0:.2f}s !")

if __name__ == "__main__":
    preparer_donnees_sam2()