import scipy.io
import numpy as np
import json
import os
from scipy.ndimage import median_filter, rotate

# ==========================================
# CONFIGURATION GÉNÉRALE
# ==========================================
FICHIER_MAT = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
FICHIER_METADATA = "transformation_metadata.json"
FICHIER_PROMPTS_SORTIE = "prompts_sam2_v4.json"
FICHIER_MASK_SORTIE = "mask_prompt_v4.npy"
FICHIER_REFERENCE_NPZ = "reference_geometrique.npz"

RESOLUTION_Z_MM = 0.12500   
EPAISSEUR_MAX_PEAU_MM = 1.5   
EPAISSEUR_PEAU_EN_PIXEL = EPAISSEUR_MAX_PEAU_MM / RESOLUTION_Z_MM

MULTIPLICATEUR_SEUIL_BAS = 2.0
PERCENTILE_COUPURE_HAUTE = 93.0  
TAILLE_FILTRE_SURFACE = 30       
PERCENTILE_REJET_EXTREMES = 1.0  # Ajusté à 1.0% selon ta recommandation

MARGE_Y_BORDURES = 0.10          
LARGEUR_PEAU_MIN = 0.30          
# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'): return k
    return None

def generer_prompts_v4():
    print("--- 🎯 GÉNÉRATION V4 : MASK PROMPTING (LE RUBAN) ---")
    
    with open(FICHIER_METADATA, 'r') as f: meta = json.load(f)
    mat = scipy.io.loadmat(FICHIER_MAT)
    volume = mat[trouver_variable_volume(mat)]
    
    angle = meta["angle_rotation_applique"]
    if abs(angle) > 0: volume = rotate(volume, angle, axes=(0, 1), reshape=True, order=1)
    volume = volume[meta["rognage_x"]["min"]:meta["rognage_x"]["max"], :, :]
    nx, ny, nz = volume.shape

    seuil_bas = volume.mean() + MULTIPLICATEUR_SEUIL_BAS * volume.std()
    pixels_utiles = volume[volume > seuil_bas]
    seuil_haut = np.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else np.max(volume)
    
    masque_brillant = (volume > seuil_bas) & (volume < seuil_haut)
    surface_brute = np.argmax(masque_brillant, axis=2)
    colonnes_vides = np.max(volume, axis=2) < seuil_bas
    surface_brute[colonnes_vides] = nz 
    surface_lisse = median_filter(surface_brute, size=TAILLE_FILTRE_SURFACE)
    erreur = np.abs(surface_brute.astype(float) - surface_lisse.astype(float))

    purete_par_tranche = np.zeros(ny)
    largeur_par_tranche = np.zeros(ny)
    marge_y_pixels = int(ny * MARGE_Y_BORDURES)
    
    pad_top = meta["format_sam2"]["padding"]["top"]
    pad_left = meta["format_sam2"]["padding"]["left"]
    taille_max = max(meta["format_sam2"]["shape_rognee"]["largeur_w"], meta["format_sam2"]["shape_rognee"]["hauteur_h"])
    ratio_scale = 1024.0 / taille_max
    projeter = lambda val, pad: int((val + pad) * ratio_scale)
    
    matrice_reference_1024 = np.full((ny, 1024), -1.0, dtype=np.float32)
    
    print("⚙️ Évaluation et création de la Vérité Terrain...")
    for y in range(marge_y_pixels, ny - marge_y_pixels):
        masque_signal = ~colonnes_vides[:, y]
        pixels_actifs = np.where(masque_signal)[0]
        if len(pixels_actifs) > 0:
            largeur_par_tranche[y] = pixels_actifs[-1] - pixels_actifs[0] + 1
            masque_pur = (erreur[:, y] <= EPAISSEUR_PEAU_EN_PIXEL) & masque_signal
            pixels_purs_x = np.where(masque_pur)[0]
            purete_par_tranche[y] = len(pixels_purs_x) / len(pixels_actifs)
            for x_p in pixels_purs_x:
                x_1024 = projeter(x_p, pad_left)
                if 0 <= x_1024 < 1024:
                    matrice_reference_1024[y, x_1024] = projeter(surface_lisse[x_p, y], pad_top)

    candidats_y = np.where(largeur_par_tranche >= nx * LARGEUR_PEAU_MIN)[0]
    golden_frame_idx = int(candidats_y[np.argmax(purete_par_tranche[candidats_y])]) if len(candidats_y) > 0 else ny // 2
    print(f"   🏆 Golden Frame : {golden_frame_idx} (Pureté: {purete_par_tranche[golden_frame_idx]*100:.1f}%)")

    # --- CRÉATION DE LA BOUNDING BOX ET DU MASQUE RUBAN ---
    marge_bb_1024 = int(EPAISSEUR_PEAU_EN_PIXEL * ratio_scale)
    gf_masque_signal = ~colonnes_vides[:, golden_frame_idx]
    gf_surface_lisse = surface_lisse[:, golden_frame_idx]
    
    x_actifs = np.where(gf_masque_signal)[0]
    x_min_raw = np.percentile(x_actifs, PERCENTILE_REJET_EXTREMES)
    x_max_raw = np.percentile(x_actifs, 100 - PERCENTILE_REJET_EXTREMES)
    x_actifs_filtres = x_actifs[(x_actifs >= x_min_raw) & (x_actifs <= x_max_raw)]
    
    z_actifs = gf_surface_lisse[x_actifs_filtres]
    z_min_raw, z_max_raw = z_actifs.min(), z_actifs.max()
    
    box_x_min = max(0, projeter(x_min_raw, pad_left) - marge_bb_1024)
    box_x_max = min(1023, projeter(x_max_raw, pad_left) + marge_bb_1024)
    box_y_min = max(0, projeter(z_min_raw, pad_top) - marge_bb_1024)
    # On définit une limite basse "raisonnable" pour le ruban (ex: z_max + 40 pixels de marge)
    epaisseur_ruban_1024 = int(40 * ratio_scale) 
    box_y_max = min(1023, projeter(z_max_raw, pad_top) + epaisseur_ruban_1024)
    
    bounding_box = [box_x_min, box_y_min, box_x_max, box_y_max]
    
    # ⚡ NOUVEAUTÉ V4 : Génération de l'image Masque 1024x1024
    mask_prompt = np.zeros((1024, 1024), dtype=np.float32)
    
    gf_erreur = erreur[:, golden_frame_idx]
    pixels_purs_gf = np.where((gf_erreur <= EPAISSEUR_PEAU_EN_PIXEL) & gf_masque_signal)[0]
    
    for x_brut in pixels_purs_gf:
        x_1024 = projeter(x_brut, pad_left)
        z_1024 = projeter(gf_surface_lisse[x_brut], pad_top)
        
        if 0 <= x_1024 < 1024:
            # On étire le pixel vers le bas jusqu'à la limite de la bounding box
            mask_prompt[z_1024 : box_y_max, x_1024] = 1.0

    # Sauvegardes
    np.savez_compressed(FICHIER_REFERENCE_NPZ, pure_z_1024=matrice_reference_1024)
    np.save(FICHIER_MASK_SORTIE, mask_prompt) # Le Ruban en fichier binaire
    
    with open(FICHIER_PROMPTS_SORTIE, 'w') as f:
        json.dump({"initial_frame_index": golden_frame_idx, "box": bounding_box}, f, indent=4)
        
    print(f"✅ Ruban généré et sauvegardé dans '{FICHIER_MASK_SORTIE}' !")

if __name__ == "__main__":
    generer_prompts_v4()