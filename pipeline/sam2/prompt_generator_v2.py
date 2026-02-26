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
FICHIER_PROMPTS_SORTIE = "prompts_sam2.json"

# --- PARAMÈTRES BIOLOGIQUES & PHYSIQUES ---
RESOLUTION_Z_MM = 0.12500   
EPAISSEUR_MAX_PEAU_MM = 1.5   
MARGE_HAUTE_POILS_MM = 0.5    

# 🚨 LA LIMITE BIOLOGIQUE ABSOLUE
EPAISSEUR_PEAU_EN_PIXEL = EPAISSEUR_MAX_PEAU_MM / RESOLUTION_Z_MM

# --- PARAMÈTRES D'EXTRACTION DE SURFACE ---
MULTIPLICATEUR_SEUIL_BAS = 2.0
PERCENTILE_COUPURE_HAUTE = 93.0  
TAILLE_FILTRE_SURFACE = 60       

# --- PARAMÈTRES DE LA BOUNDING BOX ---
PERCENTILE_REJET_EXTREMES = 1.0  # Coupe X% des pixels isolés à gauche et à droite pour éviter d'étirer la Box

# --- PARAMÈTRES D'INTELLIGENCE CLINIQUE ---
MARGE_Y_BORDURES = 0.10          
LARGEUR_PEAU_MIN = 0.30          
ESPACEMENT_MIN_Y_POURCENT = 0.05 
ALPHA_MMR = 0.7                  

MAX_GOLDEN_FRAMES = 5            
NB_POINTS_PROMPT = 15            
# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def generer_prompts():
    print("--- 🎯 GÉNÉRATION DU PROMPT (POINTS + BOUNDING BOX ROBUSTE) ---")
    
    with open(FICHIER_METADATA, 'r') as f:
        meta = json.load(f)
        
    mat = scipy.io.loadmat(FICHIER_MAT)
    key = trouver_variable_volume(mat)
    volume = mat[key]
    
    angle = meta["angle_rotation_applique"]
    if abs(angle) > 0:
        volume = rotate(volume, angle, axes=(0, 1), reshape=True, order=1)
        
    x_min = meta["rognage_x"]["min"]
    x_max = meta["rognage_x"]["max"]
    volume = volume[x_min:x_max, :, :]
    
    nx, ny, nz = volume.shape

    print("🕵️ Extraction de la ligne de peau...")
    seuil_bas = volume.mean() + MULTIPLICATEUR_SEUIL_BAS * volume.std()
    
    pixels_utiles = volume[volume > seuil_bas]
    seuil_haut = np.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else np.max(volume)
    
    masque_brillant = (volume > seuil_bas) & (volume < seuil_haut)
    surface_brute = np.argmax(masque_brillant, axis=2)
    
    colonnes_vides = np.max(volume, axis=2) < seuil_bas
    surface_brute[colonnes_vides] = nz 
    surface_lisse = median_filter(surface_brute, size=TAILLE_FILTRE_SURFACE)
    
    erreur = np.abs(surface_brute.astype(float) - surface_lisse.astype(float))

    print("\n📊 Évaluation clinique des tranches...")
    purete_par_tranche = np.zeros(ny)
    largeur_par_tranche = np.zeros(ny)
    marge_y_pixels = int(ny * MARGE_Y_BORDURES)
    
    for y in range(marge_y_pixels, ny - marge_y_pixels):
        masque_signal = ~colonnes_vides[:, y]
        pixels_actifs = np.where(masque_signal)[0]
        
        if len(pixels_actifs) > 0:
            largeur = pixels_actifs[-1] - pixels_actifs[0] + 1
            largeur_par_tranche[y] = largeur
            erreurs_y = erreur[:, y]
            nb_pixels_purs = np.sum((erreurs_y <= EPAISSEUR_PEAU_EN_PIXEL) & masque_signal)
            purete_par_tranche[y] = nb_pixels_purs / len(pixels_actifs)

    candidats_y = np.where(largeur_par_tranche >= nx * LARGEUR_PEAU_MIN)[0]
    golden_frames_y = []
    
    if len(candidats_y) == 0:
        golden_frames_y.append(ny // 2)
    else:
        meilleur_y_initial = candidats_y[np.argmax(purete_par_tranche[candidats_y])]
        golden_frames_y.append(meilleur_y_initial)
        espacement_min_pixels = int(ny * ESPACEMENT_MIN_Y_POURCENT)
        
        while len(golden_frames_y) < MAX_GOLDEN_FRAMES:
            meilleur_score = -1.0
            meilleure_candidature = -1
            for y_cand in candidats_y:
                if y_cand in golden_frames_y: continue
                if any(abs(y_cand - gf) <= espacement_min_pixels for gf in golden_frames_y): continue
                
                distance_min = min(abs(y_cand - gf) for gf in golden_frames_y)
                score = (ALPHA_MMR * purete_par_tranche[y_cand]) + ((1 - ALPHA_MMR) * (distance_min / ny))
                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_candidature = y_cand
            if meilleure_candidature == -1: break 
            golden_frames_y.append(meilleure_candidature)

    golden_frames_y.sort()

    pad_top = meta["format_sam2"]["padding"]["top"]
    pad_left = meta["format_sam2"]["padding"]["left"]
    taille_max = max(meta["format_sam2"]["shape_rognee"]["largeur_w"], meta["format_sam2"]["shape_rognee"]["hauteur_h"])
    ratio_scale = 1024.0 / taille_max
    
    offset_bas_pixels = int(EPAISSEUR_MAX_PEAU_MM / RESOLUTION_Z_MM)
    offset_haut_pixels = int(MARGE_HAUTE_POILS_MM / RESOLUTION_Z_MM)
    
    # Marge de la Bounding Box mise à l'échelle
    marge_bb_1024 = int(EPAISSEUR_PEAU_EN_PIXEL * ratio_scale)

    prompts_finaux = []

    for gf_y in golden_frames_y:
        print(f"\n   ⚙️ Traitement Frame Y={gf_y}")
        erreur_sur_golden = erreur[:, gf_y]
        surface_golden = surface_lisse[:, gf_y]
        masque_signal = ~colonnes_vides[:, gf_y]
        
        # --- CALCUL DE LA BOUNDING BOX GLOBALE (AVEC PERCENTILES ANTI-BRUIT) ---
        x_actifs = np.where(masque_signal)[0]
        if len(x_actifs) > 0:
            # ⚡ UTILISATION DES PERCENTILES
            x_min_raw = np.percentile(x_actifs, PERCENTILE_REJET_EXTREMES)
            x_max_raw = np.percentile(x_actifs, 100 - PERCENTILE_REJET_EXTREMES)
            
            # On filtre les indices X pour ne garder que ceux dans la bonne plage
            x_actifs_filtres = x_actifs[(x_actifs >= x_min_raw) & (x_actifs <= x_max_raw)]
            
            # On va chercher les hauteurs (Z) correspondant exactement à ces X
            z_actifs = surface_golden[x_actifs_filtres]

            if len(z_actifs) > 0:
                z_min_raw, z_max_raw = z_actifs.min(), z_actifs.max()
            else:
                z_min_raw, z_max_raw = surface_golden[x_actifs].min(), surface_golden[x_actifs].max()
            
            projeter = lambda val, pad: int((val + pad) * ratio_scale)
            
            # Projection en 1024 et ajout de l'épaisseur de peau comme marge
            box_x_min = max(0, projeter(x_min_raw, pad_left) - marge_bb_1024)
            box_x_max = min(1024, projeter(x_max_raw, pad_left) + marge_bb_1024)
            box_y_min = max(0, projeter(z_min_raw, pad_top) - marge_bb_1024)
            box_y_max = min(1024, projeter(z_max_raw, pad_top) + marge_bb_1024)
            
            bounding_box = [box_x_min, box_y_min, box_x_max, box_y_max]
        else:
            bounding_box = [0, 0, 1024, 1024]
        
        # --- CALCUL DES POINTS D'AMORCE ---
        seuil_pixel_actuel = 0.5 
        pixels_valides_x = []
        while seuil_pixel_actuel <= EPAISSEUR_PEAU_EN_PIXEL:
            pixels_valides_x = np.where((erreur_sur_golden <= seuil_pixel_actuel) & masque_signal)[0]
            if len(pixels_valides_x) >= NB_POINTS_PROMPT: break
            seuil_pixel_actuel += 0.5
            
        if len(pixels_valides_x) < NB_POINTS_PROMPT:
            if len(pixels_valides_x) == 0: pixels_valides_x = x_actifs 

        x_min_valide, x_max_valide = pixels_valides_x.min(), pixels_valides_x.max()
        grille_ideale_x = np.linspace(x_min_valide, x_max_valide, NB_POINTS_PROMPT)
        points_finaux_x = sorted(list(set(pixels_valides_x[(np.abs(pixels_valides_x[:, None] - grille_ideale_x)).argmin(axis=0)])))

        prompts_points = []
        prompts_labels = []

        for x_brut in points_finaux_x:
            z_brut = surface_golden[x_brut]
            x_1024 = projeter(x_brut, pad_left)
            prompts_points.extend([[x_1024, projeter(z_brut, pad_top)], 
                                   [x_1024, projeter(max(0, z_brut - offset_haut_pixels), pad_top)], 
                                   [x_1024, projeter(min(nz - 1, z_brut + offset_bas_pixels), pad_top)]])
            prompts_labels.extend([1, 0, 0])

        prompts_finaux.append({
            "frame_index": int(gf_y), 
            "points": prompts_points, 
            "labels": prompts_labels,
            "box": bounding_box
        })

    with open(FICHIER_PROMPTS_SORTIE, 'w') as f:
        json.dump({"nb_golden_frames": len(prompts_finaux), "prompts": prompts_finaux}, f, indent=4)
        
    print(f"\n✅ Prompts + Bounding Box générés avec succès !")

if __name__ == "__main__":
    generer_prompts()