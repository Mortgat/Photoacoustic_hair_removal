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

# --- PARAMÈTRES D'INTELLIGENCE CLINIQUE (SÉLECTION TEMPORELLE) ---
MARGE_Y_BORDURES = 0.10          # Exclut 10% du début et de la fin (Approche sonde)
LARGEUR_PEAU_MIN = 0.30          # Exige que la peau couvre au moins 30% de la largeur X
ESPACEMENT_MIN_Y_POURCENT = 0.05 # Veto (NMS) : 5% d'écart strict entre deux frames
ALPHA_MMR = 0.7                  # Arbitrage : 70% importance Pureté / 30% importance Distance

MAX_GOLDEN_FRAMES = 5            # Limite stricte pour SAM 2
NB_POINTS_PROMPT = 15            # Nombre de clics ciblés par frame
# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def generer_prompts():
    print("--- 🎯 GÉNÉRATION DU PROMPT (MMR & PHYSIQUE CLINIQUE) ---")
    print(f"🧬 Limite d'erreur biologique fixée à : {EPAISSEUR_PEAU_EN_PIXEL:.1f} pixels")

    # 1. Chargement et Alignement
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

    # 2. Extraction de la surface
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

    # =================================================================
    # 3. ÉVALUATION DE LA PURETÉ (OPTIMISÉE)
    # =================================================================
    print("\n📊 Évaluation clinique des tranches (Pureté & Largeur)...")
    purete_par_tranche = np.zeros(ny)
    largeur_par_tranche = np.zeros(ny)
    
    # ⚡ OPTIMISATION : On calcule la marge tout de suite
    marge_y_pixels = int(ny * MARGE_Y_BORDURES)
    
    # ⚡ OPTIMISATION : La boucle ne tourne QUE sur le cœur de la vidéo
    for y in range(marge_y_pixels, ny - marge_y_pixels):
        masque_signal = ~colonnes_vides[:, y]
        pixels_actifs = np.where(masque_signal)[0]
        
        if len(pixels_actifs) > 0:
            largeur = pixels_actifs[-1] - pixels_actifs[0] + 1
            largeur_par_tranche[y] = largeur
            
            erreurs_y = erreur[:, y]
            nb_pixels_purs = np.sum((erreurs_y <= EPAISSEUR_PEAU_EN_PIXEL) & masque_signal)
            purete_par_tranche[y] = nb_pixels_purs / len(pixels_actifs)

    # Filtrage direct (plus besoin de vérifier y_indices)
    candidats_y = np.where(largeur_par_tranche >= nx * LARGEUR_PEAU_MIN)[0]

    # =================================================================
    # 4. SÉLECTION TEMPORELLE (NMS + MMR)
    # =================================================================
    golden_frames_y = []
    
    if len(candidats_y) == 0:
        print("   ⚠️ Aucun candidat valide. Fallback sur le milieu de la vidéo.")
        golden_frames_y.append(ny // 2)
    else:
        print(f"   🌍 {len(candidats_y)} tranches candidates dans la zone de recherche.")
        
        meilleur_y_initial = candidats_y[np.argmax(purete_par_tranche[candidats_y])]
        golden_frames_y.append(meilleur_y_initial)
        
        espacement_min_pixels = int(ny * ESPACEMENT_MIN_Y_POURCENT)
        
        while len(golden_frames_y) < MAX_GOLDEN_FRAMES:
            meilleur_score = -1.0
            meilleure_candidature = -1
            
            for y_cand in candidats_y:
                if y_cand in golden_frames_y: continue
                
                # VETO (NMS)
                trop_proche = any(abs(y_cand - gf) <= espacement_min_pixels for gf in golden_frames_y)
                if trop_proche: continue
                
                # ARBITRAGE (MMR)
                distance_min = min(abs(y_cand - gf) for gf in golden_frames_y)
                distance_normalisee = distance_min / ny 
                
                score = (ALPHA_MMR * purete_par_tranche[y_cand]) + ((1 - ALPHA_MMR) * distance_normalisee)
                
                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_candidature = y_cand
                    
            if meilleure_candidature == -1:
                break 
                
            golden_frames_y.append(meilleure_candidature)

    golden_frames_y.sort()
    print(f"   🏆 {len(golden_frames_y)} Golden Frame(s) finale(s) : {golden_frames_y}")

    # =================================================================
    # 5. GÉNÉRATION DES PROMPTS (AIMANTATION X & RELÂCHEMENT Z)
    # =================================================================
    pad_top = meta["format_sam2"]["padding"]["top"]
    pad_left = meta["format_sam2"]["padding"]["left"]
    taille_max = max(meta["format_sam2"]["shape_rognee"]["largeur_w"], meta["format_sam2"]["shape_rognee"]["hauteur_h"])
    ratio_scale = 1024.0 / taille_max
    
    offset_bas_pixels = int(EPAISSEUR_MAX_PEAU_MM / RESOLUTION_Z_MM)
    offset_haut_pixels = int(MARGE_HAUTE_POILS_MM / RESOLUTION_Z_MM)

    prompts_finaux = []

    for gf_y in golden_frames_y:
        print(f"\n   ⚙️ Traitement Frame Y={gf_y} (Pureté: {purete_par_tranche[gf_y]*100:.1f}%)")
        
        erreur_sur_golden = erreur[:, gf_y]
        surface_golden = surface_lisse[:, gf_y]
        masque_signal = ~colonnes_vides[:, gf_y]
        
        seuil_pixel_actuel = 0.5 
        pixels_valides_x = []
        
        while seuil_pixel_actuel <= EPAISSEUR_PEAU_EN_PIXEL:
            pixels_valides_x = np.where((erreur_sur_golden <= seuil_pixel_actuel) & masque_signal)[0]
            if len(pixels_valides_x) >= NB_POINTS_PROMPT:
                break
            seuil_pixel_actuel += 0.5
            
        if len(pixels_valides_x) < NB_POINTS_PROMPT:
            print(f"      ⚠️ Quota non atteint sous la limite bio ({EPAISSEUR_PEAU_EN_PIXEL:.1f}px). Utilisation des meilleurs trouvés.")
            if len(pixels_valides_x) == 0: pixels_valides_x = np.where(masque_signal)[0] 

        x_min_valide, x_max_valide = pixels_valides_x.min(), pixels_valides_x.max()
        grille_ideale_x = np.linspace(x_min_valide, x_max_valide, NB_POINTS_PROMPT)
        
        points_finaux_x = sorted(list(set(pixels_valides_x[(np.abs(pixels_valides_x[:, None] - grille_ideale_x)).argmin(axis=0)])))

        prompts_points = []
        prompts_labels = []

        for x_brut in points_finaux_x:
            z_brut = surface_golden[x_brut]
            projeter = lambda val, pad: int((val + pad) * ratio_scale)
            x_1024 = projeter(x_brut, pad_left)
            
            prompts_points.extend([[x_1024, projeter(z_brut, pad_top)], 
                                   [x_1024, projeter(max(0, z_brut - offset_haut_pixels), pad_top)], 
                                   [x_1024, projeter(min(nz - 1, z_brut + offset_bas_pixels), pad_top)]])
            prompts_labels.extend([1, 0, 0])

        prompts_finaux.append({"frame_index": int(gf_y), "points": prompts_points, "labels": prompts_labels})

    with open(FICHIER_PROMPTS_SORTIE, 'w') as f:
        json.dump({"nb_golden_frames": len(prompts_finaux), "prompts": prompts_finaux}, f, indent=4)
        
    print(f"\n✅ Prompts intelligents générés avec succès dans '{FICHIER_PROMPTS_SORTIE}' !")

if __name__ == "__main__":
    generer_prompts()