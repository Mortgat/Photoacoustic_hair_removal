import numpy as np
import os
import gc
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import frangi
from skimage.measure import label, regionprops

# ==========================================
# CONFIGURATION V6.8 (DÉFINITIVE)
# ==========================================
FICHIER_NORMAL = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_DENOISED = r"dicom_data/dicom/4261_fromdcm_denoised_doux.npz"
DOSSIER_SORTIE = r"pipeline/2steps_pu_learning"

DX_MM, DY_MM, DZ_MM = 0.125, 0.125, 0.125
POURCENTAGE_PEAU_UTILE = 98.0 
GAUSSIAN_SMOOTH_FRANGI = 0.7 

# --- PARAMÈTRES POILS ---
PERCENTILE_POILS_INT_GRAINE = 95.0    
PERCENTILE_POILS_INT_TEST = 75.0      
PERCENTILE_FRANGI_POILS = 75.0        
LONGUEUR_MIN_POIL_MM = 0.8          
LIMITE_LONGUEUR_POIL_MM = 10.0      
OFFSET_PEAU_STANDARD_VOXELS = 0     
PROFONDEUR_FORET_MAX_MM = 1.5       

# --- PARAMÈTRES VAISSEAUX ---
PERCENTILE_FRANGI_VAISSEAUX = 93.0  
LONGUEUR_MIN_VAISSEAU_MM = 6.0
RATIO_HORIZONTALITE_MAX = 0.15  
PROFONDEUR_MIN_VAISSEAU_MM = 0.5
RATIO_PROFONDEUR_RELATIVE = 0.5
QUOTA_VAISSEAUX_PROFONDS = 0.20

# --- PARAMÈTRES BRUIT DE FOND ---
PERCENTILE_HN_MIN = 50.0
PERCENTILE_HN_MAX = 80.0
HAUTEUR_AIR_MAX_MM = 5.0        
MARGE_SECURITE_AIR_VOXELS = 3

# --- RECONSTRUCTION 3D ---
Z_EXPANSION_MAX_VOXELS = 10         
Z_DIFF_INTENSITE_MAX = 0.30  

def calculer_peau_propre(nx, ny, nz):
    data = np.load(FICHIER_DENOISED)
    vol = data["volume"]
    seuil = vol.mean() + 2 * vol.std()
    surf = np.argmax(vol > seuil, axis=2)
    surf[np.max(vol, axis=2) < seuil] = nz 
    
    surf_lisse = median_filter(surf, size=30).astype(np.float32)
    surf_lisse[surf_lisse >= nz-1] = np.nan
    
    fraction_a_couper = (100.0 - POURCENTAGE_PEAU_UTILE) / 2.0 / 100.0
    y_valides_global = np.where(np.any(~np.isnan(surf_lisse), axis=0))[0]
    if len(y_valides_global) > 0:
        y_min, y_max = y_valides_global[0], y_valides_global[-1]
        p_y = int((y_max - y_min) * fraction_a_couper)
        if p_y > 0:
            surf_lisse[:, y_min : y_min + p_y] = np.nan
            surf_lisse[:, y_max - p_y + 1 : y_max + 1] = np.nan

    for y in range(ny):
        col_y = surf_lisse[:, y]
        valides = np.where(~np.isnan(col_y))[0]
        if len(valides) > 0:
            x_min, x_max = valides[0], valides[-1]
            p_x = int((x_max - x_min) * fraction_a_couper)
            if p_x > 0:
                surf_lisse[x_min : x_min + p_x, y] = np.nan
                surf_lisse[x_max - p_x + 1 : x_max + 1, y] = np.nan
                
    return surf_lisse

def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    print("📂 Chargement du volume brut...")
    vol_brut = np.load(FICHIER_NORMAL)["volume"].astype(np.float32)
    nx, ny, nz = vol_brut.shape

    print("🔍 [1/5] Extraction de la surface de la peau...")
    surface_peau = calculer_peau_propre(nx, ny, nz)
    np.save(os.path.join(DOSSIER_SORTIE, "surface_peau.npy"), surface_peau)
    
    print("⚙️ [2/5] Calcul du MIP 2D et filtrage de Frangi...")
    mip_global = np.max(vol_brut, axis=2)
    z_indices_max = np.argmax(vol_brut, axis=2)
    
    lo, hi = np.percentile(mip_global, 1), np.percentile(mip_global, 99)
    mip_norm = np.clip((mip_global - lo) / (hi - lo), 0, 1)

    f2d = frangi(mip_norm, sigmas=[0.5, 1.0, 1.5], black_ridges=False)
    f2d = gaussian_filter(f2d, sigma=GAUSSIAN_SMOOTH_FRANGI)
    f2d = (f2d - f2d.min()) / (f2d.max() - f2d.min() + 1e-8)

    # =======================================================
    print("🟢 [3/5] Extraction des Poils Sûrs (Standards & Forêt)...")
    seuil_p_graine = np.percentile(mip_norm, PERCENTILE_POILS_INT_GRAINE)
    seuil_p_test = np.percentile(mip_norm, PERCENTILE_POILS_INT_TEST)
    seuil_f_poils = np.percentile(f2d, PERCENTILE_FRANGI_POILS)

    graines_poils = (mip_norm > seuil_p_graine) & (f2d > seuil_f_poils)
    masque_test_poils = (mip_norm > seuil_p_test) & (f2d > seuil_f_poils)
    
    poils_2d = np.zeros_like(mip_norm, dtype=bool)
    labels_test_poils = label(masque_test_poils)
    
    for reg in regionprops(labels_test_poils):
        xs, ys = reg.coords[:, 0], reg.coords[:, 1]
        
        # Logique stricte de l'ancien script V6
        if np.any(graines_poils[xs, ys]):
            longueur_test_mm = max(xs.max() - xs.min(), ys.max() - ys.min()) * DX_MM
            
            masque_graines_locales = graines_poils[xs, ys]
            g_xs = xs[masque_graines_locales]
            g_ys = ys[masque_graines_locales]
            longueur_graine_mm = max(g_xs.max() - g_xs.min(), g_ys.max() - g_ys.min()) * DX_MM
            
            if longueur_graine_mm >= LONGUEUR_MIN_POIL_MM:
                z_skin = surface_peau[g_xs, g_ys]
                valid = ~np.isnan(z_skin)
                
                if np.any(valid):
                    skin_med = np.median(z_skin[valid])
                    z_med = np.median(z_indices_max[g_xs, g_ys])
                    
                    if z_med <= skin_med + OFFSET_PEAU_STANDARD_VOXELS:
                        poils_2d[g_xs, g_ys] = True
                    elif skin_med < z_med <= skin_med + (PROFONDEUR_FORET_MAX_MM / DZ_MM):
                        if longueur_test_mm <= LIMITE_LONGUEUR_POIL_MM:
                            poils_2d[g_xs, g_ys] = True

    # =======================================================
    print("🔴 [4/5] Extraction des Hard Negatives (Vaisseaux & Speckle)...")
    
    # --- CLIFF DETECTOR ---
    grad_x = np.gradient(z_indices_max.astype(float), axis=0)
    grad_y = np.gradient(z_indices_max.astype(float), axis=1)
    mag_grad = np.sqrt(grad_x**2 + grad_y**2)
    m_falaises = mag_grad > (0.25 / DZ_MM)
    
    seuil_f_vs = np.percentile(f2d, PERCENTILE_FRANGI_VAISSEAUX)
    cand_vs = (f2d > seuil_f_vs) & (~poils_2d) & (~m_falaises)
    labels_vs = label(cand_vs)
    
    vaisseaux_2d = np.zeros_like(mip_norm, dtype=bool)
    v_med_coords, v_deep_coords, v_med_vol = [], [], 0
    p_min_vox = PROFONDEUR_MIN_VAISSEAU_MM / DZ_MM

    for reg in regionprops(labels_vs):
        if reg.major_axis_length >= (LONGUEUR_MIN_VAISSEAU_MM / DX_MM):
            xs, ys = reg.coords[:, 0], reg.coords[:, 1]
            z_s = surface_peau[xs, ys]
            mask_nan = ~np.isnan(z_s)
            
            if np.any(mask_nan):
                sk_m = np.median(z_s[mask_nan])
                z_m = np.median(z_indices_max[xs, ys])
                
                # Vaisseaux Profonds
                if z_m >= sk_m + RATIO_PROFONDEUR_RELATIVE * (nz - sk_m):
                    v_deep_coords.append((xs, ys))
                # Vaisseaux Médians avec dz_relatif
                elif z_m >= sk_m + p_min_vox:
                    z_rel = z_indices_max[xs, ys] - z_s
                    z_rel_clean = z_rel[~np.isnan(z_rel)]
                    if len(z_rel_clean) > 0:
                        if (np.ptp(z_rel_clean) / max(1, xs.max()-xs.min(), ys.max()-ys.min())) <= RATIO_HORIZONTALITE_MAX:
                            v_med_coords.append((xs, ys))
                            v_med_vol += len(xs)

    # Quota aléatoire pour les vaisseaux profonds
    np.random.shuffle(v_deep_coords)
    curr_v_deep = 0
    for xs, ys in v_med_coords: 
        vaisseaux_2d[xs, ys] = True
    for xs, ys in v_deep_coords:
        if curr_v_deep <= v_med_vol * QUOTA_VAISSEAUX_PROFONDS:
            vaisseaux_2d[xs, ys] = True
            curr_v_deep += len(xs)

    # Bruit Speckle Amorphe
    seuil_hn_min = np.percentile(mip_norm, PERCENTILE_HN_MIN)
    seuil_hn_max = np.percentile(mip_norm, PERCENTILE_HN_MAX)
    hn_2d = (mip_norm > seuil_hn_min) & (mip_norm < seuil_hn_max) & (f2d < 0.05) & (~poils_2d)
    m_hn_3d = np.zeros_like(vol_brut, dtype=bool)
    xs_hn, ys_hn = np.where(hn_2d)
    
    for x, y in zip(xs_hn, ys_hn):
        if not np.isnan(surface_peau[x, y]):
            za0 = int(surface_peau[x,y] - (HAUTEUR_AIR_MAX_MM/DZ_MM))
            za1 = int(surface_peau[x,y] - MARGE_SECURITE_AIR_VOXELS)
            if za1 > za0:
                m_hn_3d[x, y, np.random.randint(max(0,za0), max(1,za1))] = True

    # =======================================================
    print("🧊 [5/5] Z-Expansion et Sauvegarde (Binaire Strict)...")
    
    def expand_3d(mask_2d, is_vaisseau=False):
        mask_3d = np.zeros_like(vol_brut, dtype=bool)
        xs, ys = np.where(mask_2d)
        
        for x, y in zip(xs, ys):
            zc = z_indices_max[x, y]
            
            limite_haute_z = surface_peau[x,y] + p_min_vox if is_vaisseau else 0
            if is_vaisseau and (np.isnan(surface_peau[x,y]) or zc < limite_haute_z):
                continue 
                    
            val_base = vol_brut[x, y, zc]
            mask_3d[x, y, zc] = True
            
            for direction in [-1, 1]:
                for step in range(1, Z_EXPANSION_MAX_VOXELS + 1):
                    cz = zc + (direction * step)
                    
                    if 0 <= cz < nz and not (is_vaisseau and cz < limite_haute_z):
                        val_curr = vol_brut[x, y, cz]
                        if abs(val_base - val_curr) / (abs(val_base) + 1e-8) <= Z_DIFF_INTENSITE_MAX:
                            mask_3d[x, y, cz] = True
                        else: 
                            break
        return mask_3d

    masque_poils_3d = expand_3d(poils_2d, is_vaisseau=False)
    masque_vaisseaux_3d = expand_3d(vaisseaux_2d, is_vaisseau=True)

    np.save(os.path.join(DOSSIER_SORTIE, "masque_poils_surs.npy"), masque_poils_3d)
    np.save(os.path.join(DOSSIER_SORTIE, "masque_hard_negatives.npy"), m_hn_3d)
    np.save(os.path.join(DOSSIER_SORTIE, "masque_vaisseaux_surs.npy"), masque_vaisseaux_3d)
    
    carte_f3d = np.zeros_like(vol_brut, dtype=np.float16)
    idx_f = np.where(f2d > 0.05)
    carte_f3d[idx_f[0], idx_f[1], z_indices_max[idx_f]] = f2d[idx_f].astype(np.float16)
    np.save(os.path.join(DOSSIER_SORTIE, "carte_frangi.npy"), carte_f3d)
    
    print(f"✅ Terminé. Poils : {np.sum(masque_poils_3d)} voxels | HN : {np.sum(m_hn_3d)} voxels | Vaisseaux : {np.sum(masque_vaisseaux_3d)} voxels.")

if __name__ == "__main__":
    main()