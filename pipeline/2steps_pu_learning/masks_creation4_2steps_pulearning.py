import numpy as np
import os
import gc
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import frangi
from skimage.measure import label, regionprops

# ==========================================
# CONFIGURATION V2.9
# ==========================================
FICHIER_NORMAL = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_DENOISED = r"dicom_data/dicom/4261_fromdcm_denoised_doux.npz"
DOSSIER_SORTIE = r"pipeline/2steps_pu_learning"

DX_MM, DY_MM, DZ_MM = 0.125, 0.125, 0.125
LONGUEUR_MIN_POIL_MM = 0.8  
OFFSET_PEAU_MM = 0.1       
GAUSSIAN_SMOOTH_FRANGI = 0.7 

Z_EXPANSION_MAX_VOXELS = 3    
Z_DIFF_INTENSITE_MAX = 0.30   
POURCENTAGE_PEAU_UTILE = 98.0 

PERCENTILE_POILS_INTENSITE = 95.0 
PERCENTILE_FRANGI_SURS = 75.0
PERCENTILE_HN_INT_MAX = 80    
SEUIL_FRANGI_HN_MAX = 0.05    

# Limites Spatiales
HAUTEUR_AIR_MAX_MM = 5.0        # Plafond maximum pour un poil sûr ET pour les HN air
MARGE_SECURITE_AIR_VOXELS = 3
PROFONDEUR_SOUS_PEAU_HN_MM = 1.5 # Limite stricte sous la peau

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
        pixels_y_a_couper = int((y_max - y_min) * fraction_a_couper)
        if pixels_y_a_couper > 0:
            surf_lisse[:, y_min : y_min + pixels_y_a_couper] = np.nan
            surf_lisse[:, y_max - pixels_y_a_couper + 1 : y_max + 1] = np.nan

    for y in range(ny):
        col_y = surf_lisse[:, y]
        valides = np.where(~np.isnan(col_y))[0]
        if len(valides) > 0:
            x_min, x_max = valides[0], valides[-1]
            pixels_x_a_couper = int((x_max - x_min) * fraction_a_couper)
            if pixels_x_a_couper > 0:
                surf_lisse[x_min : x_min + pixels_x_a_couper, y] = np.nan
                surf_lisse[x_max - pixels_x_a_couper + 1 : x_max + 1, y] = np.nan
                
    return surf_lisse

def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    print("📂 Chargement du volume brut...")
    vol_brut = np.load(FICHIER_NORMAL)["volume"].astype(np.float32)
    nx, ny, nz = vol_brut.shape

    print("🔍 [1/5] Extraction et rognage de la surface de la peau...")
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

    print("🟢 [3/5] Extraction des Poils Sûrs (Couloir spatial strict)...")
    seuil_p = np.percentile(mip_norm, PERCENTILE_POILS_INTENSITE)
    seuil_f = np.percentile(f2d, PERCENTILE_FRANGI_SURS)
    
    candidats_2d = (mip_norm > seuil_p) & (f2d > seuil_f)
    labels = label(candidats_2d)
    poils_surs_2d = np.zeros_like(candidats_2d, dtype=bool)
    
    offset_vox = OFFSET_PEAU_MM / DZ_MM
    plafond_air_vox = HAUTEUR_AIR_MAX_MM / DZ_MM
    
    for reg in regionprops(labels):
        if reg.major_axis_length >= (LONGUEUR_MIN_POIL_MM / DX_MM):
            xs, ys = reg.coords[:, 0], reg.coords[:, 1]
            
            # Extraction des valeurs de peau valides pour éviter les erreurs liées aux NaNs
            z_skin_valid = surface_peau[xs, ys]
            z_skin_valid = z_skin_valid[~np.isnan(z_skin_valid)]
            
            if len(z_skin_valid) > 0:
                skin_median = np.median(z_skin_valid)
                z_median = np.median(z_indices_max[xs, ys])
                
                # Condition stricte : Entre (Peau - 5mm) et (Peau + Offset)
                if (skin_median - plafond_air_vox) <= z_median <= (skin_median + offset_vox):
                    poils_surs_2d[xs, ys] = True

    masque_poils_3d = np.zeros_like(vol_brut, dtype=bool)
    xs, ys = np.where(poils_surs_2d)
    for x, y in zip(xs, ys):
        z_max = z_indices_max[x, y]
        val_max = vol_brut[x, y, z_max]
        masque_poils_3d[x, y, z_max] = True
        for direction in [-1, 1]:
            for step in range(1, Z_EXPANSION_MAX_VOXELS + 1):
                curr_z = z_max + (direction * step)
                if 0 <= curr_z < nz:
                    val_curr = vol_brut[x, y, curr_z]
                    if abs(val_max - val_curr) / (abs(val_max) + 1e-8) <= Z_DIFF_INTENSITE_MAX:
                        masque_poils_3d[x, y, curr_z] = True
                    else: break

    print("🔴 [4/5] Extraction des Hard Negatives (Air & Sous-cutané)...")
    seuil_hn_min = np.percentile(mip_norm, 50)
    seuil_hn_max = np.percentile(mip_norm, PERCENTILE_HN_INT_MAX)
    
    z_min_air = (surface_peau - plafond_air_vox).clip(0, nz)
    z_max_air = (surface_peau - MARGE_SECURITE_AIR_VOXELS).clip(0, nz) 
    
    z_min_sous = surface_peau.clip(0, nz)
    z_max_sous = (surface_peau + (PROFONDEUR_SOUS_PEAU_HN_MM / DZ_MM)).clip(0, nz)
    
    hn_2d_mask = (mip_norm > seuil_hn_min) & (mip_norm < seuil_hn_max) & (f2d < SEUIL_FRANGI_HN_MAX) & (~poils_surs_2d)
    
    masque_hn_3d = np.zeros_like(vol_brut, dtype=bool)
    xs_hn, ys_hn = np.where(hn_2d_mask)
    for x, y in zip(xs_hn, ys_hn):
        if not np.isnan(surface_peau[x, y]):
            za_start, za_end = int(z_min_air[x, y]), int(z_max_air[x, y])
            if za_end > za_start:
                idx_za = np.random.randint(za_start, za_end, size=1)
                masque_hn_3d[x, y, idx_za] = True
                
            zs_start, zs_end = int(z_min_sous[x, y]), int(z_max_sous[x, y])
            if zs_end > zs_start:
                idx_zs = np.random.randint(zs_start, zs_end, size=1)
                masque_hn_3d[x, y, idx_zs] = True

    print("💾 [5/5] Sauvegarde des tenseurs sur le disque...")
    np.save(os.path.join(DOSSIER_SORTIE, "masque_poils_surs.npy"), masque_poils_3d)
    np.save(os.path.join(DOSSIER_SORTIE, "masque_hard_negatives.npy"), masque_hn_3d)
    
    carte_f3d = np.zeros_like(vol_brut, dtype=np.float16)
    idx_f = np.where(f2d > 0.05)
    carte_f3d[idx_f[0], idx_f[1], z_indices_max[idx_f]] = f2d[idx_f].astype(np.float16)
    np.save(os.path.join(DOSSIER_SORTIE, "carte_frangi.npy"), carte_f3d)
    
    print(f"✅ Terminé. Poils validés : {np.sum(masque_poils_3d)} voxels | Hard Negatives : {np.sum(masque_hn_3d)} voxels.")

if __name__ == "__main__":
    main()