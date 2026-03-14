import numpy as np
import os

try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter, label
    GPU_ACTIF = True
except ImportError:
    import numpy as cp
    from scipy.ndimage import median_filter, label
    GPU_ACTIF = False

def generer_masque_pu():
    # ==========================================
    # ⚙️ CONFIGURATION DES PARAMÈTRES
    # ==========================================
    
    fichier_denoised = r"dicom_data/dicom/4261_fromdcm_denoised_doux.npz"
    fichier_normal = r"dicom_data/dicom/4261_fromdcm.npz"
    dossier_sortie = r"pipeline/body_hair_extraction_methods"
    
    # Paramètres Peau (Volume Débruité)
    MULTIPLICATEUR_SEUIL_PEAU = 2
    PERCENTILE_COUPURE_HAUTE = 93
    TAILLE_FILTRE_SURFACE = 30
    POURCENTAGE_SURFACE_MIN = 0.05  
    
    # Paramètres Poils (Volume Normal)
    PERCENTILE_POILS = 99.0         
    ECART_SECURITE_AIR_MM = 0.5     
    DZ_MM = 0.12500                 
    
    # ==========================================
    # ÉTAPE A : CALCUL DE LA SURFACE
    # ==========================================
    if not os.path.exists(fichier_denoised) or not os.path.exists(fichier_normal):
        print("Erreur : Fichiers .npz introuvables.")
        return

    os.makedirs(dossier_sortie, exist_ok=True)

    print("Chargement du volume débruité...")
    data_denoised = np.load(fichier_denoised)
    volume_denoised = cp.asarray(data_denoised['volume'])
    nx, ny, nz = volume_denoised.shape

    seuil_bas_peau = volume_denoised.mean() + MULTIPLICATEUR_SEUIL_PEAU * volume_denoised.std()
    pixels_utiles = volume_denoised[volume_denoised > seuil_bas_peau]
    seuil_haut = cp.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else cp.max(volume_denoised)

    masque_brillant = (volume_denoised > seuil_bas_peau) & (volume_denoised < seuil_haut)
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    max_colonnes = cp.max(volume_denoised, axis=2)
    colonnes_vides = max_colonnes < seuil_bas_peau
    masque_valide_2d = ~colonnes_vides

    print("Nettoyage des îlots volants...")
    labels, num_features = label(masque_valide_2d)
    if num_features > 0:
        if GPU_ACTIF:
            tailles = cp.bincount(labels.ravel())
            seuil_taille = POURCENTAGE_SURFACE_MIN * nx * ny
            labels_valides = cp.where(tailles >= seuil_taille)[0]
            labels_valides = labels_valides[labels_valides != 0]
            masque_valide_2d = cp.isin(labels, labels_valides)
        else:
            tailles = np.bincount(labels.ravel())
            seuil_taille = POURCENTAGE_SURFACE_MIN * nx * ny
            labels_valides = np.where(tailles >= seuil_taille)[0]
            labels_valides = labels_valides[labels_valides != 0]
            masque_valide_2d = np.isin(labels, labels_valides)

    surface_brute_z[~masque_valide_2d] = nz 
    surface_lisse_z = median_filter(surface_brute_z, size=TAILLE_FILTRE_SURFACE)

    # ==========================================
    # ÉTAPE B : NETTOYAGE VRAM
    # ==========================================
    del volume_denoised
    del data_denoised
    del masque_brillant
    if GPU_ACTIF:
        cp.get_default_memory_pool().free_all_blocks()

    # ==========================================
    # ÉTAPE C : EXTRACTION DES POILS PAR PERCENTILE
    # ==========================================
    print("Chargement du volume normal...")
    data_normal = np.load(fichier_normal)
    volume_normal = cp.asarray(data_normal['volume'])

    print(f"Calcul du seuil au percentile {PERCENTILE_POILS}...")
    seuil_poils = cp.percentile(volume_normal, PERCENTILE_POILS)
    masque_intensite = volume_normal > seuil_poils

    Z_grid = cp.arange(nz, dtype=cp.float32).reshape(1, 1, nz)
    Z_grid = cp.broadcast_to(Z_grid, (nx, ny, nz))
    Surface_grid = surface_lisse_z[:, :, cp.newaxis]

    ecart_voxels_z = ECART_SECURITE_AIR_MM / DZ_MM
    masque_spatial = Z_grid < (Surface_grid - ecart_voxels_z)
    masque_valide_3d = masque_valide_2d[:, :, cp.newaxis]

    masque_positifs = masque_intensite & masque_spatial & masque_valide_3d

    # ==========================================
    # ÉTAPE D : SAUVEGARDE (.npy séparés)
    # ==========================================
    if GPU_ACTIF:
        masque_positifs_cpu = masque_positifs.get()
        surface_lisse_cpu = surface_lisse_z.get()
    else:
        masque_positifs_cpu = masque_positifs
        surface_lisse_cpu = surface_lisse_z

    chemin_poils = os.path.join(dossier_sortie, "masque_positifs_pu.npy")
    chemin_peau = os.path.join(dossier_sortie, "surface_peau.npy")
    
    # Sauvegarde en deux fichiers npy distincts
    np.save(chemin_poils, masque_positifs_cpu)
    np.save(chemin_peau, surface_lisse_cpu)
    
    print(f"Fichiers générés : \n- {chemin_poils}\n- {chemin_peau}")

if __name__ == "__main__":
    generer_masque_pu()