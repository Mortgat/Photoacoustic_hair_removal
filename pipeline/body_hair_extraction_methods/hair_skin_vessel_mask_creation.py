import numpy as np
import os
import time

try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter, label as gpu_label
    GPU_ACTIF = True
except ImportError:
    import numpy as cp
    from scipy.ndimage import median_filter, label as gpu_label
    GPU_ACTIF = False

from scipy.ndimage import label as cpu_label, find_objects


def to_cpu(arr):
    if GPU_ACTIF:
        return cp.asnumpy(arr)
    return arr


def synchroniser_gpu():
    if GPU_ACTIF:
        cp.cuda.Stream.null.synchronize()


def free_gpu():
    if GPU_ACTIF:
        synchroniser_gpu()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def filter_filament_components_cpu(
    labels_cpu,
    num_features,
    valid_labels_set,
    min_anisotropy,
):
    objets = find_objects(labels_cpu, max_label=num_features)
    out = np.zeros_like(labels_cpu, dtype=bool)

    tested = 0
    kept = 0

    for lab in valid_labels_set:
        slc = objets[lab - 1]
        if slc is None:
            continue

        tested += 1
        sub_labels = labels_cpu[slc]
        sub_mask = (sub_labels == lab)

        coords = np.where(sub_mask)
        if coords[0].size == 0:
            continue

        dx = coords[0].max() - coords[0].min() + 1
        dy = coords[1].max() - coords[1].min() + 1
        dz = coords[2].max() - coords[2].min() + 1

        dims = np.array([dx, dy, dz], dtype=float)
        anis = dims.max() / max(dims.min(), 1.0)

        if anis < min_anisotropy:
            continue

        out[slc] |= sub_mask
        kept += 1

    return out, tested, kept


def generer_masques():
    # =====================================================
    # CONFIG
    # =====================================================
    fichier_denoised = r"dicom_data/dicom/4261_fromdcm_denoised_doux.npz"
    fichier_normal = r"dicom_data/dicom/4261_fromdcm.npz"
    dossier_sortie = r"pipeline/body_hair_extraction_methods"

    # ---- Peau
    MULTIPLICATEUR_SEUIL_PEAU = 2
    PERCENTILE_COUPURE_HAUTE = 93
    TAILLE_FILTRE_SURFACE = 30
    POURCENTAGE_SURFACE_MIN = 0.05

    # ---- Résolution Z
    DZ_MM = 0.125

    # ---- Poils sûrs
    PERCENTILE_POILS = 99.7
    DIST_MIN_AIR_MM = 0.125
    DIST_MAX_AIR_MM = 5.0
    MIN_VOXELS_POILS = 6
    MAX_VOXELS_POILS = 250
    MIN_ANISOTROPIE_POILS = 3.0

    # ---- Vaisseaux potentiels
    PERCENTILE_VAISSEAUX = 99.5
    DIST_MIN_SOUS_PEAU_MM = 0.25
    DIST_MAX_SOUS_PEAU_MM = 8.0
    MIN_VOXELS_VAISSEAUX = 10
    MAX_VOXELS_VAISSEAUX = 10000
    MIN_ANISOTROPIE_VAISSEAUX = 2.5

    # ---- Sorties
    chemin_peau = os.path.join(dossier_sortie, "surface_peau.npy")
    chemin_poils = os.path.join(dossier_sortie, "masque_poils_surs_v3.npy")
    chemin_vaisseaux = os.path.join(dossier_sortie, "masque_vaisseaux_potentiels.npy")

    os.makedirs(dossier_sortie, exist_ok=True)

    if not os.path.exists(fichier_denoised):
        print(f"❌ Introuvable : {fichier_denoised}")
        return
    if not os.path.exists(fichier_normal):
        print(f"❌ Introuvable : {fichier_normal}")
        return

    t0 = time.perf_counter()

    # =====================================================
    # A) CALCUL DE LA SURFACE DE PEAU
    # =====================================================
    print("Chargement du volume débruité...")
    data_denoised = np.load(fichier_denoised)
    volume_denoised = cp.asarray(data_denoised["volume"])
    nx, ny, nz = volume_denoised.shape

    seuil_bas_peau = volume_denoised.mean() + MULTIPLICATEUR_SEUIL_PEAU * volume_denoised.std()
    pixels_utiles = volume_denoised[volume_denoised > seuil_bas_peau]
    seuil_haut = cp.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else cp.max(volume_denoised)

    masque_brillant = (volume_denoised > seuil_bas_peau) & (volume_denoised < seuil_haut)
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    max_colonnes = cp.max(volume_denoised, axis=2)
    colonnes_vides = max_colonnes < seuil_bas_peau
    masque_valide_2d = ~colonnes_vides

    print("Nettoyage des îlots volants de peau...")
    labels_surface, num_surface = gpu_label(masque_valide_2d)
    synchroniser_gpu()

    if int(num_surface) > 0:
        tailles_surface = cp.bincount(labels_surface.ravel())
        seuil_taille = POURCENTAGE_SURFACE_MIN * nx * ny
        labels_valides = cp.where(tailles_surface >= seuil_taille)[0]
        labels_valides = labels_valides[labels_valides != 0]
        masque_valide_2d = cp.isin(labels_surface, labels_valides)

    surface_brute_z[~masque_valide_2d] = nz
    surface_lisse_z = median_filter(surface_brute_z, size=TAILLE_FILTRE_SURFACE)

    surface_cpu = to_cpu(surface_lisse_z).astype(np.float32)
    masque_valide_2d_cpu = to_cpu(masque_valide_2d).astype(bool)

    np.save(chemin_peau, surface_cpu)
    print(f"✅ Surface peau sauvegardée : {chemin_peau}")

    del volume_denoised, data_denoised, masque_brillant, labels_surface, surface_brute_z
    free_gpu()

    # =====================================================
    # B) POILS SÛRS SUR GPU
    # =====================================================
    print("\n--- Construction masque poils sûrs (GPU si dispo) ---")
    t1 = time.perf_counter()

    data_normal = np.load(fichier_normal)
    volume_gpu = cp.asarray(data_normal["volume"])

    z_grid = cp.arange(nz, dtype=cp.float32).reshape(1, 1, nz)
    surface_grid = cp.asarray(surface_cpu)[:, :, None]
    masque_valide_3d = cp.asarray(masque_valide_2d_cpu)[:, :, None]

    seuil_poils = cp.percentile(volume_gpu, PERCENTILE_POILS)

    min_air_vox = DIST_MIN_AIR_MM / DZ_MM
    max_air_vox = DIST_MAX_AIR_MM / DZ_MM

    borne_air_min = surface_grid - max_air_vox
    borne_air_max = surface_grid - min_air_vox

    masque_air = (z_grid >= borne_air_min) & (z_grid <= borne_air_max)
    candidats_poils = (volume_gpu > seuil_poils) & masque_air & masque_valide_3d

    labels_poils, n_poils = gpu_label(candidats_poils)
    synchroniser_gpu()
    n_poils = int(n_poils)

    if n_poils > 0:
        tailles_poils = cp.bincount(labels_poils.ravel(), minlength=n_poils + 1)
        labels_poils_valides = cp.where(
            (tailles_poils >= MIN_VOXELS_POILS) & (tailles_poils <= MAX_VOXELS_POILS)
        )[0]
        labels_poils_valides = labels_poils_valides[labels_poils_valides != 0]

        labels_poils_cpu = to_cpu(labels_poils)
        labels_poils_valides_cpu = set(to_cpu(labels_poils_valides).tolist())

        masque_poils_cpu, tested_h, kept_h = filter_filament_components_cpu(
            labels_cpu=labels_poils_cpu,
            num_features=n_poils,
            valid_labels_set=labels_poils_valides_cpu,
            min_anisotropy=MIN_ANISOTROPIE_POILS,
        )
    else:
        masque_poils_cpu = np.zeros((nx, ny, nz), dtype=bool)
        tested_h, kept_h = 0, 0

    np.save(chemin_poils, masque_poils_cpu)
    print(f"✅ Masque poils sûrs sauvegardé : {chemin_poils}")
    print(f"Composantes poils testées/gardées : {tested_h}/{kept_h}")
    print(f"Nb voxels poils : {np.count_nonzero(masque_poils_cpu)}")
    print(f"Temps poils : {time.perf_counter() - t1:.2f} s")

    del volume_gpu, z_grid, surface_grid, masque_valide_3d, masque_air, candidats_poils, labels_poils
    free_gpu()

    # =====================================================
    # C) VAISSEAUX POTENTIELS SUR CPU
    # =====================================================
    print("\n--- Construction masque vaisseaux potentiels (CPU) ---")
    t2 = time.perf_counter()

    volume_cpu = data_normal["volume"].astype(np.float32)
    z_grid_cpu = np.arange(nz, dtype=np.float32).reshape(1, 1, nz)
    surface_grid_cpu = surface_cpu[:, :, None]
    masque_valide_3d_cpu = masque_valide_2d_cpu[:, :, None]

    seuil_vaisseaux = np.percentile(volume_cpu, PERCENTILE_VAISSEAUX)

    min_sous_vox = DIST_MIN_SOUS_PEAU_MM / DZ_MM
    max_sous_vox = DIST_MAX_SOUS_PEAU_MM / DZ_MM

    borne_v_min = surface_grid_cpu + min_sous_vox
    borne_v_max = surface_grid_cpu + max_sous_vox

    masque_sous_peau = (z_grid_cpu >= borne_v_min) & (z_grid_cpu <= borne_v_max)
    candidats_vaisseaux = (volume_cpu > seuil_vaisseaux) & masque_sous_peau & masque_valide_3d_cpu

    labels_v, n_v = cpu_label(candidats_vaisseaux)
    n_v = int(n_v)

    if n_v > 0:
        tailles_v = np.bincount(labels_v.ravel(), minlength=n_v + 1)
        labels_v_valides = np.where(
            (tailles_v >= MIN_VOXELS_VAISSEAUX) & (tailles_v <= MAX_VOXELS_VAISSEAUX)
        )[0]
        labels_v_valides = labels_v_valides[labels_v_valides != 0]
        labels_v_valides_set = set(labels_v_valides.tolist())

        masque_v_cpu, tested_v, kept_v = filter_filament_components_cpu(
            labels_cpu=labels_v,
            num_features=n_v,
            valid_labels_set=labels_v_valides_set,
            min_anisotropy=MIN_ANISOTROPIE_VAISSEAUX,
        )
    else:
        masque_v_cpu = np.zeros((nx, ny, nz), dtype=bool)
        tested_v, kept_v = 0, 0

    np.save(chemin_vaisseaux, masque_v_cpu)
    print(f"✅ Masque vaisseaux potentiels sauvegardé : {chemin_vaisseaux}")
    print(f"Composantes vaisseaux testées/gardées : {tested_v}/{kept_v}")
    print(f"Nb voxels vaisseaux : {np.count_nonzero(masque_v_cpu)}")
    print(f"Temps vaisseaux : {time.perf_counter() - t2:.2f} s")

    print(f"\n✅ Terminé en {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    generer_masques()