import numpy as np
import os
import time

try:
    import cupy as cp
    from cupyx.scipy.ndimage import label
    GPU_ACTIF = True
except ImportError:
    import numpy as cp
    from scipy.ndimage import label
    GPU_ACTIF = False

from scipy.ndimage import find_objects


def to_cpu(arr):
    if GPU_ACTIF:
        return cp.asnumpy(arr)
    return arr


def synchroniser_gpu():
    if GPU_ACTIF:
        cp.cuda.Stream.null.synchronize()


def generer_masque_poils_surs():
    # ==========================================
    # ⚙️ CONFIGURATION
    # ==========================================
    fichier_volume = r"dicom_data/dicom/4261_fromdcm.npz"
    fichier_peau = r"pipeline/body_hair_extraction_methods/surface_peau.npy"
    dossier_sortie = r"pipeline/body_hair_extraction_methods"

    DZ_MM = 0.125

    # Fenêtre STRICTEMENT au-dessus de la peau
    # On interdit totalement le sous-cutané
    DIST_MIN_AU_DESSUS_PEAU_MM = 0.125   # 1 voxel au-dessus
    DIST_MAX_AU_DESSUS_PEAU_MM = 1.5     # ~12 voxels au-dessus

    # Intensité
    PERCENTILE_POILS = 99.8

    # Filtrage morphologique
    MIN_VOXELS = 6
    MAX_VOXELS = 10000
    MIN_ANISOTROPIE = 3.0

    fichier_sortie = os.path.join(dossier_sortie, "masque_poils_surs_v3.npy")

    # ==========================================
    # CHARGEMENT
    # ==========================================
    if not os.path.exists(fichier_volume):
        print(f"❌ Volume introuvable : {fichier_volume}")
        return
    if not os.path.exists(fichier_peau):
        print(f"❌ Surface peau introuvable : {fichier_peau}")
        return

    os.makedirs(dossier_sortie, exist_ok=True)

    t0 = time.perf_counter()

    print("Chargement du volume...")
    data = np.load(fichier_volume)
    volume = cp.asarray(data["volume"])

    print("Chargement de la surface de peau...")
    surface_z = cp.asarray(np.load(fichier_peau).astype(np.float32))

    nx, ny, nz = volume.shape
    print(f"Volume shape : {volume.shape}")
    print(f"Surface shape: {surface_z.shape}")
    print(f"GPU actif    : {GPU_ACTIF}")

    if surface_z.shape != (nx, ny):
        print("❌ Incompatibilité entre surface_z et volume.")
        return

    # ==========================================
    # MASQUE DE VALIDITÉ DE LA PEAU
    # ==========================================
    masque_valide_2d = surface_z < nz

    # ==========================================
    # SEUIL D'INTENSITÉ
    # ==========================================
    t1 = time.perf_counter()
    print(f"Calcul du seuil d'intensité au percentile {PERCENTILE_POILS}...")
    seuil_poils = cp.percentile(volume, PERCENTILE_POILS)
    masque_intensite = volume > seuil_poils
    synchroniser_gpu()
    print(f"Temps seuil intensité : {time.perf_counter() - t1:.2f} s")

    # ==========================================
    # BANDE STRICTEMENT AU-DESSUS DE LA PEAU
    # ==========================================
    t2 = time.perf_counter()
    print("Construction de la bande strictement au-dessus de la peau...")

    min_au_dessus_vox = DIST_MIN_AU_DESSUS_PEAU_MM / DZ_MM
    max_au_dessus_vox = DIST_MAX_AU_DESSUS_PEAU_MM / DZ_MM

    z_grid = cp.arange(nz, dtype=cp.float32).reshape(1, 1, nz)
    surface_grid = surface_z[:, :, cp.newaxis]

    # Convention : z augmente vers le bas
    # "au-dessus de la peau" => z < surface_z
    borne_min = surface_grid - max_au_dessus_vox
    borne_max = surface_grid - min_au_dessus_vox

    masque_spatial = (z_grid >= borne_min) & (z_grid <= borne_max)
    masque_valide_3d = masque_valide_2d[:, :, cp.newaxis]

    masque_candidats = masque_intensite & masque_spatial & masque_valide_3d

    nb_candidats = int(to_cpu(cp.count_nonzero(masque_candidats)))
    synchroniser_gpu()

    print(f"Nb voxels candidats après intensité + bande air : {nb_candidats}")
    print(f"Temps bande peau : {time.perf_counter() - t2:.2f} s")

    # ==========================================
    # COMPOSANTES CONNEXES
    # ==========================================
    t3 = time.perf_counter()
    print("Calcul des composantes connexes...")
    labels, num_features = label(masque_candidats)
    synchroniser_gpu()

    num_features = int(num_features)
    print(f"Nombre de composantes détectées : {num_features}")
    print(f"Temps label : {time.perf_counter() - t3:.2f} s")

    if num_features == 0:
        np.save(fichier_sortie, np.zeros((nx, ny, nz), dtype=bool))
        print("Aucune composante. Masque vide sauvegardé.")
        return

    # ==========================================
    # FILTRE RAPIDE PAR TAILLE
    # ==========================================
    t4 = time.perf_counter()
    print("Filtrage rapide par taille...")
    tailles = cp.bincount(labels.ravel(), minlength=num_features + 1)

    labels_valides_taille = cp.where(
        (tailles >= MIN_VOXELS) & (tailles <= MAX_VOXELS)
    )[0]
    labels_valides_taille = labels_valides_taille[labels_valides_taille != 0]

    nb_taille = int(to_cpu(labels_valides_taille.size))
    synchroniser_gpu()

    print(f"Composantes gardées après filtre taille : {nb_taille}")
    print(f"Temps filtre taille : {time.perf_counter() - t4:.2f} s")

    if nb_taille == 0:
        np.save(fichier_sortie, np.zeros((nx, ny, nz), dtype=bool))
        print("Aucune composante après filtre taille. Masque vide sauvegardé.")
        return

    # ==========================================
    # TRANSFERT CPU UNIQUE + FIND_OBJECTS
    # ==========================================
    t5 = time.perf_counter()
    print("Transfert CPU du volume labellisé...")
    labels_cpu = to_cpu(labels)
    surface_cpu = to_cpu(surface_z)
    labels_valides_taille_cpu = set(to_cpu(labels_valides_taille).tolist())
    print(f"Temps transfert CPU : {time.perf_counter() - t5:.2f} s")

    t6 = time.perf_counter()
    print("Calcul des boîtes englobantes minimales...")
    objets = find_objects(labels_cpu, max_label=num_features)
    print(f"Temps find_objects : {time.perf_counter() - t6:.2f} s")

    # ==========================================
    # FILTRAGE MORPHOLOGIQUE FIN
    # ==========================================
    t7 = time.perf_counter()
    print("Filtrage morphologique fin...")
    masque_final_cpu = np.zeros((nx, ny, nz), dtype=bool)

    nb_testees = 0
    nb_gardees = 0
    nb_rejetees_peau = 0
    nb_rejetees_aniso = 0

    for lab in labels_valides_taille_cpu:
        slc = objets[lab - 1]
        if slc is None:
            continue

        nb_testees += 1

        sub_labels = labels_cpu[slc]
        sub_mask = (sub_labels == lab)

        coords = np.where(sub_mask)
        if coords[0].size == 0:
            continue

        # dimensions locales bbox
        dx = coords[0].max() - coords[0].min() + 1
        dy = coords[1].max() - coords[1].min() + 1
        dz = coords[2].max() - coords[2].min() + 1

        dims = np.array([dx, dy, dz], dtype=float)
        anisotropie = dims.max() / max(dims.min(), 1.0)

        if anisotropie < MIN_ANISOTROPIE:
            nb_rejetees_aniso += 1
            continue

        # Coordonnées globales de la composante
        x0, y0, z0 = slc[0].start, slc[1].start, slc[2].start
        xs = coords[0] + x0
        ys = coords[1] + y0
        zs = coords[2] + z0

        # Rejet strict si la composante touche ou passe sous la peau
        surface_local = surface_cpu[xs, ys]

        # On veut strictement au-dessus :
        # zs < surface_local - min_au_dessus_vox
        if np.any(zs >= (surface_local - min_au_dessus_vox)):
            nb_rejetees_peau += 1
            continue

        masque_final_cpu[slc] |= sub_mask
        nb_gardees += 1

    print(f"Composantes testées morphologiquement : {nb_testees}")
    print(f"Composantes rejetées (anisotropie)   : {nb_rejetees_aniso}")
    print(f"Composantes rejetées (peau)          : {nb_rejetees_peau}")
    print(f"Composantes gardées                  : {nb_gardees}")
    print(f"Temps filtrage morphologique : {time.perf_counter() - t7:.2f} s")

    # ==========================================
    # SAUVEGARDE
    # ==========================================
    t8 = time.perf_counter()
    np.save(fichier_sortie, masque_final_cpu)
    print(f"Temps sauvegarde : {time.perf_counter() - t8:.2f} s")

    print("\n✅ Terminé.")
    print(f"Masque sauvegardé : {fichier_sortie}")
    print(f"Nb voxels finaux : {np.count_nonzero(masque_final_cpu)}")
    print(f"Temps total : {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    generer_masque_poils_surs()