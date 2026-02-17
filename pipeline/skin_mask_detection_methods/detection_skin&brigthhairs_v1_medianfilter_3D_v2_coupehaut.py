import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# ==========================================
# IMPORT GPU (CUPY)
# ==========================================
try:
    import cupy as cp
    GPU_ACTIF = True
    print("🚀 CuPy détecté : Détection accélérée sur GPU.")
except ImportError:
    print("⚠️ CuPy non trouvé. Bascule sur CPU.")
    import numpy as cp
    GPU_ACTIF = False

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# Paramètres ajustables
MULTIPLICATEUR_SEUIL = 2  
FORCE_LISSAGE_GAUSSIEN = 30  

# --- NOUVEAU PARAMÈTRE ---
# Ex: 98.0 ignore les 2% des pixels les plus lumineux (poils denses/artéfacts)
PERCENTILE_COUPURE_HAUTE = 95.0 

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def extraction_peau_topographique_bande_passante():    
    print("--- 🕵️ EXTRACTION DE LA PEAU (TOPOGRAPHIQUE + PASSE-BANDE) ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier introuvable : {fichier_mat}")
        return None, None, None

    # 1. Chargement et transfert GPU
    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume_cpu = mat[key]
    nx, ny, nz = volume_cpu.shape
    volume = cp.asarray(volume_cpu)

    # 2. Seuils (Bas et Haut)
    moyenne = volume.mean()
    std = volume.std()
    seuil_bas = moyenne + MULTIPLICATEUR_SEUIL * std
    
    pixels_utiles = volume[volume > seuil_bas]
    if pixels_utiles.size > 0:
        seuil_haut = cp.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE)
    else:
        seuil_haut = cp.max(volume)

    print(f"📊 Statistiques :")
    print(f"   - Seuil Bas (Peau) : {seuil_bas:.2f}")
    print(f"   - Seuil Haut (Top {100-PERCENTILE_COUPURE_HAUTE}% ignorés) : {seuil_haut:.2f}")

    # 3. Extraction de la surface brute (Filtre Passe-Bande)
    print("🔍 Scan des colonnes (Ignorance de la canopée de poils)...")
    masque_brillant = (volume > seuil_bas) & (volume < seuil_haut)
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    # Identification des colonnes valides/invalides
    # VRAIE AMÉLIORATION : Une colonne n'est valide que si elle a au moins un pixel dans la "bonne" tranche
    masque_valide = cp.any(masque_brillant, axis=2)

    # Rapatriement sur CPU pour l'interpolation
    print("⬅️ Rapatriement sur CPU pour lissage mathématique...")
    if GPU_ACTIF:
        surface_brute_cpu = surface_brute_z.get().astype(float)
        masque_valide_cpu = masque_valide.get()
    else:
        surface_brute_cpu = surface_brute_z.astype(float)
        masque_valide_cpu = masque_valide

    # 4. INTERPOLATION (Le rebouchage des trous géants et des poils ignorés)
    print("🌉 Construction des ponts (Interpolation cubique)...")
    
    coords_valides = np.array(np.nonzero(masque_valide_cpu)).T
    valeurs_valides = surface_brute_cpu[masque_valide_cpu]
    
    grille_x, grille_y = np.mgrid[0:nx, 0:ny]
    
    surface_interpolee = griddata(
        coords_valides, 
        valeurs_valides, 
        (grille_x, grille_y), 
        method='cubic' 
    )

    # Si la méthode cubique crée des NaN sur les extrêmes bords de l'image, 
    # on colmate avec la méthode la plus proche (sécurité)
    if np.isnan(surface_interpolee).any():
        print("⚠️ Colmatage des bords extrêmes...")
        mask_nan = np.isnan(surface_interpolee)
        surface_nearest = griddata(coords_valides, valeurs_valides, (grille_x, grille_y), method='nearest')
        surface_interpolee[mask_nan] = surface_nearest[mask_nan]

    # 5. LISSAGE GAUSSIEN (La courbe parfaite)
    print(f"🌪️ Application du lissage Gaussien (Force : {FORCE_LISSAGE_GAUSSIEN})...")
    surface_lisse_finale = gaussian_filter(surface_interpolee, sigma=FORCE_LISSAGE_GAUSSIEN)

    # Affichage de la courbe brute : on met les trous en NaN pour couper la ligne rouge
    surface_brute_affichage = surface_brute_cpu.copy()
    surface_brute_affichage[~masque_valide_cpu] = np.nan

    return volume_cpu, surface_brute_affichage, surface_lisse_finale

def afficher_parcours(volume, surface_brute, surface_finale):
    if volume is None:
        return

    nx, ny, nz = volume.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    y_init = ny // 2
    img = ax.imshow(volume[:, y_init, :].T, cmap='gray', aspect='auto', vmax=np.percentile(volume, 99))
    
    line_brute, = ax.plot(surface_brute[:, y_init], color='red', linewidth=1, linestyle=':', alpha=0.8, label='Surface Brute (filtrée)')
    line_finale, = ax.plot(surface_finale[:, y_init], color='lime', linewidth=2, label='Surface Parfaite (Interpolée + Gaussien)')
    
    ax.set_title(f"Détection Topographique - Tranche Y = {y_init}/{ny-1}")
    ax.set_ylabel("Z (Profondeur laser)")
    ax.set_xlabel("X")
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_y = Slider(ax=ax_slider, label='Tranche Y', valmin=0, valmax=ny - 1, valinit=y_init, valstep=1)

    def update(val):
        y = int(slider_y.val)
        img.set_data(volume[:, y, :].T)
        line_brute.set_ydata(surface_brute[:, y])
        line_finale.set_ydata(surface_finale[:, y])
        ax.set_title(f"Détection Topographique - Tranche Y = {y}/{ny-1}")
        fig.canvas.draw_idle()

    slider_y.on_changed(update)
    plt.show()

if __name__ == "__main__":
    vol, surf_b, surf_f = extraction_peau_topographique_bande_passante()
    afficher_parcours(vol, surf_b, surf_f)