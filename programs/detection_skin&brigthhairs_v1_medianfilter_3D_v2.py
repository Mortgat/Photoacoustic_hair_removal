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
MULTIPLICATEUR_SEUIL = 2  # Plus bas = plus tolérant au sombre (ex: 2.5 ou 3.0 au lieu de 4.0)
FORCE_LISSAGE_GAUSSIEN = 30  # Plus grand = ligne plus tendue et lisse (gomme totalement les poils)

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def extraction_peau_topographique():    
    print("--- 🕵️ EXTRACTION DE LA PEAU (MÉTHODE TOPOGRAPHIQUE) ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier introuvable : {fichier_mat}")
        return None, None, None

    # 1. Chargement et transfert GPU
    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume_cpu = mat[key]
    nx, ny, nz = volume_cpu.shape
    volume = cp.asarray(volume_cpu)

    # 2. Seuillage légèrement plus permissif
    moyenne = volume.mean()
    std = volume.std()
    seuil = moyenne + MULTIPLICATEUR_SEUIL * std
    print(f"📊 Seuil calculé : {seuil:.2f}")

    # 3. Extraction de la surface brute (Carte Z)
    print("🔍 Scan des colonnes...")
    masque_brillant = volume > seuil
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    # Identification des colonnes valides/invalides
    max_colonnes = cp.max(volume, axis=2)
    masque_valide = max_colonnes >= seuil

    # Rapatriement sur CPU pour l'interpolation (très rapide sur CPU)
    print("⬅️ Rapatriement sur CPU pour lissage mathématique...")
    if GPU_ACTIF:
        surface_brute_cpu = surface_brute_z.get().astype(float)
        masque_valide_cpu = masque_valide.get()
    else:
        surface_brute_cpu = surface_brute_z.astype(float)
        masque_valide_cpu = masque_valide

    # 4. INTERPOLATION (Le rebouchage intelligent des trous géants)
    print("🌉 Construction des ponts (Interpolation des zones sombres)...")
    
    # On récupère les coordonnées (X, Y) des pixels valides et leurs hauteurs (Z)
    coords_valides = np.array(np.nonzero(masque_valide_cpu)).T
    valeurs_valides = surface_brute_cpu[masque_valide_cpu]
    
    # On prépare la grille complète (X, Y)
    grille_x, grille_y = np.mgrid[0:nx, 0:ny]
    
    # griddata va boucher les trous en se basant sur les pixels valides les plus proches
    surface_interpolee = griddata(
        coords_valides, 
        valeurs_valides, 
        (grille_x, grille_y), 
        method='linear' # 'nearest' bouche tout, on lissera juste après
    )

    # 5. LISSAGE GAUSSIEN (La courbe parfaite)
    print(f"🌪️ Application du lissage Gaussien (Force : {FORCE_LISSAGE_GAUSSIEN})...")
    surface_lisse_finale = gaussian_filter(surface_interpolee, sigma=FORCE_LISSAGE_GAUSSIEN)

    # Pour l'affichage de la courbe brute : on met les trous en NaN pour couper la ligne rouge
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
    
    line_brute, = ax.plot(surface_brute[:, y_init], color='red', linewidth=1, linestyle=':', alpha=0.8, label='Surface Brute (avec trous)')
    line_finale, = ax.plot(surface_finale[:, y_init], color='lime', linewidth=2, label='Surface Parfaite (Interpolée + Gaussien)')
    
    ax.set_title(f"Détection Topographique - Tranche Y = {y_init}/{ny-1}")
    ax.set_ylabel("Z (Profondeur)")
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
    vol, surf_b, surf_f = extraction_peau_topographique()
    afficher_parcours(vol, surf_b, surf_f)