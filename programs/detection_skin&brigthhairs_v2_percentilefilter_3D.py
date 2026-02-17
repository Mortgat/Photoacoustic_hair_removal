import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from scipy.interpolate import griddata

# NOUVEAU : On importe le filtre percentile
from scipy.ndimage import percentile_filter 

try:
    import cupy as cp
    GPU_ACTIF = True
except ImportError:
    import numpy as cp
    GPU_ACTIF = False

fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# --- PARAMÈTRES OPTIMISÉS ---
MULTIPLICATEUR_SEUIL_BAS = 2.0
PERCENTILE_COUPURE_HAUTE = 95.0 
TAILLE_FILTRE_SURFACE = 60 

# --- LE NOUVEAU PARAMÈTRE CLÉ ---
# 50 = Filtre médian classique. 
# 85 = On force la ligne à aller chercher le fond (la peau) en ignorant la canopée de poils
PERCENTILE_SURFACE = 85  

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def extraction_peau_percentile():
    print("--- 🕵️ EXTRACTION PEAU (FILTRE PERCENTILE BIAISÉ VERS LE FOND) ---")
    
    if not os.path.exists(fichier_mat):
        return None, None, None

    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume_cpu = mat[key]
    nx, ny, nz = volume_cpu.shape
    volume = cp.asarray(volume_cpu)

    # 1. Seuils
    seuil_bas = volume.mean() + MULTIPLICATEUR_SEUIL_BAS * volume.std()
    pixels_utiles = volume[volume > seuil_bas]
    seuil_haut = cp.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else cp.max(volume)

    # 2. Détection Brute
    masque_brillant = (volume > seuil_bas) & (volume < seuil_haut)
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    # 3. Identification du vrai vide
    max_colonnes = cp.max(volume, axis=2)
    masque_valide = max_colonnes >= seuil_bas

    # 4. Rapatriement sur CPU
    print("⬅️ Rapatriement sur CPU pour le filtrage...")
    if GPU_ACTIF:
        surface_brute_cpu = surface_brute_z.get().astype(float)
        masque_valide_cpu = masque_valide.get()
    else:
        surface_brute_cpu = surface_brute_z.astype(float)
        masque_valide_cpu = masque_valide

    # 5. EXTRAPOLATION (Pour éviter l'effet falaise sur les bords qu'on a vu avant)
    coords_valides = np.array(np.nonzero(masque_valide_cpu)).T
    valeurs_valides = surface_brute_cpu[masque_valide_cpu]
    grille_x, grille_y = np.mgrid[0:nx, 0:ny]
    
    surface_etendue = griddata(
        coords_valides, 
        valeurs_valides, 
        (grille_x, grille_y), 
        method='nearest'
    )

    # 6. FILTRAGE PERCENTILE (La solution à la forte densité de poils)
    print(f"🌪️ Application du Filtre Percentile (P={PERCENTILE_SURFACE}, Taille {TAILLE_FILTRE_SURFACE}x{TAILLE_FILTRE_SURFACE})...")
    # percentile_filter va naturellement se caler sur les valeurs de Z les plus grandes (donc les plus profondes = la peau)
    surface_lisse_cpu = percentile_filter(surface_etendue, percentile=PERCENTILE_SURFACE, size=TAILLE_FILTRE_SURFACE)

    # 7. DÉCOUPAGE ("Cookie-Cutter")
    print("✂️ Nettoyage des zones de vide...")
    surface_brute_cpu[~masque_valide_cpu] = np.nan
    surface_lisse_cpu[~masque_valide_cpu] = np.nan

    return volume_cpu, surface_brute_cpu, surface_lisse_cpu

def afficher_parcours(volume, surface_brute, surface_finale):
    if volume is None:
        return

    nx, ny, nz = volume.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    y_init = ny // 2
    img = ax.imshow(volume[:, y_init, :].T, cmap='gray', aspect='auto', vmax=np.percentile(volume, 99))
    
    line_brute, = ax.plot(surface_brute[:, y_init], color='red', linewidth=1, linestyle=':', alpha=0.8, label='Surface Brute')
    line_finale, = ax.plot(surface_finale[:, y_init], color='lime', linewidth=2, label=f'Surface Lissée (Percentile {PERCENTILE_SURFACE})')
    
    ax.set_title(f"Détection de la Peau - Tranche Y = {y_init}/{ny-1}")
    ax.set_ylabel("Z (Profondeur laser)")
    ax.set_xlabel("X")
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_y = Slider(ax=ax_slider, label='Parcourir Y', valmin=0, valmax=ny - 1, valinit=y_init, valstep=1)

    def update(val):
        y = int(slider_y.val)
        img.set_data(volume[:, y, :].T)
        line_brute.set_ydata(surface_brute[:, y])
        line_finale.set_ydata(surface_finale[:, y])
        ax.set_title(f"Détection de la Peau - Tranche Y = {y}/{ny-1}")
        fig.canvas.draw_idle()

    slider_y.on_changed(update)
    plt.show()

if __name__ == "__main__":
    vol, surf_b, surf_f = extraction_peau_percentile()
    afficher_parcours(vol, surf_b, surf_f)