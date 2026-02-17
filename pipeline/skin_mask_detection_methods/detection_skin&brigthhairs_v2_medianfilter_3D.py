import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# On importe uniquement cupy, plus besoin du lourd filtre 3D !
try:
    import cupy as cp
    # NOUVEAU : On utilise le filtre 2D pour la carte de surface
    from cupyx.scipy.ndimage import median_filter 
    GPU_ACTIF = True
except ImportError:
    import numpy as cp
    from scipy.ndimage import median_filter
    GPU_ACTIF = False

fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# --- PARAMÈTRES ---
MULTIPLICATEUR_SEUIL_BAS = 2
PERCENTILE_COUPURE_HAUTE = 93

# TAILLE DU FILTRE 2D (Topographique)
# Maintenant, 30 signifie vraiment un carré de 30x30 "pixels de peau" sur la surface !
TAILLE_FILTRE_SURFACE = 50

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def extraction_peau_optimisee():
    print("--- 🕵️ EXTRACTION PEAU (FILTRE SURFACE 2D + BANDE PASSANTE) ---")
    
    if not os.path.exists(fichier_mat):
        return None, None, None

    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume_cpu = mat[key]
    nx, ny, nz = volume_cpu.shape
    volume = cp.asarray(volume_cpu)

    # 1. Statistiques
    seuil_bas = volume.mean() + MULTIPLICATEUR_SEUIL_BAS * volume.std()
    pixels_utiles = volume[volume > seuil_bas]
    seuil_haut = cp.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else cp.max(volume)

    # 2. Détection Brute (Le rayon Z s'arrête sur la vraie peau, ignore les gros poils)
    masque_brillant = (volume > seuil_bas) & (volume < seuil_haut)
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    # 3. Gestion du vrai vide (Les colonnes où il n'y a pas de tissu)
    max_colonnes = cp.max(volume, axis=2)
    colonnes_vides = max_colonnes < seuil_bas

    # ON REMET LES ZONES VIDES AU FOND (NZ) POUR NE PAS FAUSSER LE LISSAGE
    surface_brute_z[colonnes_vides] = nz 

    # 4. L'ÉTAPE MAGIQUE : Lissage de la carte de surface en 2D !
    # On ne lisse plus un cube 3D géant. On lisse la carte des altitudes (Z).
    # Ça correspond EXACTEMENT à ce que tu imaginais : regarder les pixels de peau voisins.
    print(f"🌪️ Application du Filtre Médian 2D sur les altitudes (Taille {TAILLE_FILTRE_SURFACE}x{TAILLE_FILTRE_SURFACE})...")
    surface_lisse_z = median_filter(surface_brute_z, size=TAILLE_FILTRE_SURFACE)

    # On remet le vide à 0 pour l'identifier facilement au moment de l'affichage
    surface_lisse_z[colonnes_vides] = 0

    # Rapatriement sur CPU
    if GPU_ACTIF:
        surface_brute_cpu = surface_brute_z.get().astype(float)
        surface_lisse_cpu = surface_lisse_z.get().astype(float)
        colonnes_vides_cpu = colonnes_vides.get()
    else:
        surface_brute_cpu = surface_brute_z.astype(float)
        surface_lisse_cpu = surface_lisse_z.astype(float)
        colonnes_vides_cpu = colonnes_vides

    # Mettre à NaN pour faire disparaître la ligne dans le vide
    surface_brute_cpu[colonnes_vides_cpu] = np.nan
    surface_lisse_cpu[colonnes_vides_cpu] = np.nan

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
    line_finale, = ax.plot(surface_finale[:, y_init], color='lime', linewidth=2, label=f'Surface Lissée (Filtre Surface 2D)')
    
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
    vol, surf_b, surf_f = extraction_peau_optimisee()
    afficher_parcours(vol, surf_b, surf_f)