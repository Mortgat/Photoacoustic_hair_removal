import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from scipy.interpolate import griddata

# NOUVEAU : Import de la fermeture morphologique
from scipy.ndimage import median_filter, grey_closing

try:
    import cupy as cp
    GPU_ACTIF = True
except ImportError:
    import numpy as cp
    GPU_ACTIF = False

fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# --- PARAMÈTRES GÉNÉRALISABLES ---
MULTIPLICATEUR_SEUIL_BAS = 2.0
PERCENTILE_COUPURE_HAUTE = 95.0 

# La taille de fermeture doit juste être plus large que le plus gros paquet de poils possible.
# Si tu mets 60, ça détruit les forêts de poils de 60 pixels de large. 
# S'il n'y a pas de poils, ça ne change RIEN à la peau ! C'est parfaitement stable.
TAILLE_FERMETURE = 60 

# Un petit lissage à la fin pour rendre la courbe esthétique et gommer les "marches d'escalier"
TAILLE_LISSAGE = 15 

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def extraction_peau_morphologique():
    print("--- 🕵️ EXTRACTION PEAU (FERMETURE MORPHOLOGIQUE ROBUSTE) ---")
    
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
    print("⬅️ Rapatriement sur CPU...")
    if GPU_ACTIF:
        surface_brute_cpu = surface_brute_z.get().astype(float)
        masque_valide_cpu = masque_valide.get()
    else:
        surface_brute_cpu = surface_brute_z.astype(float)
        masque_valide_cpu = masque_valide

    # 5. EXTRAPOLATION (Pour stabiliser les bords)
    coords_valides = np.array(np.nonzero(masque_valide_cpu)).T
    valeurs_valides = surface_brute_cpu[masque_valide_cpu]
    grille_x, grille_y = np.mgrid[0:nx, 0:ny]
    
    surface_etendue = griddata(
        coords_valides, 
        valeurs_valides, 
        (grille_x, grille_y), 
        method='nearest'
    )

    # 6. ÉTAPE CLÉ : FERMETURE MORPHOLOGIQUE (Destruction de la canopée)
    print(f"🧱 Rabotage des poils (Fermeture morphologique, Taille {TAILLE_FERMETURE}x{TAILLE_FERMETURE})...")
    # Efface les valeurs Z petites (poils) en conservant la ligne de base (peau)
    surface_sans_poils = grey_closing(surface_etendue, size=TAILLE_FERMETURE)

    # 7. LISSAGE FINAL
    print(f"🌪️ Lissage de finition (Filtre Médian, Taille {TAILLE_LISSAGE}x{TAILLE_LISSAGE})...")
    # Adoucit la ligne après le passage de la morphologie
    surface_lisse_cpu = median_filter(surface_sans_poils, size=TAILLE_LISSAGE)

    # 8. DÉCOUPAGE ("Cookie-Cutter")
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
    line_finale, = ax.plot(surface_finale[:, y_init], color='lime', linewidth=2, label=f'Surface Morphologique')
    
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
    vol, surf_b, surf_f = extraction_peau_morphologique()
    afficher_parcours(vol, surf_b, surf_f)