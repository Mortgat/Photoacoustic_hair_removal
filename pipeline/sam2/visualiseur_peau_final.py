import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
import os
from scipy.ndimage import rotate

# Gestion du GPU pour le filtre de surface
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter 
    GPU_ACTIF = True
except ImportError:
    from scipy.ndimage import median_filter
    GPU_ACTIF = False

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_MAT = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
FICHIER_METADATA = "transformation_metadata.json"
FICHIER_SAM2 = "masque_sam2_v4.npz"

# --- PARAMÈTRES GÉOMÉTRIQUES ---
MULTIPLICATEUR_SEUIL_BAS = 2.0
PERCENTILE_COUPURE_HAUTE = 93.0
TAILLE_FILTRE_SURFACE = 30
# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def lancer_visualiseur_rawdata():
    print("--- 👁️ VISUALISATION CLINIQUE : GÉOMÉTRIE vs SAM 2 SUR RAW DATA ---")

    if not os.path.exists(FICHIER_MAT) or not os.path.exists(FICHIER_SAM2):
        print("❌ Erreur : Fichier .mat ou .npz introuvable.")
        return

    # 1. Chargement des métadonnées et de la matrice
    with open(FICHIER_METADATA, 'r') as f: meta = json.load(f)

    print(f"⏳ Chargement du volume .mat ({FICHIER_MAT})...")
    mat = scipy.io.loadmat(FICHIER_MAT)
    volume_brut = mat[trouver_variable_volume(mat)]

    angle = meta["angle_rotation_applique"]
    if abs(angle) > 0:
        volume_brut = rotate(volume_brut, angle, axes=(0, 1), reshape=True, order=1)
        
    x_min, x_max = meta["rognage_x"]["min"], meta["rognage_x"]["max"]
    volume = volume_brut[x_min:x_max, :, :]
    nx, ny, nz = volume.shape

    # 2. Extraction Géométrique (Calcul Hybride CPU/GPU)
    print("🕵️ Calcul de la méthode géométrique pure (Hybride CPU/GPU)...")
    
    vol_cpu = np.asarray(volume)
    seuil_bas = vol_cpu.mean() + MULTIPLICATEUR_SEUIL_BAS * vol_cpu.std()
    
    masque_brillant = (vol_cpu > seuil_bas)
    pixels_utiles = vol_cpu[masque_brillant]
    seuil_haut = np.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else np.max(vol_cpu)
    
    masque_brillant &= (vol_cpu < seuil_haut)
    surface_brute_z = np.argmax(masque_brillant, axis=2)
    
    colonnes_vides = np.max(vol_cpu, axis=2) < seuil_bas
    surface_brute_z[colonnes_vides] = nz 
    
    if GPU_ACTIF:
        surface_brute_gpu = cp.asarray(surface_brute_z)
        surface_lisse_gpu = median_filter(surface_brute_gpu, size=TAILLE_FILTRE_SURFACE)
        surface_lisse_z = surface_lisse_gpu.get()
    else:
        surface_lisse_z = median_filter(surface_brute_z, size=TAILLE_FILTRE_SURFACE)

    surface_lisse_cpu = surface_lisse_z.astype(float)
    surface_lisse_cpu[colonnes_vides] = np.nan # Disparition dans le vide

    # 3. Chargement de SAM 2 et préparation de l'inversion
    print("⏳ Chargement du masque SAM 2...")
    sam2_masques = np.load(FICHIER_SAM2)["masque"]

    pad_top = meta["format_sam2"]["padding"]["top"]
    pad_left = meta["format_sam2"]["padding"]["left"]
    taille_max = max(meta["format_sam2"]["shape_rognee"]["largeur_w"], meta["format_sam2"]["shape_rognee"]["hauteur_h"])
    ratio_scale = 1024.0 / taille_max

    def inverser_coordonnees(val_1024, pad):
        return (val_1024 / ratio_scale) - pad

    # 4. Interface Graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    fig.canvas.manager.set_window_title("Validation Clinique : Geo vs SAM 2")

    y_init = ny // 2
    vmax_val = np.percentile(volume, 99)
    img = ax.imshow(volume[:, y_init, :].T, cmap='gray', aspect='auto', vmax=vmax_val)
    
    # Lignes superposées
    line_geo, = ax.plot(np.arange(nx), surface_lisse_cpu[:, y_init], color='lime', linewidth=2.5, alpha=0.7, label='Méthode Géométrique')
    line_sam, = ax.plot([], [], color='cyan', linewidth=1.5, label='Peau SAM 2 (Projetée)')
    
    ax.set_title(f"Comparaison sur Raw Data - Tranche Y = {y_init}/{ny-1}")
    ax.set_ylabel("Z (Profondeur laser)")
    ax.set_xlabel("X")
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_y = Slider(ax=ax_slider, label='Parcourir Y', valmin=0, valmax=ny - 1, valinit=y_init, valstep=1)

    def update(val):
        y = int(slider_y.val)
        
        # Fond & Géométrie
        img.set_data(volume[:, y, :].T)
        line_geo.set_ydata(surface_lisse_cpu[:, y])
        
        # Inversion SAM 2
        masque_frame = sam2_masques[y]
        colonnes_valides = np.any(masque_frame, axis=0)
        x_1024 = np.where(colonnes_valides)[0]
        
        if len(x_1024) > 0:
            z_1024 = np.argmax(masque_frame[:, x_1024], axis=0)
            x_brut = inverser_coordonnees(x_1024, pad_left)
            z_brut = inverser_coordonnees(z_1024, pad_top)
            line_sam.set_data(x_brut, z_brut)
        else:
            line_sam.set_data([], [])
            
        ax.set_title(f"Comparaison sur Raw Data - Tranche Y = {y}/{ny-1}")
        fig.canvas.draw_idle()

    slider_y.on_changed(update)
    update(y_init)
    plt.show()

if __name__ == "__main__":
    lancer_visualiseur_rawdata()