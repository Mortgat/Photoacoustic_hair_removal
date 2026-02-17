import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# ==========================================
# IMPORT GPU (CUPY)
# ==========================================
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter
    GPU_ACTIF = True
    print("🚀 CuPy détecté : Le calcul tournera sur le GPU.")
except ImportError:
    print("⚠️ CuPy non trouvé. Bascule sur CPU.")
    import numpy as cp
    from scipy.ndimage import median_filter
    GPU_ACTIF = False

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
TAILLE_FILTRE_3D = 30  

# --- PARAMÈTRES CONSERVATIFS ---
MULTIPLICATEUR_SEUIL_BAS = 2

# Très conservatif : on garde 99.5% du signal utile. 
# On ne considère comme "trop brillant" que le top 0.5%.
PERCENTILE_COUPURE_HAUTE = 96.5

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def creer_masque_peau_3d():
    print("--- 🕵️ EXTRACTION PEAU (FILTRE 3D + BANDE PASSANTE CONSERVATIVE) ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier introuvable : {fichier_mat}")
        return None, None, None

    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume_cpu = mat[key]
    nx, ny, nz = volume_cpu.shape

    print("📦 Transfert vers le GPU...")
    volume = cp.asarray(volume_cpu)

    # 1. Seuil Bas
    moyenne = volume.mean()
    std = volume.std()
    seuil_bas = moyenne + MULTIPLICATEUR_SEUIL_BAS * std
    
    # 2. Seuil Haut (Approche Ultra-Conservative)
    pixels_utiles = volume[volume > seuil_bas]
    if pixels_utiles.size > 0:
        seuil_haut = cp.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE)
    else:
        seuil_haut = cp.max(volume)

    print(f"📊 Statistiques de détection :")
    print(f"   - Seuil Bas (Moy + 2.5*Std) : {seuil_bas:.2f}")
    print(f"   - Seuil Haut (Top {100 - PERCENTILE_COUPURE_HAUTE:.1f}% exclus) : {seuil_haut:.2f}")

    # 3. Argmax avec Bande Passante
    print("🔍 Scan des colonnes...")
    masque_brillant = (volume > seuil_bas) & (volume < seuil_haut)
    surface_brute_z = cp.argmax(masque_brillant, axis=2)

    # 4. Gestion du vide absolu (colonnes qui n'atteignent même pas le seuil bas)
    max_colonnes = cp.max(volume, axis=2)
    colonnes_vides = max_colonnes < seuil_bas
    
    # 5. Création Masque 3D
    print("🧱 Construction du masque binaire 3D...")
    Z_grid = cp.arange(nz).reshape(1, 1, nz)
    surface_brute_expanded = surface_brute_z[:, :, cp.newaxis]
    
    mask_peau_3d = (Z_grid >= surface_brute_expanded).astype(cp.uint8)
    mask_peau_3d[colonnes_vides, :] = 0

    # 6. Filtre Médian 3D
    print(f"🌪️ Application du Filtre Médian 3D (Taille {TAILLE_FILTRE_3D})...")
    mask_peau_3d_lisse = median_filter(mask_peau_3d, size=TAILLE_FILTRE_3D)

    # 7. Extraction Surface Finale
    print("🌍 Extraction de la surface lissée...")
    surface_finale_z = cp.argmax(mask_peau_3d_lisse > 0, axis=2)
    colonnes_vides_finales = cp.max(mask_peau_3d_lisse, axis=2) == 0

    print("⬅️ Rapatriement sur CPU pour l'affichage...")
    if GPU_ACTIF:
        surface_brute_cpu = surface_brute_z.get().astype(float)
        colonnes_vides_cpu = colonnes_vides.get()
        surface_finale_cpu = surface_finale_z.get().astype(float)
        colonnes_vides_finales_cpu = colonnes_vides_finales.get()
    else:
        surface_brute_cpu = surface_brute_z.astype(float)
        colonnes_vides_cpu = colonnes_vides
        surface_finale_cpu = surface_finale_z.astype(float)
        colonnes_vides_finales_cpu = colonnes_vides_finales

    surface_brute_cpu[colonnes_vides_cpu] = np.nan
    surface_finale_cpu[colonnes_vides_finales_cpu] = np.nan

    return volume_cpu, surface_brute_cpu, surface_finale_cpu

def afficher_parcours_tranches(volume, surface_brute, surface_finale):
    if volume is None:
        return

    nx, ny, nz = volume.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    y_init = ny // 2
    coupe_vol = volume[:, y_init, :].T

    img = ax.imshow(coupe_vol, cmap='gray', aspect='auto', vmax=np.percentile(volume, 99))
    
    # Ligne brute en pointillés fins pour voir l'effet du rejet des extrêmes
    line_brute, = ax.plot(surface_brute[:, y_init], color='red', linewidth=1, linestyle=':', alpha=0.8, label=f'Surface Brute (<= P{PERCENTILE_COUPURE_HAUTE})')
    line_finale, = ax.plot(surface_finale[:, y_init], color='lime', linewidth=2, label='Surface Continue (Lissée 3D)')
    
    ax.set_title(f"Détection de la Peau - Tranche Y = {y_init}/{ny-1}")
    ax.set_ylabel("Z (Profondeur laser)")
    ax.set_xlabel("X")
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_y = Slider(
        ax=ax_slider,
        label='Parcourir Y',
        valmin=0,
        valmax=ny - 1,
        valinit=y_init,
        valstep=1
    )

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
    vol, surf_brute, surf_finale = creer_masque_peau_3d()
    afficher_parcours_tranches(vol, surf_brute, surf_finale)