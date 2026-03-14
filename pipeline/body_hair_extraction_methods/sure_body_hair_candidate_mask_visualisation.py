import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

def visualiser_resultats():
    fichier_volume = r"dicom_data/dicom/4261_fromdcm.npz"
    fichier_poils = r"pipeline/body_hair_extraction_methods/masque_positifs_pu.npy"
    fichier_peau = r"pipeline/body_hair_extraction_methods/surface_peau.npy"

    if not os.path.exists(fichier_volume) or not os.path.exists(fichier_poils) or not os.path.exists(fichier_peau):
        print("Erreur : Un ou plusieurs fichiers sont introuvables. Vérifiez les chemins.")
        return

    print("Chargement des données en cours...")
    data_vol = np.load(fichier_volume)
    volume = data_vol['volume']
    
    # Chargement direct des tableaux NumPy
    masque_positifs = np.load(fichier_poils)
    surface_z = np.load(fichier_peau)
    
    nx, ny, nz = volume.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    y_init = ny // 2
    
    vmax_val = np.percentile(volume[::10, y_init, ::10], 99)
    img_vol = ax.imshow(volume[:, y_init, :].T, cmap='gray', aspect='auto', vmax=vmax_val)
    
    overlay = np.zeros((nz, nx, 4), dtype=float)
    
    def maj_overlay(y_idx):
        overlay.fill(0)
        tranche_masque = masque_positifs[:, y_idx, :].T
        overlay[tranche_masque, 0] = 1.0  # R
        overlay[tranche_masque, 1] = 0.0  # G
        overlay[tranche_masque, 2] = 1.0  # B
        overlay[tranche_masque, 3] = 1.0  # Alpha
        return overlay

    img_mask = ax.imshow(maj_overlay(y_init), aspect='auto')
    
    line_surf, = ax.plot(surface_z[:, y_init], color='lime', linewidth=1.5, label='Surface Peau')

    ax.set_title(f"Visualisation PU (Percentile 99.9) - Tranche Y = {y_init}/{ny-1}")
    ax.set_ylabel("Z (Profondeur)")
    ax.set_xlabel("X")
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_y = Slider(ax=ax_slider, label='Y', valmin=0, valmax=ny - 1, valinit=y_init, valstep=1)

    def update(val):
        y = int(slider_y.val)
        img_vol.set_data(volume[:, y, :].T)
        img_mask.set_data(maj_overlay(y))
        line_surf.set_ydata(surface_z[:, y])
        ax.set_title(f"Visualisation PU - Tranche Y = {y}/{ny-1}")
        fig.canvas.draw_idle()

    slider_y.on_changed(update)
    plt.show()

if __name__ == "__main__":
    visualiser_resultats()