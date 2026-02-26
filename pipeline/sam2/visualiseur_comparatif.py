import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
DOSSIER_FRAMES = "frames_sam2" 
FICHIER_REFERENCE = "reference_geometrique.npz"
FICHIER_SAM2 = "masque_sam2_v4.npz"
# ==========================================

def lancer_visualiseur():
    print("--- 🔍 CHARGEMENT DU COMPARATEUR (GÉOMÉTRIE vs SAM 2) ---")
    
    if not os.path.exists(FICHIER_SAM2) or not os.path.exists(FICHIER_REFERENCE):
        print("❌ Erreur : Les fichiers .npz sont introuvables. As-tu bien fait tourner la V4 ?")
        return

    # 1. Chargement des données
    fichiers_jpg = sorted([f for f in os.listdir(DOSSIER_FRAMES) if f.endswith('.jpg')])
    nb_frames = len(fichiers_jpg)
    
    print("⏳ Chargement de la Vérité Terrain...")
    reference_z = np.load(FICHIER_REFERENCE)["pure_z_1024"]
    
    print("⏳ Chargement des Masques SAM 2...")
    sam2_masques = np.load(FICHIER_SAM2)["masque"]

    # 2. Configuration de l'interface Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1e1e1e')
    fig.canvas.manager.set_window_title("Comparateur de Peau : Géométrie vs IA")
    plt.subplots_adjust(bottom=0.2) # On laisse de la place pour le slider

    # Thème sombre pour les axes
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Initialisation de l'image de fond
    img_display = ax.imshow(np.zeros((1024, 1024)), cmap='gray', aspect='auto')
    
    # Initialisation des deux courbes
    line_geo, = ax.plot([], [], color='lime', linewidth=3, alpha=0.8, label='Géométrie Pure')
    scatter_sam, = ax.plot([], [], color='red', marker='.', markersize=5, linestyle='none', label='SAM 2 (Toit du ruban)')
    
    titre_ax = ax.set_title("Chargement...", color='white', pad=10, fontsize=14)
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    # 3. Fonction de mise à jour synchronisée
    def update(val):
        idx = int(slider.val)
        
        # Affichage de l'image brute
        img_path = os.path.join(DOSSIER_FRAMES, fichiers_jpg[idx])
        img_array = np.array(Image.open(img_path))
        img_display.set_data(img_array)
        
        # --- MISE À JOUR GÉOMÉTRIE ---
        ref_y = reference_z[idx]
        x_geo = np.where(ref_y != -1.0)[0]
        if len(x_geo) > 0:
            line_geo.set_data(x_geo, ref_y[x_geo])
        else:
            line_geo.set_data([], [])

        # --- MISE À JOUR SAM 2 (Extraction du Toit) ---
        masque_frame = sam2_masques[idx]
        colonnes_valides = np.any(masque_frame, axis=0)
        x_sam = np.where(colonnes_valides)[0]
        
        if len(x_sam) > 0:
            # On cherche le premier pixel "True" en partant du haut pour chaque colonne valide
            z_sam = np.argmax(masque_frame[:, x_sam], axis=0)
            scatter_sam.set_data(x_sam, z_sam)
        else:
            scatter_sam.set_data([], [])

        # Mise à jour des textes
        nb_geo = len(x_geo)
        nb_sam = len(x_sam)
        titre_ax.set_text(f"Comparaison - Tranche Y = {idx}/{nb_frames-1} | Pixels : Geo({nb_geo}), SAM2({nb_sam})")
        
        fig.canvas.draw_idle()

    # 4. Création du Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='gray')
    slider = Slider(
        ax=ax_slider,
        label='Parcourir Y',
        valmin=0,
        valmax=nb_frames - 1,
        valinit=0,
        valstep=1,
        color='cyan'
    )
    slider.label.set_color('white')
    slider.valtext.set_color('white')
    slider.on_changed(update)

    # Affichage initial
    update(0)
    plt.show()

if __name__ == "__main__":
    lancer_visualiseur()