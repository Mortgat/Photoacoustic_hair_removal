import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Dossier où se trouvent tes images préparées
DOSSIER_FRAMES = "frames_sam2"

def verifier_donnees():
    print(f"🔍 Recherche des images dans '{DOSSIER_FRAMES}'...")
    
    # 1. Lister et trier toutes les images JPEG
    chemins_images = sorted(glob.glob(os.path.join(DOSSIER_FRAMES, "*.jpg")))
    
    if not chemins_images:
        print(f"❌ Aucune image trouvée dans le dossier {DOSSIER_FRAMES}.")
        return

    nb_images = len(chemins_images)
    print(f"✅ {nb_images} images trouvées. Chargement de la première...")

    # 2. Configuration de Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15) # Place pour le slider

    # On charge la première image
    # cv2 lit en BGR. On la convertit en Niveaux de Gris (1 seul canal) pour que la colormap fonctionne
    img_bgr = cv2.imread(chemins_images[nb_images // 2]) 
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Affichage avec la colormap 'turbo' (ultra contrastée)
    # cmap='hot' ou cmap='jet' sont aussi d'excellents choix
    im_display = ax.imshow(img_gray, cmap='turbo', vmin=0, vmax=255)
    
    ax.set_title(f"Vérification SAM 2 - Frame {nb_images // 2} / {nb_images - 1}")
    ax.axis('off') # On cache les axes (0-1024) pour mieux voir l'image

    # 3. Création du Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_y = Slider(ax=ax_slider, label='Frame (Y)', valmin=0, valmax=nb_images - 1, valinit=nb_images // 2, valstep=1)

    # 4. Fonction de mise à jour lors du scroll
    def update(val):
        index = int(slider_y.val)
        # Lecture rapide
        img = cv2.imread(chemins_images[index])
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Mise à jour de l'affichage
        im_display.set_data(img_g)
        ax.set_title(f"Vérification SAM 2 - Frame {index} / {nb_images - 1}")
        fig.canvas.draw_idle()

    slider_y.on_changed(update)
    
    print("🚀 Lancement de l'interface visuelle. Ferme la fenêtre pour quitter.")
    plt.show()

if __name__ == "__main__":
    verifier_donnees()