import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from PIL import Image

def generer_images_debug():
    print("--- 🐛 LANCEMENT DU DEBUGGING (POINTS + BOX) ---")
    
    DOSSIER_FRAMES = "frames_sam2" # Modifie par frames_sam2_inv si tu utilises toujours les inversées
    FICHIER_PROMPTS = "prompts_sam2.json"
    DOSSIER_SORTIE = "verif_debug_v2"
    
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    with open(FICHIER_PROMPTS, 'r') as f:
        donnees = json.load(f)
        
    fichiers_jpg = sorted([f for f in os.listdir(DOSSIER_FRAMES) if f.endswith('.jpg')])
    prompts_a_tester = donnees["prompts"][:5] 
    
    for p in prompts_a_tester:
        frame_idx = p["frame_index"]
        points = np.array(p["points"])
        labels = np.array(p["labels"])
        box = p.get("box", None)
        
        chemin_img = os.path.join(DOSSIER_FRAMES, fichiers_jpg[frame_idx])
        img = Image.open(chemin_img)
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.imshow(img, cmap='gray')
        
        # Dessin des points
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='*', s=150, label='Clic Positif')
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=100, label='Clic Négatif')
            
        # Dessin de la Bounding Box
        if box is not None:
            x_min, y_min, x_max, y_max = box
            ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                           linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'))
            
        plt.title(f"Debug Frame Y={frame_idx}", color='white')
        plt.legend(loc='upper right')
        plt.axis('off')
        
        plt.savefig(os.path.join(DOSSIER_SORTIE, f"debug_frame_{frame_idx}.jpg"), dpi=150, facecolor='black', bbox_inches='tight')
        plt.close()
        
    print(f"✅ Terminé ! Va ouvrir les images dans '{DOSSIER_SORTIE}'.")

if __name__ == "__main__":
    generer_images_debug()