import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from PIL import Image

def generer_images_debug():
    print("--- 🐛 DEBUG V4 (MASK PROMPTING / RUBAN) ---")
    
    DOSSIER_FRAMES = "frames_sam2"
    FICHIER_PROMPTS = "prompts_sam2_v4.json"
    FICHIER_MASK = "mask_prompt_v4.npy"
    DOSSIER_SORTIE = "verif_debug_v4"
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    with open(FICHIER_PROMPTS, 'r') as f: donnees = json.load(f)
    f_idx = donnees["initial_frame_index"]
    box = donnees["box"]
    
    mask = np.load(FICHIER_MASK)
    
    fichiers_jpg = sorted([f for f in os.listdir(DOSSIER_FRAMES) if f.endswith('.jpg')])
    img = Image.open(os.path.join(DOSSIER_FRAMES, fichiers_jpg[f_idx]))
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.imshow(img, cmap='gray')
    
    # Affichage du Ruban (Masque) en rouge transparent
    overlay = np.zeros((1024, 1024, 4))
    overlay[..., 0] = 1.0 # Rouge
    overlay[..., 3] = mask * 0.5 # Opacité 50% là où il y a le ruban
    ax.imshow(overlay)
        
    # Affichage de la Bounding Box
    x_min, y_min, x_max, y_max = box
    ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                   linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'))
        
    plt.title(f"Debug Ruban - Golden Frame Y={f_idx}", color='white')
    plt.axis('off')
    
    chemin_sortie = os.path.join(DOSSIER_SORTIE, f"debug_ruban_{f_idx}.jpg")
    plt.savefig(chemin_sortie, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
        
    print(f"✅ Terminé ! Va vérifier ton Rideau Rouge dans '{DOSSIER_SORTIE}'.")

if __name__ == "__main__":
    generer_images_debug()