import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

def generer_images_debug():
    print("--- 🐛 LANCEMENT DU DEBUGGING VISUEL ---")
    
    DOSSIER_FRAMES = "frames_sam2"
    FICHIER_PROMPTS = "prompts_sam2.json"
    FICHIER_MASQUE_SAM = "masque_sam2_brut.npz"
    DOSSIER_SORTIE = "verif_debug"
    
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    # 1. Chargement des données brutes de SAM 2
    print(f"Chargement des prompts depuis {FICHIER_PROMPTS}...")
    with open(FICHIER_PROMPTS, 'r') as f:
        donnees = json.load(f)
        
    print(f"Chargement du tenseur 1024x1024 depuis {FICHIER_MASQUE_SAM}...")
    masques_bruts = np.load(FICHIER_MASQUE_SAM)['masque']
    
    fichiers_jpg = sorted([f for f in os.listdir(DOSSIER_FRAMES) if f.endswith('.jpg')])
    
    # On prend les 5 premières frames qui contiennent des clics cliniques
    prompts_a_tester = donnees["prompts"][:5] 
    
    for p in prompts_a_tester:
        frame_idx = p["frame_index"]
        points = np.array(p["points"])
        labels = np.array(p["labels"])
        
        print(f"Génération de l'image d'inspection pour la frame {frame_idx}...")
        
        # Image JPEG de fond (en 1024x1024)
        chemin_img = os.path.join(DOSSIER_FRAMES, fichiers_jpg[frame_idx])
        img = Image.open(chemin_img)
        
        # Masque brut sorti par SAM 2
        masque = masques_bruts[frame_idx]
        
        plt.figure(figsize=(10, 10), facecolor='black')
        plt.imshow(img, cmap='gray')
        
        # Calque rouge pour le masque SAM 2
        overlay = np.zeros((*masque.shape, 4))
        overlay[..., 0] = 1.0 
        overlay[..., 3] = masque * 0.4 
        plt.imshow(overlay)
        
        # Dessin des points d'amorçage
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        
        if len(pos_points) > 0:
            plt.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='*', s=150, label='Clic Positif (Peau)')
        if len(neg_points) > 0:
            plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=100, label='Clic Négatif (Fond)')
            
        plt.title(f"Debug Frame Y={frame_idx} (Masque Brut + Prompts)", color='white')
        plt.legend(loc='upper right')
        plt.axis('off')
        
        chemin_sortie = os.path.join(DOSSIER_SORTIE, f"debug_frame_{frame_idx}.jpg")
        plt.savefig(chemin_sortie, dpi=150, facecolor='black', bbox_inches='tight')
        plt.close()
        
    print(f"✅ Terminé ! Va ouvrir les images dans le dossier '{DOSSIER_SORTIE}'.")

if __name__ == "__main__":
    generer_images_debug()