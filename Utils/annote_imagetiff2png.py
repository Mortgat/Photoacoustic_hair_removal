import scipy.io
import pandas as pd
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
fichier_mat = 'IPH009L_PA797.mat'
nom_variable = 'Data'
fichier_csv = 'endpoint/keio4261_points.csv'
dossier_sortie = 'Sortie_Annotations_PNG'
mode_annotation = 'stack' # Choix : 'stack' (toutes les tranches) ou 'mip' (image unique)
# ---------------------

def annoter():
    print("Chargement des données et du CSV...")
    mat = scipy.io.loadmat(fichier_mat)
    volume = mat[nom_variable]
    df = pd.read_csv(fichier_csv)
    
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    # Récupération des coord (Matlab commence à 1, Python à 0 -> on fait -1)
    pts_x = df['Var1'].values - 1 
    pts_y = df['Var2'].values - 1
    pts_z = df['Var3'].values - 1

    if mode_annotation == 'mip':
        # --- MODE MIP (1 seule image résumée) ---
        print("Génération du MIP annoté...")
        mip = np.max(volume, axis=2)
        
        # Normalisation pour affichage (conversion 16bit -> 8bit visible)
        mip_norm = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Conversion en couleur pour pouvoir mettre du rouge
        img_color = cv2.cvtColor(mip_norm, cv2.COLOR_GRAY2BGR)
        
        # Dessiner toutes les croix
        for x, y in zip(pts_x, pts_y):
            cv2.drawMarker(img_color, (int(x), int(y)), (0, 0, 255), 
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            
        cv2.imwrite(os.path.join(dossier_sortie, "MIP_Annotated.png"), img_color)

    elif mode_annotation == 'stack':
        # --- MODE STACK (Image par image) ---
        print("Génération des tranches annotées...")
        depth = volume.shape[2]
        
        for i in range(depth):
            # Chercher si des points existent sur la tranche i
            indices = np.where(pts_z == (i + 1))[0] # +1 car le CSV est en index Matlab (1-based)
            # OU: indices = np.where(pts_z == i)[0] si tu as déjà converti Z en 0-based plus haut.
            # Vérifions : pts_z a subi -1 plus haut. Donc on compare à i.
            indices = np.where(pts_z == i)[0]

            if len(indices) > 0:
                # On ne traite que les images qui ont des annotations
                slice_data = volume[:, :, i]
                
                # Normalisation visuelle
                slice_norm = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_color = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2BGR)
                
                for idx in indices:
                    x, y = pts_x[idx], pts_y[idx]
                    cv2.drawMarker(img_color, (int(x), int(y)), (0, 0, 255), 
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                
                nom = os.path.join(dossier_sortie, f"annotated_slice_{i+1:04d}.png")
                cv2.imwrite(nom, img_color)
                print(f"Annotation sauvegardée : {nom}")

if __name__ == "__main__":
    annoter()