import os
import numpy as np
import pydicom
import scipy.io

# ==========================================
# CONFIGURATION
# ==========================================
# Liste des fichiers DICOM (Cubes) à convertir
fichiers_a_convertir = [
    r"../dicom_data/dicom/3888_20180222_084017227_Volume2.dcm",
    r"dicom_data/dicom/4261_20180326_112555703_Volume2.dcm"
]

# Dossier où on va poser les .mat propres
dossier_sortie = r"dicom_data/dicom"

def convertir_final(chemin_dicom):
    nom_fichier = os.path.basename(chemin_dicom)
    print(f"\n--- 🚀 TRAITEMENT : {nom_fichier} ---")
    
    if not os.path.exists(chemin_dicom):
        print("❌ Fichier introuvable.")
        return

    try:
        # 1. Lecture du fichier
        ds = pydicom.dcmread(chemin_dicom)
        
        # 2. Extraction du Volume (Données Images)
        # Note : Pydicom lit souvent en (Z, Y, X) ou (Frames, Rows, Cols)
        data = ds.pixel_array
        print(f"   📊 Dimensions brutes (Pydicom) : {data.shape}")
        
        # Réorientation pour MATLAB (souvent H, W, D)
        # Si la forme est (N_frames, Height, Width), on veut (Height, Width, N_frames)
        if data.ndim == 3:
            volume = np.moveaxis(data, 0, -1).astype(np.uint16)
            print(f"   🔄 Réorienté pour Matlab (Y, X, Z) : {volume.shape}")
        else:
            print("   ⚠️ Attention : Ce n'est pas un volume 3D standard.")
            volume = data
            
        # 3. Récupération de la résolution X/Y
        resolution = [1.0, 1.0, 1.0] # [ResX, ResY, ResZ]
        ps = ds.get('PixelSpacing', None)
        
        if ps:
            res_x = float(ps[1]) # Colonnes
            res_y = float(ps[0]) # Lignes
            resolution[0] = res_x
            resolution[1] = res_y
            print(f"   📏 Résolution X/Y trouvée : {res_x} x {res_y} mm")
            
            # --- APPLICATION DE LA LOGIQUE CANON IMPACT ---
            # Sur cette machine, les reconstructions sont isotropes (Cube)
            # Si Z est manquant, on applique Z = X
            resolution[2] = res_x 
            print(f"   ✅ Z manquant -> Forcé à {resolution[2]} mm (Hypothèse Isotrope Canon ImPACT)")
            
        else:
            print("   ❌ ERREUR CRITIQUE : Pas de PixelSpacing X/Y dans le fichier.")
            # On laisse 1.0 par défaut ou on arrête

        # 4. Sauvegarde
        nom_sortie = nom_fichier.replace('.dcm', '_fixed.mat')
        chemin_sortie = os.path.join(dossier_sortie, nom_sortie)
        
        print(f"   💾 Sauvegarde vers : {nom_sortie} ...")
        
        scipy.io.savemat(chemin_sortie, {
            'Data': volume,         # Le cube d'image
            'Resolution': resolution, # [0.125, 0.125, 0.125]
            'Machine': 'Canon ImPACT-WF'
        })
        print("   ✅ Terminé.")

    except Exception as e:
        print(f"   ❌ Erreur : {e}")

if __name__ == "__main__":
    for f in fichiers_a_convertir:
        convertir_final(f)