import scipy.io
import tifffile
import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

# =================
# 1. CONFIGURATION 
# =================


fichier_mat = "dicom_data/dicom/3851_PA797.mat"

dossier_sortie_racine = "tiff_files/tiff_stacks"

noms_views = [
    "3851_PA797_Stack_View1",      # Axe 0
    "3851_PA797_Stack_View2",      # Axe 1
    "3851_PA797_Stack_View3"       # Axe 2 
]
# =================

def worker_save_slice(data, chemin):
    """Fonction exécutée par les processus travailleurs."""
    try:
        tifffile.imwrite(chemin, data)
        return True
    except Exception as e:
        return f"Erreur sur {chemin}: {e}"

def trouver_variable_volume(mat_dict):
    cles = [k for k in mat_dict.keys() if not k.startswith('__')]
    for cle in cles:
        data = mat_dict[cle]
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return cle
    return 'Data'

def extraire_stacks_opti():
    print(f"--- 🐧 EXTRACTION OPTIMISÉE (LINUX MODE) ---")
    start_time = time.time()
    
    # Vérification stricte du chemin 
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier .mat introuvable : {fichier_mat}")
        print("   👉 Vérifie bien le chemin et la casse (Maj/Min) !")
        print(f"   Dossier actuel : {os.getcwd()}")
        return

    # 1. Chargement
    print("⏳ Chargement du volume en mémoire RAM...")
    try:
        mat = scipy.io.loadmat(fichier_mat)
        nom_var = trouver_variable_volume(mat)
        volume = mat[nom_var]
        print(f"✅ Volume chargé : {volume.shape} | Variable: '{nom_var}'")
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return

    # 2. Préparation et Exécution Parallèle
    # Sur Linux, ProcessPoolExecutor est très rapide grâce au 'forking'
    with ProcessPoolExecutor() as executor:
        futures = []
        total_slices = 0
        
        for axis in [0, 1, 2]:
            nom_dossier = noms_views[axis]
            chemin_dossier = os.path.join(dossier_sortie_racine, nom_dossier)
            
            # Création du dossier (exist_ok=True évite les erreurs si déjà là)
            os.makedirs(chemin_dossier, exist_ok=True)
            
            nb_tranches = volume.shape[axis]
            print(f"   👉 Préparation Axe {axis} ({nb_tranches} images)...")
            
            for i in range(nb_tranches):
                # Extraction et copie
                slice_data = np.take(volume, i, axis=axis).astype(np.uint16).copy()
                
                nom_fichier = f"slice_{i+1:04d}.tif"
                chemin_final = os.path.join(chemin_dossier, nom_fichier)
                
                # Envoi au travailleur
                futures.append(executor.submit(worker_save_slice, slice_data, chemin_final))
                total_slices += 1

        print(f"\n🔥 Sauvegarde de {total_slices} fichiers en parallèle...")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n🎉 Terminé en {duration:.2f} secondes !")
    print(f"   Vitesse : {total_slices/duration:.1f} images/sec.")

if __name__ == "__main__":
    extraire_stacks_opti()