import scipy.io
import tifffile
import os
import numpy as np

# ==========================================
# 1. CONFIGURATION 
# ==========================================

# Le fichier .mat source
fichier_mat = r"Dicom docs\dicom\3851_PA797.mat"

# Le dossier racine où tu veux enregistrer les dossiers de stacks
dossier_sortie_racine = r"Tiff_files\Tiff_stacks"

# Les noms des 3 SOUS-DOSSIERS qui contiendront les stacks.
# L'ordre correspond aux axes [0, 1, 2] de la matrice.
noms_views = [
    "3851_PA797_Stack_View1",      # Coupes le long de l'axe 0
    "3851_PA797_Stack_View2",      # Coupes le long de l'axe 1
    "3851_PA797_Stack_View3"       # Coupes le long de l'axe 2 
]

# ==========================================

def trouver_variable_volume(mat_dict):
    """Fonction intelligente pour trouver la matrice 3D."""
    cles = [k for k in mat_dict.keys() if not k.startswith('__')]
    for cle in cles:
        data = mat_dict[cle]
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return cle
    return 'Data' # Par défaut

def extraire_stacks_multiview():
    print(f"--- 📚 EXTRACTION DE STACKS MULTI-VUES ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier .mat introuvable : {fichier_mat}")
        return

    try:
        # 1. Chargement
        mat = scipy.io.loadmat(fichier_mat)
        nom_var = trouver_variable_volume(mat)
        volume = mat[nom_var]
        
        print(f"✅ Volume chargé : {volume.shape}")
        
        # 2. Boucle sur les 3 axes (0, 1, 2)
        for axis in [0, 1, 2]:
            nom_dossier_vue = noms_views[axis]
            chemin_dossier_vue = os.path.join(dossier_sortie_racine, nom_dossier_vue)
            
            # Création du sous-dossier
            if not os.path.exists(chemin_dossier_vue):
                os.makedirs(chemin_dossier_vue)
            
            # Nombre de tranches dans cet axe
            nb_tranches = volume.shape[axis]
            
            print(f"   👉 Traitement Axe {axis} -> {nb_tranches} images dans '{nom_dossier_vue}'...")
            
            # Boucle pour extraire chaque tranche
            for i in range(nb_tranches):
                # np.take permet de couper le volume selon l'axe dynamique (0, 1 ou 2)
                # C'est l'équivalent de vol[i,:,:] ou vol[:,i,:] ou vol[:,:,i]
                slice_data = np.take(volume, i, axis=axis).astype(np.uint16)
                
                # Nom du fichier : view1_slice_0001.tif
                nom_fichier = f"slice_{i+1:04d}.tif"
                chemin_final = os.path.join(chemin_dossier_vue, nom_fichier)
                
                tifffile.imwrite(chemin_final, slice_data)
                
            print(f"      ✅ Axe {axis} terminé.")

        print("\n🎉 Extraction complète ! Tes données sont triplées.")

    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    extraire_stacks_multiview()