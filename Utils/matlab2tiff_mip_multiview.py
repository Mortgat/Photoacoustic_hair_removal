import scipy.io
import numpy as np
import tifffile
import os

# =================
# 1. CONFIGURATION 
# =================

# Le fichier .mat source
fichier_mat = r"dicom_data/dicom/4261_fromdcm.mat"

# Le dossier où tu veux enregistrer les images 
dossier_sortie = r"tiff_files/tiff_mip"

# Les noms précis pour les 3 vues.
# L'ordre correspond aux axes [0, 1, 2] de la matrice Numpy.
noms_views = [
    "4261_fromdcm_MIP_View1.tif",      # Projection Axe 0 
    "4261_fromdcm_MIP_View2.tif",      # Projection Axe 1 
    "4261_fromdcm_MIP_View3.tif"     # Projection Axe 2 
]

# =================

def trouver_variable_volume(mat_dict):
    """Cherche la matrice 3D automatiquement."""
    cles = [k for k in mat_dict.keys() if not k.startswith('__')]
    for cle in cles:
        data = mat_dict[cle]
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return cle
    return 'Data' # Fallback

def generer_mips_custom():
    print(f"--- 📸 GÉNÉRATION MIP MULTI-VUES (Noms Personnalisés) ---")
    
    # 1. Vérifications et Création dossier
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier source introuvable : {fichier_mat}")
        return

    if not os.path.exists(dossier_sortie):
        print(f"📂 Création du dossier de sortie : {dossier_sortie}")
        os.makedirs(dossier_sortie)

    try:
        # 2. Chargement
        mat = scipy.io.loadmat(fichier_mat)
        nom_var = trouver_variable_volume(mat)
        volume = mat[nom_var]
        print(f"✅ Volume chargé depuis '{nom_var}' : {volume.shape}")

        # 3. Boucle sur les 3 axes
        for axis in [0, 1, 2]:
            nom_fichier = noms_views[axis]
            chemin_complet = os.path.join(dossier_sortie, nom_fichier)
            
            print(f"   👉 Traitement Axe {axis} -> {nom_fichier} ...")
            
            # Calcul du MIP
            mip = np.max(volume, axis=axis)
            
            # Sauvegarde
            tifffile.imwrite(chemin_complet, mip)
            print(f"      💾 Sauvegardé. Dimensions image : {mip.shape}")

        print("\n✅ Terminé ! Vérifie tes 3 images dans le dossier de sortie.")

    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    generer_mips_custom()