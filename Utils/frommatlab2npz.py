import scipy.io
import numpy as np
import os

def mat_to_npz(chemin_mat):
    if not os.path.exists(chemin_mat):
        print(f"Erreur : Le fichier {chemin_mat} est introuvable.")
        return

    print(f"Chargement de {chemin_mat}...")
    mat = scipy.io.loadmat(chemin_mat)
    
    # Trouver automatiquement la clé correspondant au volume 3D
    key = None
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            key = k
            break
            
    if key is None:
        print("Erreur : Aucun volume 3D trouvé dans le fichier .mat.")
        return
        
    volume = mat[key]
    
    # Création du chemin de sortie (remplace .mat par .npz)
    chemin_sortie = os.path.splitext(chemin_mat)[0] + ".npz"
    
    print(f"Sauvegarde en cours vers {chemin_sortie}...")
    # Compression sans perte. Si c'est trop lent, utilise np.savez() à la place.
    np.savez_compressed(chemin_sortie, volume=volume)
    print("Conversion terminée avec succès.")

if __name__ == "__main__":
    # Tu peux lancer la conversion pour tes deux fichiers
    mat_to_npz(r"dicom_data/dicom/4261_fromdcm.mat")
    mat_to_npz(r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat")