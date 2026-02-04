import scipy.io
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Mets ici le fichier que tu veux corriger
fichier_cible = r"dicom_data\dicom\3851_PA797.mat" 
# ==========================================

def trouver_variable_volume(mat_dict):
    """Trouve la variable 3D dans le dictionnaire .mat"""
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def standardiser_axe_z(chemin_fichier):
    print(f"--- 🔧 STANDARDISATION DES AXES : {os.path.basename(chemin_fichier)} ---")
    
    if not os.path.exists(chemin_fichier):
        print(f"❌ Fichier introuvable : {chemin_fichier}")
        return

    # 1. Chargement
    try:
        mat_data = scipy.io.loadmat(chemin_fichier)
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")
        return

    nom_variable = trouver_variable_volume(mat_data)
    if nom_variable is None:
        print("❌ Aucune donnée 3D trouvée.")
        return

    volume = mat_data[nom_variable]
    dims_originales = volume.shape
    print(f"   📊 Dimensions actuelles : {dims_originales} (Variable: '{nom_variable}')")

    # 2. Identification de Z (La plus petite dimension)
    idx_z = np.argmin(dims_originales)
    taille_z = dims_originales[idx_z]
    
    print(f"   🧠 Analyse : L'axe Z semble être l'axe n°{idx_z} (Taille : {taille_z})")

    # 3. Vérification et Correction
    # On veut que Z soit à la fin (Axis 2). Donc on veut le format (X, Y, Z) ou (Y, X, Z)
    
    if idx_z == 2:
        print("   ✅ L'axe Z est déjà en position 2. Pas de modification nécessaire.")
        print("   (Aucun nouveau fichier n'a été créé car c'était déjà bon).")
        return
    else:
        print(f"   ⚠️ L'axe Z est mal placé (Position {idx_z}). Correction en cours...")
        
        # Calcul de la nouvelle permutation
        # On garde les autres axes dans l'ordre où ils apparaissent, et on met Z à la fin.
        # Exemple : Si Z est 0 (Z, X, Y), on veut (X, Y, Z) -> Ordre (1, 2, 0)
        nouveaux_axes = [i for i in range(3) if i != idx_z] + [idx_z]
        
        # Transposition (Réorganisation des dimensions sans perdre de données)
        volume_corrige = np.transpose(volume, axes=nouveaux_axes)
        
        print(f"   🔄 Transposition : {dims_originales} ---> {volume_corrige.shape}")

        # 4. Sauvegarde
        # On met à jour la variable dans le dictionnaire
        mat_data[nom_variable] = volume_corrige
        
        # Création du nouveau nom
        base, ext = os.path.splitext(chemin_fichier)
        nouveau_chemin = f"{base}_axisfixed{ext}"
        
        # Sauvegarde compressée (utile pour les gros fichiers) ou standard
        scipy.io.savemat(nouveau_chemin, mat_data, do_compression=True)
        
        print(f"   💾 Sauvegardé sous : {os.path.basename(nouveau_chemin)}")
        print("   🎉 Fichier prêt et standardisé au format (X, Y, Z).")

if __name__ == "__main__":
    standardiser_axe_z(fichier_cible)