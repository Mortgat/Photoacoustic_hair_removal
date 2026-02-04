import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter, gaussian_filter, laplace
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data\dicom\4261_fromdcm_axisfixed.mat"

# Paramètres (Ceux qui marchent bien chez toi)
RESOLUTION_Z_MM = 0.125
PROFONDEUR_ANALYSE_MM = 1.0 
MARGE_PIXELS = int(PROFONDEUR_ANALYSE_MM / RESOLUTION_Z_MM)

# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def extraire_features(volume):
    """
    Transforme l'image brute en un jeu de données riche pour l'IA.
    Pour chaque pixel, on calcule : Intensité, Flou (Gauss), Bords (Laplace).
    """
    print("   🧠 Création des 'Features' (Texture, Bords)...")
    
    # 1. Intensité Brute (Normalisée)
    feat_intensity = volume.astype(np.float32)
    feat_intensity /= feat_intensity.max()
    
    # 2. Texture : Flou Gaussien (Donne le contexte local)
    feat_gauss = gaussian_filter(feat_intensity, sigma=1)
    
    # 3. Bords : Laplacien (Détecte les changements brusques -> contours poils)
    feat_edge = laplace(feat_intensity)
    
    # On empile tout ça : (N_pixels, 3 features)
    # On aplatit les volumes en colonnes
    X = np.stack([
        feat_intensity.flatten(),
        feat_gauss.flatten(),
        feat_edge.flatten()
    ], axis=1)
    
    return X

def pu_learning_pipeline():
    print(f"--- 🤖 PU LEARNING : DÉTECTION INTELLIGENTE ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier introuvable.")
        return

    # --- 1. CHARGEMENT & DETECTIONS BASE (Ton code précédent) ---
    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume = mat[key]
    nx, ny, nz = volume.shape
    
    # Seuils Auto
    mean, std = volume.mean(), volume.std()
    seuil_surface = mean + 5 * std # Un peu plus strict pour le bruit
    seuil_poil = np.percentile(volume, 99.5) # Un peu plus large pour avoir plus d'exemples
    
    # Surface
    mask_matiere = volume > seuil_surface
    surface_brute = np.argmax(mask_matiere, axis=2)
    masque_vide = (surface_brute == 0) & (volume[:, :, 0] < seuil_surface)
    surface_brute[masque_vide] = nz
    surface_peau = median_filter(surface_brute, size=20)
    
    # Masque P (Positifs Sûrs)
    Z_indices = np.arange(nz).reshape(1, 1, nz)
    dist_map = Z_indices - surface_peau[:, :, np.newaxis]
    
    # On prend les poils qui sortent (-20 à +MARGE) et qui brillent
    masque_P = (volume > seuil_poil) & (dist_map > -20) & (dist_map < MARGE_PIXELS)
    
    pixels_P_count = np.sum(masque_P)
    print(f"✅ Échantillons Positifs (P) trouvés : {pixels_P_count}")
    
    if pixels_P_count < 100:
        print("❌ Pas assez de points P pour apprendre. Baisse tes seuils.")
        return

    # --- 2. PRÉPARATION DES DONNÉES POUR L'IA ---
    # On extrait les features de TOUTE l'image
    # Attention : Pour aller vite, on va entraîner sur une COUPE (Slice) représentative
    # Sinon avec 256*1000*1000 pixels, ta RAM va exploser.
    
    idx_coupe = nx // 2
    print(f"⚙️ Entraînement sur la coupe centrale X={idx_coupe}...")
    
    slice_vol = volume[idx_coupe, :, :] # (NY, NZ)
    slice_mask_P = masque_P[idx_coupe, :, :]
    
    # Extraction features 2D pour cette coupe
    feat_int = slice_vol.astype(np.float32) / slice_vol.max()
    feat_g = gaussian_filter(feat_int, sigma=1)
    feat_e = laplace(feat_int)
    
    X_slice = np.stack([feat_int.flatten(), feat_g.flatten(), feat_e.flatten()], axis=1)
    y_slice = slice_mask_P.flatten() # 1 = P, 0 = Unlabeled (U)
    
    # --- 3. BAGGING PU (L'Astuce) ---
    # On ne sait pas ce qui est Négatif. Donc on suppose que U contient des Négatifs + des Positifs cachés.
    # On entraîne plusieurs modèles en prenant :
    # - Tous les P
    # - Un échantillon aléatoire de U (considéré comme Négatif temporairement)
    
    n_estimators = 5 # Nombre de modèles (augmente à 10 ou 20 si rapide)
    final_proba = np.zeros(len(y_slice))
    
    # Indices des Positifs et des Unlabeled
    indices_P = np.where(y_slice == 1)[0]
    indices_U = np.where(y_slice == 0)[0]
    
    print(f"🚀 Démarrage du Bagging ({n_estimators} modèles)...")
    
    for i in range(n_estimators):
        # On prend autant de U que de P pour équilibrer
        sample_U = np.random.choice(indices_U, size=len(indices_P), replace=False)
        
        # Dataset temporaire équilibré
        train_idx = np.concatenate([indices_P, sample_U])
        X_train = X_slice[train_idx]
        y_train = np.concatenate([np.ones(len(indices_P)), np.zeros(len(sample_U))])
        
        # Classifieur rapide
        clf = RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Prédiction sur TOUT l'image (proba d'être un poil)
        # On accumule les probabilités
        final_proba += clf.predict_proba(X_slice)[:, 1]

    # Moyenne des votes
    final_proba /= n_estimators
    
    # On remet en forme d'image 2D
    proba_map = final_proba.reshape(slice_vol.shape).T # Transposé pour affichage vertical
    
    # --- 4. VISUALISATION DES RÉSULTATS ---
    # Seuil de décision : Si proba > 0.5 (ou plus strict), c'est un poil
    masque_final = proba_map > 0.6
    
    plt.figure(figsize=(15, 6))
    
    # Image originale
    plt.subplot(1, 3, 1)
    plt.title("Image Originale")
    plt.imshow(slice_vol.T, cmap='gray', aspect='auto', vmax=np.percentile(slice_vol, 99))
    
    # Carte de Probabilité (Ce que l'IA "voit")
    plt.subplot(1, 3, 2)
    plt.title("Probabilité IA (PU Learning)")
    plt.imshow(proba_map, cmap='inferno', aspect='auto')
    plt.colorbar(label="Probabilité d'être un poil")
    
    # Masque Final
    plt.subplot(1, 3, 3)
    plt.title("Masque Final (Proba > 0.6)")
    plt.imshow(masque_final, cmap='gray', aspect='auto')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pu_learning_pipeline()