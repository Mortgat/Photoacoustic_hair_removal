import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data\dicom\4261_fromdcm.mat"

# METADONNÉES
RESOLUTION_Z_MM = 0.125
PROFONDEUR_ANALYSE_MM = 1.0  # On cherche les poils sur 1mm de profondeur
MARGE_PIXELS = int(PROFONDEUR_ANALYSE_MM / RESOLUTION_Z_MM)

# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def detecter_seuils_auto(volume):
    """Calcule des seuils basés sur les statistiques réelles de l'image."""
    print("   📊 Calcul des statistiques de l'image...")
    mini = volume.min()
    maxi = volume.max()
    moyenne = volume.mean()
    std = volume.std()
    
    # 1. Seuil Surface : On veut être au-dessus du bruit de fond moyen
    # On prend Moyenne + 3 * Ecart-Type (Standard Deviation)
    # C'est une règle statistique classique pour exclure 99% du bruit gaussien
    seuil_surface_auto = moyenne + 4 * std
    
    # 2. Seuil Poil : On veut les trucs TRES brillants (le top 0.2%)
    seuil_poil_auto = np.percentile(volume, 99.8)

    print(f"      - Min: {mini}, Max: {maxi}, Moyenne: {moyenne:.2f}")
    print(f"      - Seuil Surface calculé (Moyenne + 4*STD) : {seuil_surface_auto:.2f}")
    print(f"      - Seuil Poil calculé (Top 0.2%)         : {seuil_poil_auto:.2f}")
    
    return seuil_surface_auto, seuil_poil_auto

def analyse_surface_axe_fixe():
    print(f"--- 🕵️ ANALYSE SURFACE & POILS (AMÉLIORÉE) ---")
    
    if not os.path.exists(fichier_mat):
        print(f"❌ Fichier introuvable : {fichier_mat}")
        return

    mat = scipy.io.loadmat(fichier_mat)
    key = trouver_variable_volume(mat)
    volume = mat[key]
    nx, ny, nz = volume.shape # On suppose format (X, Y, Z) standardisé
    
    # --- ETAPE 0 : SEUILLAGE AUTOMATIQUE ---
    seuil_surface, seuil_poil = detecter_seuils_auto(volume)

    # --- ETAPE 1 : CARTE DE SURFACE ---
    print("🌍 Calcul de la carte de surface...")
    
    # On crée un masque binaire propre
    mask_matiere = volume > seuil_surface
    
    # Argmax pour trouver le premier pixel de matière
    surface_brute = np.argmax(mask_matiere, axis=2)
    
    # CORRECTION DU VIDE :
    # Si argmax renvoie 0, cela peut vouloir dire "Surface en haut" OU "Pas de signal du tout"
    # On vérifie : si la valeur au pixel 0 est inférieure au seuil, c'est du vide.
    # On force ces zones de vide à "NZ" (fond du volume) pour ne pas fausser le lissage
    masque_vide = (surface_brute == 0) & (volume[:, :, 0] < seuil_surface)
    surface_brute[masque_vide] = nz

    # --- ETAPE 2 : LISSAGE ---
    # On utilise un filtre médian pour ignorer les poils isolés et trouver la "peau"
    surface_peau_estimee = median_filter(surface_brute, size=15)

    # --- ETAPE 3 : EXTRACTION P (POILS SURS) ---
    print("✨ Extraction des candidats poils...")

    # Grilles d'indices 3D pour comparer chaque pixel à la surface
    # Z_indices : (1, 1, NZ)
    Z_indices = np.arange(nz).reshape(1, 1, nz)
    
    # Surface : (NX, NY, 1) pour être compatible
    surface_broad = surface_peau_estimee[:, :, np.newaxis]
    
    # Distance verticale : Pixel Z - Surface estimée
    # Négatif = Au-dessus de la peau (Air/Poil sortant)
    # Positif = Dans la peau
    dist_map = Z_indices - surface_broad
    
    # CRITÈRES STRICTS POUR LE TRAINING SET "P"
    # 1. Être très brillant (Seuil Poil Auto)
    cond_lum = volume > seuil_poil
    
    # 2. Position : Être légèrement au-dessus ou pile dans l'entrée de la peau
    # On cherche ce qui "dépasse" de la peau lissée
    # Entre -20 pixels (air) et +MARGE (profondeur peau)
    cond_geo = (dist_map > -20) & (dist_map < MARGE_PIXELS)
    
    masque_poils_learning = cond_lum & cond_geo
    
    nb_pixels = np.sum(masque_poils_learning)
    print(f"👉 Pixels identifiés comme 'Poils Sûrs' : {nb_pixels}")

    if nb_pixels == 0:
        print("⚠️ ALERTE : 0 pixels trouvés. Ton seuil poil est peut-être trop haut ou la surface mal détectée.")

    # ==========================================
    # VISUALISATION
    # ==========================================
    # Coupe au milieu en X
    idx_coupe = nx // 2
    
    # Transposée pour affichage (Z vertical)
    coupe_vol = volume[idx_coupe, :, :].T
    masque_coupe = masque_poils_learning[idx_coupe, :, :].T
    
    ligne_brute = surface_brute[idx_coupe, :]
    ligne_lisse = surface_peau_estimee[idx_coupe, :]

    plt.figure(figsize=(12, 10))
    
    # Plot 1 : Image brute + Lignes Surface
    plt.subplot(3, 1, 1)
    plt.title(f"1. Détection Surface (Seuil Auto: {seuil_surface:.0f})")
    plt.imshow(coupe_vol, cmap='gray', aspect='auto', vmax=np.percentile(coupe_vol, 99))
    plt.plot(ligne_brute, color='red', linewidth=0.8, alpha=0.7, label='Surface Brute')
    plt.plot(ligne_lisse, color='lime', linewidth=2, linestyle='--', label='Peau Estimée')
    plt.legend(loc='upper right')
    plt.ylabel("Z")

    # Plot 2 : Masque P (Noir & Blanc)
    plt.subplot(3, 1, 2)
    plt.title(f"2. Masque P (Seuil Poil: {seuil_poil:.0f})")
    plt.imshow(masque_coupe, cmap='gray', aspect='auto')
    plt.ylabel("Z")
    
    # Plot 3 : Vérification (Superposition) -> Pour voir où il a cherché
    plt.subplot(3, 1, 3)
    plt.title("3. Superposition (Rouge = Poils Sûrs)")
    plt.imshow(coupe_vol, cmap='gray', aspect='auto', vmax=np.percentile(coupe_vol, 99))
    # On superpose le masque en rouge transparent
    # On crée une image RGBA
    overlay = np.zeros((masque_coupe.shape[0], masque_coupe.shape[1], 4))
    overlay[masque_coupe == 1] = [1, 0, 0, 0.8] # Rouge opaque
    plt.imshow(overlay, aspect='auto')
    plt.ylabel("Z")
    plt.xlabel("Y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyse_surface_axe_fixe()