import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter, gaussian_filter, sobel

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm.mat"

RESOLUTION_Z_MM = 0.125
PROFONDEUR_POIL_MM = 1.0
MARGE_POIL_PIXELS = int(PROFONDEUR_POIL_MM / RESOLUTION_Z_MM)

# ZONE D'EXCLUSION (Le "Trou Noir")
# Pourcentage de l'image au centre qu'on ignore totalement pour le fitting
# 30% à 40% est une bonne valeur si le sommet est très poilu
DEAD_ZONE_RATIO = 0.35 

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v
    raise ValueError("Aucun volume 3D trouvé.")

def get_clean_side_points(slice_2d):
    """
    Récupère les points de peau UNIQUEMENT sur les flancs gauche et droit.
    Ignore totalement le centre bruité.
    """
    rows, cols = slice_2d.shape
    
    # 1. Calcul du Gradient Vertical (Détection de bord)
    # On lisse pour virer le grain, on garde les bords forts
    img_smooth = gaussian_filter(slice_2d, sigma=3)
    grad = sobel(img_smooth, axis=0)
    grad[grad < 0] = 0 # On ne garde que la transition Noir -> Blanc
    
    # 2. Définition des zones (Gauche / Droite / Mort)
    center_y = cols // 2
    dead_width_half = int((cols * DEAD_ZONE_RATIO) / 2)
    
    # Indices valides (Tout sauf le centre)
    valid_indices_left = np.arange(0, center_y - dead_width_half)
    valid_indices_right = np.arange(center_y + dead_width_half, cols)
    
    y_points = []
    z_points = []
    
    # Seuil adaptatif : on prend les pixels les plus forts du gradient
    seuil_bord = np.percentile(grad, 98.5) 

    # 3. Collecte des points sur les flancs
    # On cherche le pic de gradient pour chaque colonne valide
    
    # FLANC GAUCHE
    for y in valid_indices_left:
        col = grad[:, y]
        z_max = np.argmax(col)
        if col[z_max] > seuil_bord:
            y_points.append(y)
            z_points.append(z_max)
            
    # FLANC DROIT
    for y in valid_indices_right:
        col = grad[:, y]
        z_max = np.argmax(col)
        if col[z_max] > seuil_bord:
            y_points.append(y)
            z_points.append(z_max)
            
    return np.array(y_points), np.array(z_points), grad

def fit_robust_bridge(y_points, z_points, img_width):
    """
    Ajuste un polynôme de degré 2 (Parabole) sur les points latéraux.
    Utilise RANSAC maison pour ignorer les points aberrants sur les côtés.
    """
    best_model = None
    best_score = 0
    
    n_points = len(y_points)
    if n_points < 10:
        print("⚠️ Pas assez de points sur les côtés pour faire un pont !")
        return None

    print(f"   🏗️ Construction du pont sur {n_points} points latéraux...")

    # RANSAC simple (1000 itérations)
    for _ in range(1000):
        # On prend 3 points au hasard
        idx = np.random.choice(n_points, 3, replace=False)
        ys = y_points[idx]
        zs = z_points[idx]
        
        try:
            # Fit Parabole : z = ax^2 + bx + c
            coeffs = np.polyfit(ys, zs, 2)
        except:
            continue
            
        # Contrainte de forme : Une peau est une arche (a > 0 dans ce repère image inversé)
        # Si la parabole est dans le mauvais sens ("U" inversé), on jette
        if coeffs[0] <= 1e-6: 
            continue
            
        # Évaluation (Inliers)
        poly = np.poly1d(coeffs)
        predicted_z = poly(y_points)
        error = np.abs(z_points - predicted_z)
        
        # Combien de points sont à moins de 5 pixels de ce modèle ?
        score = np.sum(error < 10)
        
        if score > best_score:
            best_score = score
            best_model = poly

    print(f"   ✅ Pont validé (Score: {best_score}/{n_points})")
    return best_model

def process_suspension_bridge(volume):
    print("--- 🌉 TRACKING : SUSPENSION BRIDGE ---")
    
    nx, ny, nz = volume.shape
    # On prend la coupe centrale pour l'instant
    idx_slice = nx // 2
    slice_2d = volume[idx_slice, :, :].T 
    
    # 1. Récupérer les "Piliers" (Points sur les côtés)
    y_pts, z_pts, grad_debug = get_clean_side_points(slice_2d)
    
    # 2. Construire le "Pont" (Fit polynomial ignorant le centre)
    rows_z, cols_y = slice_2d.shape
    bridge_poly = fit_robust_bridge(y_pts, z_pts, cols_y)
    
    # 3. Générer la courbe complète
    surface_line = np.zeros(cols_y, dtype=int)
    
    if bridge_poly is not None:
        all_y = np.arange(cols_y)
        smooth_z = bridge_poly(all_y)
        surface_line = np.clip(smooth_z, 0, rows_z - 1).astype(int)
    else:
        # Fallback ligne droite
        surface_line[:] = rows_z // 2

    # 4. Masque Final
    mask_2d = np.zeros_like(slice_2d)
    Z_grid = np.arange(rows_z).reshape(-1, 1)
    Surf_grid = surface_line.reshape(1, -1)
    
    dist = Z_grid - Surf_grid
    
    # On prend un masque large car le pont est une estimation
    # On se fie à la géométrie du pont PRIORITAIREMENT
    # Tout ce qui est sous le pont est la peau, tout ce qui est au dessus est poil/air
    
    # On veut détecter les poils : Ce qui brille ET qui est au-dessus du pont
    cond_geo = (dist > -30) & (dist < MARGE_POIL_PIXELS) 
    cond_lum = slice_2d > np.percentile(volume, 99.8)
    
    mask_2d = cond_geo & cond_lum
    
    visualize(slice_2d, mask_2d, surface_line, y_pts, z_pts, cols_y)

def visualize(img, mask, surf, y_pts, z_pts, width):
    plt.figure(figsize=(14, 8))
    
    # Calcul des zones mortes pour l'affichage
    center = width // 2
    dead_w = int(width * DEAD_ZONE_RATIO / 2)
    
    plt.subplot(1, 2, 1)
    plt.title("Méthode du Pont Suspendu")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    
    # Afficher la zone ignorée
    plt.axvspan(center - dead_w, center + dead_w, color='red', alpha=0.1, label="Zone Ignorée (Poils)")
    
    # Afficher les points utilisés (Piliers)
    plt.scatter(y_pts, z_pts, s=1, c='cyan', alpha=0.5, label="Points Piliers")
    
    # Afficher le pont final
    plt.plot(surf, color='lime', linewidth=3, label="Pont Reconstruit")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Masque Final")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
    overlay[mask == 1] = [1, 0, 0, 0.8]
    plt.imshow(overlay, aspect='auto')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    volume = load_volume(fichier_mat)
    process_suspension_bridge(volume)