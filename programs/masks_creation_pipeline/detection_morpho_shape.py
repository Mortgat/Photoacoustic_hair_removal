import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing, binary_dilation

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# CRITÈRES DE FORME
MIN_AREA = 20           # Taille minimale en pixels (vire la poussière)
MIN_ECCENTRICITY = 0.75 # 0=Rond, 1=Ligne. On veut des trucs allongés (>0.75)

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v
    raise ValueError("Aucun volume 3D trouvé.")

def filter_by_shape(volume):
    print("--- 📐 FILTRAGE GÉOMÉTRIQUE (SHAPE ANALYSIS) ---")
    
    nx, ny, nz = volume.shape
    idx_slice = nx // 2
    slice_2d = volume[idx_slice, :, :].T 
    
    # 1. BASE : Otsu 3 classes (pour avoir le masque bruité de départ)
    thresholds = threshold_multiotsu(slice_2d, classes=3)
    mask_otsu = (slice_2d > thresholds[0]) & (slice_2d < thresholds[1])
    
    # 2. ASTUCE : DILATATION HORIZONTALE
    # Avant d'analyser les formes, on force les tirets de la peau à se connecter
    # On utilise un noyau rectangle plat : [1, 10]
    kernel_h = np.ones((1, 10)) 
    mask_connected = binary_closing(mask_otsu, structure=kernel_h)
    
    # 3. ANALYSE DES OBJETS
    labeled_img = label(mask_connected)
    regions = regionprops(labeled_img)
    
    mask_skin_only = np.zeros_like(mask_otsu)
    mask_rejected = np.zeros_like(mask_otsu)
    
    print(f"   🔍 Analyse de {len(regions)} objets...")
    
    count_kept = 0
    for prop in regions:
        # Critère 1 : Taille
        if prop.area < MIN_AREA:
            mask_rejected[labeled_img == prop.label] = 1
            continue
            
        # Critère 2 : Forme (Eccentricité)
        # La peau est une ligne, donc eccentricité proche de 1.
        # Les vaisseaux sont des taches, donc eccentricité plus basse.
        if prop.eccentricity < MIN_ECCENTRICITY:
            mask_rejected[labeled_img == prop.label] = 1
            continue
            
        # Si ça passe, on garde !
        mask_skin_only[labeled_img == prop.label] = 1
        count_kept += 1
        
    print(f"   ✅ {count_kept} segments de peau conservés.")

    visualize_shape_filter(slice_2d, mask_otsu, mask_connected, mask_skin_only, mask_rejected)

def visualize_shape_filter(original, m_otsu, m_conn, m_skin, m_rej):
    plt.figure(figsize=(16, 10))
    
    # 1. Masque Otsu Original (Le bazar)
    plt.subplot(2, 2, 1)
    plt.title("1. Masque Otsu (Bruit + Peau morcelée)")
    plt.imshow(m_otsu, cmap='gray', aspect='auto')
    
    # 2. Après Closing Horizontal
    plt.subplot(2, 2, 2)
    plt.title("2. Closing Horizontal (Soudure des tirets)")
    plt.imshow(m_conn, cmap='gray', aspect='auto')
    
    # 3. Visualisation du tri
    plt.subplot(2, 2, 3)
    plt.title("3. Tri Géométrique (Vert=Gardé, Rouge=Rejeté)")
    base_rgb = np.zeros((original.shape[0], original.shape[1], 3))
    # Fond gris pour voir
    base_rgb[:, :, 0] = original / original.max()
    base_rgb[:, :, 1] = original / original.max()
    base_rgb[:, :, 2] = original / original.max()
    
    # Overlay Vert (Peau)
    base_rgb[m_skin == 1] = [0, 1, 0]
    # Overlay Rouge (Vaisseaux/Bruit)
    base_rgb[m_rej == 1] = [1, 0, 0]
    
    plt.imshow(base_rgb, aspect='auto')
    
    # 4. Résultat Final
    plt.subplot(2, 2, 4)
    plt.title("4. Masque Final (Peau Uniquement)")
    plt.imshow(original, cmap='gray', aspect='auto', vmax=np.percentile(original, 99))
    overlay = np.zeros((original.shape[0], original.shape[1], 4))
    overlay[m_skin == 1] = [0, 1, 0, 0.6]
    plt.imshow(overlay, aspect='auto')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    volume = load_volume(fichier_mat)
    filter_by_shape(volume)