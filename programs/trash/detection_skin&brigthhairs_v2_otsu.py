import numpy as np
import scipy.io
import os
import cupy as cp
import cupyx.scipy.ndimage as ndimage_gpu
import scipy.ndimage as ndimage_cpu  
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
# Fichier d'entrée : Doit correspondre exactement à la sortie du script 1
INPUT_FILE = r"dicom_data/dicom/4261_fromdcm.mat"

# Morphologie Anisotrope (Forme "Soucoupe")
# Z=2 (fin) pour ne pas baver verticalement
# XY=15 (large) pour bien connecter la peau horizontalement
SIZE_Z = 2
SIZE_XY = 15

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            print(f"✅ Volume chargé : {v.shape}")
            return v
    raise ValueError("Aucun volume 3D trouvé.")

def create_mask_hybrid(volume_cpu):
    print("--- 🎭 GÉNÉRATION DU MASQUE (MODE SENSIBLE) ---")
    
    # 1. Transfert et Otsu (GPU)
    vol_min, vol_max = volume_cpu.min(), volume_cpu.max()
    vol_norm = (volume_cpu - vol_min) / (vol_max - vol_min)
    
    print("📊 Calcul du seuil Otsu...")
    thresh = threshold_otsu(vol_norm[::4, ::4, ::4])
    print(f"   -> Seuil Otsu calculé : {thresh:.4f}")
    
    # MODIFICATION 1 : Seuil plus bas pour attraper la peau faible
    # On passe de 0.9 à 0.7
    seuil_force = thresh * 0.7
    print(f"   -> Seuil appliqué (0.7x) : {seuil_force:.4f}")
    
    vol_gpu = cp.asarray(vol_norm, dtype=cp.float32)
    mask_gpu = vol_gpu > seuil_force
    
    # 2. Morphologie sur GPU
    print(f"🛠️ Morphologie GPU (Z={SIZE_Z}, XY={SIZE_XY})...")
    
    struct_cpu = np.zeros((SIZE_Z, SIZE_XY, SIZE_XY), dtype=bool)
    cy, cx = SIZE_XY // 2, SIZE_XY // 2
    y, x = np.ogrid[-cy:SIZE_XY-cy, -cx:SIZE_XY-cx]
    disk = x*x + y*y <= (SIZE_XY//2)**2
    for z in range(SIZE_Z):
        struct_cpu[z] = disk
    struct_gpu = cp.array(struct_cpu)
    
    # MODIFICATION 2 : ON SKIP L'OPENING
    # L'opening (érosion) tuait la peau car elle est trop fine.
    # On passe directement au Closing (Soudure).
    print("   🧱 Closing (Soudure peau) direct sans Opening...")
    mask_connected = ndimage_gpu.binary_closing(mask_gpu, structure=struct_gpu)
    
    # 3. Rapatriement sur CPU
    print("🚚 Transfert vers RAM CPU...")
    mask_cpu_bool = mask_connected.get() 
    
    print("   🗑️ Libération mémoire GPU...")
    del vol_gpu, mask_gpu, mask_connected, struct_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    # 4. Fill Holes & Labeling (CPU)
    print("   🕳️ Fill Holes (CPU)...")
    mask_filled = ndimage_cpu.binary_fill_holes(mask_cpu_bool)

    print("🔍 Isolation de la peau (Largest Component sur CPU)...")
    labeled_array, num_features = ndimage_cpu.label(mask_filled)
    
    if num_features > 0:
        print(f"   📊 Analyse des {num_features} objets...")
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        
        # Sécurité : Si le plus gros objet est trop petit (juste du bruit), on avertit
        max_size = np.max(sizes)
        print(f"      -> Taille du plus gros objet : {max_size} voxels")
        
        max_label = np.argmax(sizes)
        final_mask = (labeled_array == max_label)
    else:
        print("⚠️ ALERTE : Aucun objet trouvé même après closing !")
        final_mask = mask_filled # Sera vide

    return final_mask

def visualize_check(vol, mask):
    # Visualisation rapide d'une tranche au centre
    nx = vol.shape[0]
    idx = nx // 2
    
    img = vol[idx, :, :].T
    msk = mask[idx, :, :].T
    
    # Contour vert pour vérification
    # On utilise une petite érosion pour avoir juste le bord
    erosion = ndimage_cpu.binary_erosion(msk, iterations=1)
    contour = msk ^ erosion

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Image Originale (Denoised)")
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Masque Peau Détecté (Vert)")
    
    # Affichage image de fond
    plt.imshow(img, cmap='gray', aspect='auto', vmax=np.percentile(img, 99))
    
    # Création de l'overlay Vert
    overlay = np.zeros((img.shape[0], img.shape[1], 4))
    
    # Contour Vert Opaque (alpha = 1)
    overlay[contour] = [0, 1, 0, 1]       
    
    # Intérieur Vert Transparent (alpha = 0.15)
    # On affiche l'intérieur sauf le contour
    overlay[msk & ~contour] = [0, 1, 0, 0.15] 
    
    plt.imshow(overlay, aspect='auto')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Charger le volume denoised
    try:
        vol_cpu = load_volume(INPUT_FILE)
    except Exception as e:
        print(f"ERREUR CHARGEMENT: {e}")
        print("💡 Vérifie que tu as bien lancé le Script 1 jusqu'au bout !")
        return
    
    # 2. Créer le masque (Méthode Hybride)
    try:
        mask_cpu = create_mask_hybrid(vol_cpu)
    except MemoryError:
        print("❌ ERREUR MEMOIRE CRITIQUE : Même le CPU a saturé.")
        return
    except Exception as e:
        print(f"❌ ERREUR PENDANT LE MASQUAGE : {e}")
        return
    
    # 3. Sauvegarder
    base, ext = os.path.splitext(INPUT_FILE)
    out_path = f"{base}_skin_mask.mat"
    print(f"💾 Sauvegarde du masque vers : {out_path}")
    
    # On sauvegarde en uint8 pour gagner de la place (0 ou 1)
    scipy.io.savemat(out_path, {'skin_mask': mask_cpu.astype(np.uint8)}, do_compression=True)
    
    # 4. Vérification Visuelle
    print("👀 Affichage du résultat...")
    visualize_check(vol_cpu, mask_cpu)

if __name__ == "__main__":
    main()