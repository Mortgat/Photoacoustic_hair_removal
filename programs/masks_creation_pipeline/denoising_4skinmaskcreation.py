import numpy as np
import scipy.io
import os
import time
import cupy as cp
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Fichier d'entrée (Ton volume original bruité)
INPUT_FILE = r"dicom_data/dicom/4261_fromdcm.mat"

# Paramètres de Denoising "DOUX" (Validés pour préserver la peau)
NUM_ITER = 40        # Assez pour nettoyer le bruit, pas assez pour effacer la peau
KAPPA = 30           # Sensibilité élevée : préserve les petits contrastes
GAMMA = 0.15         # Vitesse de diffusion standard

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            print(f"✅ Volume trouvé dans la variable : '{k}'")
            return v, k
    raise ValueError("Aucun volume 3D trouvé dans le fichier .mat")

def fft_destriping_gpu(image_gpu):
    """
    Supprime les stries verticales via FFT sur GPU.
    """
    rows, cols = image_gpu.shape
    crow, ccol = rows // 2, cols // 2
    
    # 1. FFT
    f = cp.fft.fft2(image_gpu)
    fshift = cp.fft.fftshift(f)
    
    # 2. Masque (Bande horizontale)
    mask = cp.ones((rows, cols), dtype=cp.float32)
    # On atténue la ligne centrale verticale (+/- 8 pixels)
    mask[crow-8:crow+8, :] = 0.05
    # On protège le centre (basses fréquences / formes globales)
    mask[crow-5:crow+5, ccol-5:ccol+5] = 1.0
    
    # 3. Inverse FFT
    fshift_filtered = fshift * mask
    img_back = cp.fft.ifft2(cp.fft.ifftshift(fshift_filtered))
    
    return cp.abs(img_back)

def anisotropic_diffusion_gpu(img_gpu, n_iter, kappa, gamma):
    """
    Lissage respectant les bords sur GPU.
    """
    img_new = cp.array(img_gpu, dtype=cp.float32)
    
    # Pré-allocation des gradients
    deltaN = cp.zeros_like(img_new)
    deltaS = cp.zeros_like(img_new)
    deltaE = cp.zeros_like(img_new)
    deltaW = cp.zeros_like(img_new)

    for i in range(n_iter):
        # Calcul des gradients
        deltaN[:-1, :] = img_new[1:, :] - img_new[:-1, :]
        deltaS[1:, :]  = img_new[:-1, :] - img_new[1:, :]
        deltaE[:, :-1] = img_new[:, 1:] - img_new[:, :-1]
        deltaW[:, 1:]  = img_new[:, :-1] - img_new[:, 1:]

        # Conduction (Perona-Malik)
        cN = 1.0 / (1.0 + (deltaN / kappa)**2)
        cS = 1.0 / (1.0 + (deltaS / kappa)**2)
        cE = 1.0 / (1.0 + (deltaE / kappa)**2)
        cW = 1.0 / (1.0 + (deltaW / kappa)**2)

        # Update
        img_new += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
        
    return img_new

def main():
    print(f"--- 🚀 DÉMARRAGE DU DÉNOISING GPU (Mode Doux) ---")
    
    # 1. Chargement
    volume_cpu, var_name = load_volume(INPUT_FILE)
    nx, ny, nz = volume_cpu.shape
    print(f"📦 Volume chargé : {nx} x {ny} x {nz}")

    # Volume de sortie vide
    volume_denoised = np.zeros_like(volume_cpu, dtype=np.float32)

    start_time = time.time()

    # 2. Traitement Slice par Slice (pour économiser VRAM)
    for x in tqdm(range(nx), desc="Traitement GPU", unit="slice"):
        # A. CPU -> GPU
        slice_gpu = cp.asarray(volume_cpu[x, :, :], dtype=cp.float32)
        
        # B. Destriping
        slice_destriped = fft_destriping_gpu(slice_gpu)
        
        # C. Diffusion
        slice_smooth = anisotropic_diffusion_gpu(slice_destriped, NUM_ITER, KAPPA, GAMMA)
        
        # D. GPU -> CPU
        volume_denoised[x, :, :] = slice_smooth.get()

    duration = time.time() - start_time
    print(f"\n✅ Terminé en {duration:.1f} secondes.")

    # 3. Sauvegarde
    base, ext = os.path.splitext(INPUT_FILE)
    out_path = f"{base}_denoised_doux.mat"
    print(f"💾 Sauvegarde vers : {out_path} ...")
    scipy.io.savemat(out_path, {var_name: volume_denoised}, do_compression=True)
    print("👋 Fin du script 1.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERREUR : {e}")