import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import time

# Tente d'importer CuPy pour le GPU
try:
    import cupy as cp
    import cupyx.scipy.fft as cpfft
    HAS_GPU = True
    print("🚀 GPU NVIDIA détecté : Mode Turbo activé avec CuPy.")
except ImportError:
    import numpy as cp
    import scipy.fft as cpfft # Fallback CPU si pas de GPU
    HAS_GPU = False
    print("⚠️ GPU non détecté (CuPy absent). Mode CPU lent activé.")

# ==========================================
# CONFIGURATION
# ==========================================
input_file = r"dicom_data\dicom\4261_fromdcm_axisfixed.mat"

# Paramètres
NUM_ITER = 30        
KAPPA = 50           
GAMMA = 0.15         

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return v, k
    raise ValueError("Aucun volume 3D trouvé.")

def fft_destriping_gpu(image_gpu):
    """
    Version GPU du destriping.
    Tout se passe dans la mémoire vidéo (VRAM).
    """
    rows, cols = image_gpu.shape
    crow, ccol = rows // 2, cols // 2
    
    # 1. FFT sur GPU
    f = cp.fft.fft2(image_gpu)
    fshift = cp.fft.fftshift(f)
    
    # 2. Masque (Créé directement sur GPU)
    mask = cp.ones((rows, cols), dtype=cp.float32)
    
    # Suppression fréquences verticales (Ligne horizontale centrale)
    # On laisse passer le centre (DC) +/- 5 pixels
    mask[crow-2:crow+2, :] = 0.1 
    mask[crow-5:crow+5, ccol-5:ccol+5] = 1.0
    
    # 3. Application
    fshift_filtered = fshift * mask
    f_ishift = cp.fft.ifftshift(fshift_filtered)
    img_back = cp.fft.ifft2(f_ishift)
    
    return cp.abs(img_back)

def anisotropic_diffusion_gpu(img_gpu, n_iter, kappa, gamma):
    """
    Version GPU de la diffusion.
    Les opérations matricielles sont parallélisées sur les milliers de coeurs CUDA.
    """
    # Copie de travail
    img_new = cp.array(img_gpu, dtype=cp.float32)
    
    for i in range(n_iter):
        # Calcul des gradients (Slicing sur GPU est très rapide)
        # On utilise des vues, pas de copies mémoire inutiles
        deltaN = cp.zeros_like(img_new)
        deltaS = cp.zeros_like(img_new)
        deltaE = cp.zeros_like(img_new)
        deltaW = cp.zeros_like(img_new)

        deltaN[:-1, :] = img_new[1:, :] - img_new[:-1, :]
        deltaS[1:, :]  = img_new[:-1, :] - img_new[1:, :]
        deltaE[:, :-1] = img_new[:, 1:] - img_new[:, :-1]
        deltaW[:, 1:]  = img_new[:, :-1] - img_new[:, 1:]

        # Conduction
        cN = 1.0 / (1.0 + (deltaN / kappa)**2)
        cS = 1.0 / (1.0 + (deltaS / kappa)**2)
        cE = 1.0 / (1.0 + (deltaE / kappa)**2)
        cW = 1.0 / (1.0 + (deltaW / kappa)**2)

        # Mise à jour
        img_new += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)
        
    return img_new

def process_denoising():
    print("--- ⚡ DÉNOISING GPU (CUPY) ---")
    
    # 1. Chargement (CPU -> RAM)
    volume_cpu, var_name = load_volume(input_file)
    nx, ny, nz = volume_cpu.shape
    print(f"✅ Volume chargé RAM : {volume_cpu.shape}")

    # Volume de sortie (RAM)
    volume_denoised_cpu = np.zeros_like(volume_cpu, dtype=np.float32)

    start_time = time.time()

    # 2. Traitement par Batch ou Slice
    # On envoie tranche par tranche sur le GPU pour ne pas saturer la VRAM
    for x in range(nx):
        # A. Transfert RAM -> VRAM (GPU)
        slice_gpu = cp.asarray(volume_cpu[x, :, :], dtype=cp.float32)
        
        # B. Traitement GPU
        slice_destriped = fft_destriping_gpu(slice_gpu)
        slice_smooth = anisotropic_diffusion_gpu(slice_destriped, NUM_ITER, KAPPA, GAMMA)
        
        # C. Retour VRAM -> RAM (CPU)
        # .get() ou cp.asnumpy() récupère le résultat
        if HAS_GPU:
            volume_denoised_cpu[x, :, :] = slice_smooth.get()
        else:
            volume_denoised_cpu[x, :, :] = slice_smooth

        if x % 10 == 0:
            print(f"Traitement GPU tranche {x}/{nx}...", end='\r')

    end_time = time.time()
    print(f"\n⏱️ Temps total : {end_time - start_time:.2f} secondes")

    # 3. Sauvegarde
    base, ext = os.path.splitext(input_file)
    out_path = f"{base}_denoised_gpu.mat"
    scipy.io.savemat(out_path, {var_name: volume_denoised_cpu}, do_compression=True)
    print(f"💾 Sauvegardé sous : {out_path}")

    # 4. Visualisation
    idx = nx // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(volume_cpu[idx, :, :].T, cmap='gray', aspect='auto', vmax=np.percentile(volume_cpu[idx], 99))
    plt.subplot(1, 2, 2)
    plt.title("Denoised GPU")
    plt.imshow(volume_denoised_cpu[idx, :, :].T, cmap='gray', aspect='auto', vmax=np.percentile(volume_denoised_cpu[idx], 99))
    plt.show()

if __name__ == "__main__":
    process_denoising()