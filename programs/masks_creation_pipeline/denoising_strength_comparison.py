import numpy as np
import scipy.io
import cupy as cp
import matplotlib.pyplot as plt
import os

# ==========================================
# FICHIER À TESTER
# ==========================================
input_file = r"dicom_data/dicom/4261_fromdcm.mat"

# ==========================================
# FONCTIONS GPU (Celles qui marchent)
# ==========================================

def fft_destriping_gpu(image_gpu):
    rows, cols = image_gpu.shape
    crow, ccol = rows // 2, cols // 2
    f = cp.fft.fft2(image_gpu)
    fshift = cp.fft.fftshift(f)
    mask = cp.ones((rows, cols), dtype=cp.float32)
    
    # On garde le filtre large (8px) car il marchait bien pour les stries
    mask[crow - 8 : crow + 8, :] = 0.05
    mask[crow - 5 : crow + 5, ccol - 5 : ccol + 5] = 1.0
    
    fshift_filtered = fshift * mask
    img_back = cp.fft.ifft2(cp.fft.ifftshift(fshift_filtered))
    return cp.abs(img_back)

def anisotropic_diffusion_gpu(img_gpu, n_iter, kappa, gamma):
    img_new = cp.array(img_gpu, dtype=cp.float32)
    deltaN = cp.zeros_like(img_new)
    deltaS = cp.zeros_like(img_new)
    deltaE = cp.zeros_like(img_new)
    deltaW = cp.zeros_like(img_new)

    for i in range(n_iter):
        deltaN[:-1, :] = img_new[1:, :] - img_new[:-1, :]
        deltaS[1:, :]  = img_new[:-1, :] - img_new[1:, :]
        deltaE[:, :-1] = img_new[:, 1:] - img_new[:, :-1]
        deltaW[:, 1:]  = img_new[:, :-1] - img_new[:, 1:]

        cN = 1.0 / (1.0 + (deltaN / kappa)**2)
        cS = 1.0 / (1.0 + (deltaS / kappa)**2)
        cE = 1.0 / (1.0 + (deltaE / kappa)**2)
        cW = 1.0 / (1.0 + (deltaW / kappa)**2)

        img_new += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)
        
    return img_new

# ==========================================
# TEST COMPARATIF
# ==========================================

def run_test():
    # 1. Chargement
    mat = scipy.io.loadmat(input_file)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            volume = v
            break
    
    # On prend une tranche au milieu
    idx = volume.shape[0] // 2
    slice_cpu = volume[idx, :, :]
    
    # Transfert GPU
    slice_gpu = cp.asarray(slice_cpu, dtype=cp.float32)
    
    # Pré-traitement FFT (commun à tous)
    slice_destriped = fft_destriping_gpu(slice_gpu)

    # --- 3 SCÉNARIOS ---
    
    # SCÉNARIO 1 : "Doux" (Préserve les détails)
    # Kappa faible = on arrête de lisser dès qu'on touche un petit bord
    res1 = anisotropic_diffusion_gpu(slice_destriped, n_iter=40, kappa=30, gamma=0.15)
    
    # SCÉNARIO 2 : "Équilibré" (Compromis)
    # Kappa moyen
    res2 = anisotropic_diffusion_gpu(slice_destriped, n_iter=60, kappa=50, gamma=0.15)
    
    # SCÉNARIO 3 : "Fort" (Mais moins fou que le précédent)
    # Kappa un peu plus haut, mais iter raisonnable
    res3 = anisotropic_diffusion_gpu(slice_destriped, n_iter=100, kappa=60, gamma=0.15)

    # Récupération CPU
    img_orig = slice_cpu.T
    img_1 = res1.get().T
    img_2 = res2.get().T
    img_3 = res3.get().T

    # --- VISUALISATION ---
    plt.figure(figsize=(18, 6))
    
    vmax = np.percentile(img_orig, 99)

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img_orig, cmap='gray', vmax=vmax, aspect='auto')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("1. DOUX\nIter=40, Kappa=30")
    plt.imshow(img_1, cmap='gray', vmax=vmax, aspect='auto')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("2. ÉQUILIBRÉ\nIter=60, Kappa=50")
    plt.imshow(img_2, cmap='gray', vmax=vmax, aspect='auto')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("3. FORT\nIter=100, Kappa=60")
    plt.imshow(img_3, cmap='gray', vmax=vmax, aspect='auto')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()