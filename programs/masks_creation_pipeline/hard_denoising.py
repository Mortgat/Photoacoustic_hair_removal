import os
import numpy as np
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_BRUT = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_HARD_OUT = r"dicom_data/dicom/4261_hard_96.npz"

NLM_H_MULT = 5.0
PERCENTILE_CUT = 96.0

def main():
    if not os.path.exists(FICHIER_BRUT):
        print(f"Erreur : Le fichier {FICHIER_BRUT} est introuvable.")
        return

    print("Chargement du volume brut...")
    vol_brut = np.load(FICHIER_BRUT)["volume"].astype(np.float32)
    nx, ny, nz = vol_brut.shape
    
    vol_hard = np.zeros_like(vol_brut)

    print(f"Début du traitement NL-Means + Coupure {PERCENTILE_CUT}% ({ny} tranches)...")
    
    for y in tqdm(range(ny), desc="Génération Volume Hard"):
        slice_2d = vol_brut[:, y, :].T  # Shape: (nz, nx)
        
        try:
            sigma_est = np.mean(estimate_sigma(slice_2d))
        except:
            sigma_est = np.std(slice_2d) * 0.1
            
        # 1. Débruitage NL-Means
        slice_nlm = denoise_nl_means(
            slice_2d, 
            h=NLM_H_MULT * sigma_est, 
            fast_mode=True, 
            patch_size=5, 
            patch_distance=6
        )
        
        # 2. Coupure Nette (Percentile)
        valeur_coupe = np.percentile(slice_nlm, PERCENTILE_CUT)
        slice_cut = np.copy(slice_nlm)
        slice_cut[slice_cut < valeur_coupe] = valeur_coupe
        
        # 3. Réintégration dans le volume
        vol_hard[:, y, :] = slice_cut.T

    print(f"Sauvegarde du volume hard vers {FICHIER_HARD_OUT}...")
    np.savez_compressed(FICHIER_HARD_OUT, volume=vol_hard)
    print("Terminé.")

if __name__ == "__main__":
    main()