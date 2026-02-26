import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import rotate

# Gestion du GPU
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter 
    GPU_ACTIF = True
except ImportError:
    from scipy.ndimage import median_filter
    GPU_ACTIF = False

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_MAT = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"
FICHIER_METADATA = "transformation_metadata.json"
FICHIER_SAM2 = "masque_sam2_v4.npz"

# --- PARAMÈTRES GÉOMÉTRIQUES & CLINIQUES ---
MULTIPLICATEUR_SEUIL_BAS = 2.0
PERCENTILE_COUPURE_HAUTE = 93.0
TAILLE_FILTRE_SURFACE = 30
PERCENTILE_REJET_EXTREMES = 1.0  

RESOLUTION_Z_MM = 0.12500
EPAISSEUR_MAX_PEAU_MM = 1.5
# Tolérance en vrais voxels physiques
TOLERANCE_PIXELS_BRUTS = EPAISSEUR_MAX_PEAU_MM / RESOLUTION_Z_MM 
# ==========================================

def trouver_variable_volume(mat_dict):
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def evaluer_modele_rawdata():
    print("--- 📊 ÉVALUATION CLINIQUE ABSOLUE (SUR RAW DATA) ---")
    
    if not os.path.exists(FICHIER_MAT) or not os.path.exists(FICHIER_SAM2):
        print("❌ Erreur : Fichier .mat ou .npz introuvable.")
        return

    # 1. Chargement des métadonnées
    with open(FICHIER_METADATA, 'r') as f: meta = json.load(f)

    # 2. Chargement et préparation du volume brut
    print(f"⏳ Chargement du volume .mat ({FICHIER_MAT})...")
    mat = scipy.io.loadmat(FICHIER_MAT)
    volume_brut = mat[trouver_variable_volume(mat)]

    angle = meta["angle_rotation_applique"]
    if abs(angle) > 0:
        volume_brut = rotate(volume_brut, angle, axes=(0, 1), reshape=True, order=1)
        
    x_min_rogne, x_max_rogne = meta["rognage_x"]["min"], meta["rognage_x"]["max"]
    volume = volume_brut[x_min_rogne:x_max_rogne, :, :]
    nx, ny, nz = volume.shape

    # 3. Extraction de la Vérité Terrain (Calcul Hybride CPU/GPU)
    print("🕵️ Calcul de la méthode géométrique pure (Hybride CPU/GPU)...")
    
    vol_cpu = np.asarray(volume)
    seuil_bas = vol_cpu.mean() + MULTIPLICATEUR_SEUIL_BAS * vol_cpu.std()
    
    masque_brillant = (vol_cpu > seuil_bas)
    pixels_utiles = vol_cpu[masque_brillant]
    seuil_haut = np.percentile(pixels_utiles, PERCENTILE_COUPURE_HAUTE) if pixels_utiles.size > 0 else np.max(vol_cpu)
    
    masque_brillant &= (vol_cpu < seuil_haut)
    surface_brute_z = np.argmax(masque_brillant, axis=2)
    
    colonnes_vides = np.max(vol_cpu, axis=2) < seuil_bas
    surface_brute_z[colonnes_vides] = nz 
    
    if GPU_ACTIF:
        surface_brute_gpu = cp.asarray(surface_brute_z)
        surface_lisse_gpu = median_filter(surface_brute_gpu, size=TAILLE_FILTRE_SURFACE)
        surface_lisse_z = surface_lisse_gpu.get()
    else:
        surface_lisse_z = median_filter(surface_brute_z, size=TAILLE_FILTRE_SURFACE)

    surface_geo = surface_lisse_z.astype(float)
    surface_geo[colonnes_vides] = -1.0 # Marqueur de vide

    # 4. Chargement de SAM 2
    print("⏳ Chargement du masque SAM 2...")
    sam2_masques = np.load(FICHIER_SAM2)["masque"]

    # --- MÉCANIQUE DE PROJECTION ---
    pad_top = meta["format_sam2"]["padding"]["top"]
    pad_left = meta["format_sam2"]["padding"]["left"]
    ratio_scale = 1024.0 / max(meta["format_sam2"]["shape_rognee"]["largeur_w"], meta["format_sam2"]["shape_rognee"]["hauteur_h"])

    projeter_x = lambda val_brut: int((val_brut + pad_left) * ratio_scale)
    inverser_z = lambda val_1024: (val_1024 / ratio_scale) - pad_top

    # Variables de stockage pour les stats
    toutes_les_erreurs_brutes = []
    frames_valides, mae_liste, std_liste, max_liste = [], [], [], []
    total_pixels_purs_utiles, total_pixels_predits = 0, 0

    print(f"⚙️ Évaluation spatiale (Exclusion de {PERCENTILE_REJET_EXTREMES}% aux extrémités)...")
    
    for y in range(ny):
        x_purs = np.where(surface_geo[:, y] != -1.0)[0]
        
        if len(x_purs) == 0:
            continue
            
        x_min_utile = np.percentile(x_purs, PERCENTILE_REJET_EXTREMES)
        x_max_utile = np.percentile(x_purs, 100 - PERCENTILE_REJET_EXTREMES)
        x_utiles = x_purs[(x_purs >= x_min_utile) & (x_purs <= x_max_utile)]
        
        total_pixels_purs_utiles += len(x_utiles)
        
        masque_frame = sam2_masques[y]
        colonnes_sam_valides = np.any(masque_frame, axis=0)
        z_1024_toit = np.argmax(masque_frame, axis=0)
        
        erreurs_frame = []
        
        for x_brut in x_utiles:
            x_1024 = projeter_x(x_brut)
            
            if 0 <= x_1024 < 1024 and colonnes_sam_valides[x_1024]:
                z_1024 = z_1024_toit[x_1024]
                z_sam_brut = inverser_z(z_1024)
                z_geo_brut = surface_geo[x_brut, y]
                
                erreur_absolue = abs(z_geo_brut - z_sam_brut)
                erreurs_frame.append(erreur_absolue)
                total_pixels_predits += 1

        if len(erreurs_frame) > 0:
            toutes_les_erreurs_brutes.extend(erreurs_frame)
            frames_valides.append(y)
            mae_liste.append(np.mean(erreurs_frame))
            std_liste.append(np.std(erreurs_frame))
            max_liste.append(np.max(erreurs_frame))
        else:
            frames_valides.append(y)
            mae_liste.append(np.nan)
            std_liste.append(np.nan)
            max_liste.append(np.nan)

    # --- CALCULS GLOBAUX ---
    erreurs_globales = np.array(toutes_les_erreurs_brutes)
    
    print("\n" + "="*65)
    print(" 🏆 RAPPORT STATISTIQUE DÉTAILLÉ (ESPACE RAW DATA)")
    print("="*65)
    
    couverture = (total_pixels_predits / total_pixels_purs_utiles) * 100
    print(f"📍 Couverture Spatiale : {couverture:.2f}% ({total_pixels_predits:d} / {total_pixels_purs_utiles:d} px)")
    
    if len(erreurs_globales) > 0:
        mae_global = np.mean(erreurs_globales)
        medae_global = np.median(erreurs_globales)
        std_global = np.std(erreurs_globales)
        rmse_global = np.sqrt(np.mean(erreurs_globales**2))
        p95_global = np.percentile(erreurs_globales, 95)
        max_global = np.max(erreurs_globales)
        
        similitude_stricte = np.sum(erreurs_globales <= 1.0) / len(erreurs_globales) * 100
        precision_tolerance = np.sum(erreurs_globales <= TOLERANCE_PIXELS_BRUTS) / len(erreurs_globales) * 100
        
        print("\n--- MÉTRIQUES EN VOXELS PHYSIQUES (1 Voxel Z = 0.125 mm) ---")
        print(f"   • Médiane (MedAE)            : {medae_global:.2f} voxels ({medae_global * RESOLUTION_Z_MM:.3f} mm)")
        print(f"   • Moyenne (MAE)              : {mae_global:.2f} voxels ({mae_global * RESOLUTION_Z_MM:.3f} mm)")
        print(f"   • Écart-Type (Std)           : ± {std_global:.2f} voxels")
        print(f"   • RMSE (Pénalité outliers)   : {rmse_global:.2f} voxels")
        
        print("\n--- EXTRÊMES PHYSIQUES ---")
        print(f"   • 95e Centile (P95)          : {p95_global:.2f} voxels ({p95_global * RESOLUTION_Z_MM:.3f} mm)")
        print(f"   • Erreur Maximale Absolue    : {max_global:.2f} voxels ({max_global * RESOLUTION_Z_MM:.3f} mm)")
        
        print("\n--- TAUX DE SIMILITUDE ---")
        print(f"   • Similitude Stricte (≤ 1px) : {similitude_stricte:.2f} %")
        print(f"   • Tolérance Clinique (≤{TOLERANCE_PIXELS_BRUTS:.1f}px): {precision_tolerance:.2f} %")
    
    print("="*65)

    # --- CRÉATION DU GRAPHIQUE ---
    plt.figure(figsize=(16, 8), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('white')
    
    mae_arr = np.array(mae_liste)
    std_arr = np.array(std_liste)
    max_arr = np.array(max_liste)
    
    plt.fill_between(frames_valides, 
                     np.clip(mae_arr - std_arr, 0, None), 
                     mae_arr + std_arr, 
                     color='cyan', alpha=0.2, label='Écart-Type ($\pm 1 \sigma$)')
                     
    plt.plot(frames_valides, max_arr, color='orange', linewidth=0.8, linestyle=':', alpha=0.8, label='Erreur Maximale (Pire voxel)')
    plt.plot(frames_valides, mae_arr, color='cyan', linewidth=2, label='Erreur Moyenne (MAE)')
    plt.axhline(y=TOLERANCE_PIXELS_BRUTS, color='lime', linestyle='--', linewidth=2, label=f'Tolérance ({TOLERANCE_PIXELS_BRUTS:.1f} voxels)')
    
    if len(erreurs_globales) > 0:
        plt.axhline(y=p95_global, color='magenta', linestyle='-.', linewidth=1.5, label=f'95e Centile ({p95_global:.1f} voxels)')

    plt.title("Analyse Statistique sur Données Brutes (Raw Data)", color='white', pad=15, fontsize=14)
    plt.xlabel("Index de la Frame (Y)", color='white', fontsize=12)
    plt.ylabel("Écart Absolu (Voxels physiques)", color='white', fontsize=12)
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')
    plt.grid(color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("evaluation_statistique_rawdata.png", dpi=150, facecolor='#1e1e1e')
    print("\n📈 Graphique absolu sauvegardé sous 'evaluation_statistique_rawdata.png'.")
    plt.show()

if __name__ == "__main__":
    evaluer_modele_rawdata()