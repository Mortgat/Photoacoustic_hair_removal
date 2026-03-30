import os
import sys
import time
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import frangi
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_NORMAL = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_PEAU_HIST = r"pipeline/2steps_pu_learning/surface_peau.npy"

# Paramètres Sniper
GAUSSIAN_SMOOTH_FRANGI = 0.7 
SEUIL_FRANGI_DIG = 0.05    
MAX_ITERATIONS = 60            

# Paramètres NL-Means & Coupure
NLM_H_MULT = 5.0
PERCENTILE_CUT = 96.0

# Paramètres Lissage
MEDIAN_FILTER_SIZE = 40        
POURCENTAGE_PEAU_UTILE = 98.0

class IterativeDigViewer(QtWidgets.QWidget):
    def __init__(self, vol_brut, skin_hist):
        super().__init__()
        self.setWindowTitle(f"Pipeline Complet : Sniper 60 + NLM(h=5) + Coupure 96%")
        
        self.vol_brut = vol_brut
        self.skin_hist = skin_hist
        self.nx, self.ny, self.nz = vol_brut.shape
        self.vol_work = self.vol_brut.copy()
        
        layout = QtWidgets.QHBoxLayout(self)
        
        # Vue MIP (Gauche)
        self.plot_mip = pg.PlotWidget(title="MIP Final (Après Sniper)")
        self.plot_mip.invertY(True)
        self.plot_mip.setAspectLocked(True)
        vb_mip = self.plot_mip.getViewBox()
        vb_mip.setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.ny, maxXRange=self.nx, maxYRange=self.ny)
        self.img_mip = pg.ImageItem()
        self.plot_mip.addItem(self.img_mip)
        layout.addWidget(self.plot_mip, stretch=1)
        
        # Panel Droit (Coupe XZ)
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)
        
        self.plot_xz = pg.PlotWidget(title="Coupe XZ (Affichage sur Volume Original)")
        self.plot_xz.invertY(True)
        self.plot_xz.setAspectLocked(False) 
        self.img_xz = pg.ImageItem()
        self.plot_xz.addItem(self.img_xz)
        right_panel.addWidget(self.plot_xz)
        
        self.curve_original = pg.PlotCurveItem(pen=pg.mkPen('y', width=2), connect='finite')
        self.plot_xz.addItem(self.curve_original)
        self.curve_final_smooth = pg.PlotCurveItem(pen=pg.mkPen('c', width=3), connect='finite')
        self.plot_xz.addItem(self.curve_final_smooth)
        
        controls = QtWidgets.QHBoxLayout()
        right_panel.addLayout(controls)
        
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y.setRange(0, self.ny - 1)
        self.slider_y.setValue(self.ny // 2)
        self.slider_y.valueChanged.connect(self.update_xz_view)
        controls.addWidget(self.slider_y, stretch=1)
        self.label_y = QtWidgets.QLabel(f"Y: {self.ny // 2}")
        controls.addWidget(self.label_y)

        self.vol_lo = np.percentile(self.vol_brut, 1)
        self.vol_hi = np.percentile(self.vol_brut, 99)
        if self.vol_hi <= self.vol_lo: self.vol_hi = self.vol_lo + 1e-6
        
        self.run_pipeline()

    def run_pipeline(self):
        seuil_bruit_brut = self.vol_brut.mean() + 2 * self.vol_brut.std()
        X, Y = np.indices((self.nx, self.ny))

        # --- ÉTAPE 1 : GRIGNOTAGE SNIPER ---
        print(f"\n--- ÉTAPE 1 : GRIGNOTAGE SNIPER ({MAX_ITERATIONS} itérations) ---")
        for iteration in range(MAX_ITERATIONS):
            mip_actuel = np.max(self.vol_work, axis=2)
            z_actuel = np.argmax(self.vol_work, axis=2)
            
            lo, hi = np.percentile(mip_actuel, 1), np.percentile(mip_actuel, 99)
            if hi <= lo: hi = lo + 1e-6
            mip_norm = np.clip((mip_actuel - lo) / (hi - lo), 0, 1)
            
            f2d = frangi(mip_norm, sigmas=[0.5, 1.0, 1.5], black_ridges=False)
            f2d = gaussian_filter(f2d, sigma=GAUSSIAN_SMOOTH_FRANGI)
            f2d = (f2d - f2d.min()) / (f2d.max() - f2d.min() + 1e-8)
            
            mask_dig = (f2d > SEUIL_FRANGI_DIG) & (mip_actuel > seuil_bruit_brut)
            pixels_to_dig = np.sum(mask_dig)
            
            if pixels_to_dig == 0:
                print(f"Itération {iteration+1}: 0 pixel tubulaire. Fin anticipée.")
                break
                
            self.vol_work[X[mask_dig], Y[mask_dig], z_actuel[mask_dig]] = 0

        # --- ÉTAPE 2 : DÉBRUITAGE NLM + COUPURE PERCENTILE ---
        print(f"\n--- ÉTAPE 2 : NL-MEANS + COUPURE {PERCENTILE_CUT}% ---")
        self.vol_hard = np.zeros_like(self.vol_work)
        
        for y in tqdm(range(self.ny), desc="Génération Volume Hard", unit="slice"):
            slice_2d = self.vol_work[:, y, :].T  # Shape: (nz, nx)
            
            try:
                sigma_est = np.mean(estimate_sigma(slice_2d))
            except:
                sigma_est = np.std(slice_2d) * 0.1
                
            # NL-Means
            slice_nlm = denoise_nl_means(
                slice_2d, 
                h=NLM_H_MULT * sigma_est, 
                fast_mode=True, 
                patch_size=5, 
                patch_distance=6
            )
            
            # Coupure Nette
            valeur_coupe = np.percentile(slice_nlm, PERCENTILE_CUT)
            slice_cut = np.copy(slice_nlm)
            slice_cut[slice_cut < valeur_coupe] = valeur_coupe
            
            self.vol_hard[:, y, :] = slice_cut.T

        # --- ÉTAPE 3 : CALCUL SURFACE FINALE ---
        print("\n--- ÉTAPE 3 : CALCUL SURFACE DE PEAU ---")
        self.z_final_smooth = np.zeros((self.nx, self.ny), dtype=np.float32)
        
        for y in range(self.ny):
            slice_hard = self.vol_hard[:, y, :].T # (nz, nx)
            
            fond_plat = np.min(slice_hard, axis=0) 
            mask_signal = slice_hard > (fond_plat + 1e-5)
            
            z_raw = np.argmax(mask_signal, axis=0).astype(np.float32)
            
            vide = np.max(mask_signal, axis=0) == False
            z_raw[vide] = self.nz 
            
            # Filtre médian strict de taille 40
            z_smooth = median_filter(z_raw, size=MEDIAN_FILTER_SIZE).astype(np.float32)
            z_smooth[vide] = np.nan
            z_smooth[z_smooth >= self.nz - 1] = np.nan
            
            self.z_final_smooth[:, y] = z_smooth
        
        # Rognage des bords extérieurs (1%)
        fraction = (100.0 - POURCENTAGE_PEAU_UTILE) / 200.0
        for surf in [self.skin_hist, self.z_final_smooth]:
            y_v = np.where(np.any(~np.isnan(surf), axis=0))[0]
            if len(y_v) > 0:
                ymin, ymax = y_v[0], y_v[-1]
                cy = int((ymax - ymin) * fraction)
                if cy > 0:
                    surf[:, ymin:ymin+cy] = np.nan
                    surf[:, ymax-cy+1:ymax+1] = np.nan
            for y_i in range(self.ny):
                col = surf[:, y_i]
                v = np.where(~np.isnan(col))[0]
                if len(v) > 0:
                    xmin, xmax = v[0], v[-1]
                    cx = int((xmax - xmin) * fraction)
                    if cx > 0:
                        surf[xmin:xmin+cx, y_i] = np.nan
                        surf[xmax-cx+1:xmax+1, y_i] = np.nan

        # MAJ MIP
        mip_f = np.max(self.vol_work, axis=2)
        lo, hi = np.percentile(mip_f, 1), np.percentile(mip_f, 99)
        if hi <= lo: hi = lo + 1e-6
        mip_norm = np.clip((mip_f - lo) / (hi - lo), 0, 1)
        self.img_mip.setImage((255 * mip_norm).astype(np.uint8).T, autoLevels=False)
        
        self.update_xz_view()

    def update_xz_view(self):
        y = self.slider_y.value()
        self.label_y.setText(f"Y: {y}")
        
        # L'affichage de fond se fait sur le volume brut (Doux) pour mieux voir la canopée
        slice_xz = self.vol_brut[:, y, :].T
        norm_xz = np.clip((slice_xz - self.vol_lo) / (self.vol_hi - self.vol_lo), 0, 1)
        self.img_xz.setImage((255 * norm_xz).astype(np.uint8), autoLevels=False)
        
        x_vals = np.arange(self.nx)
        self.curve_original.setData(x_vals, self.skin_hist[:, y])
        self.curve_final_smooth.setData(x_vals, self.z_final_smooth[:, y])

def main():
    if not os.path.exists(FICHIER_NORMAL) or not os.path.exists(FICHIER_PEAU_HIST):
        print("Erreur : Fichiers requis introuvables.")
        return
    vol_brut = np.load(FICHIER_NORMAL)["volume"].astype(np.float32)
    skin_hist = np.load(FICHIER_PEAU_HIST)
    
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = IterativeDigViewer(vol_brut, skin_hist)
    win.resize(1800, 900)
    win.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()