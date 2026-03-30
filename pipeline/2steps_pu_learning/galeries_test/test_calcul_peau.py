import sys
import os
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from scipy.ndimage import median_filter

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_HARD = r"dicom_data/dicom/4261_hard_96.npz"
FICHIER_BRUT = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_DOUX = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_PEAU_HIST = r"pipeline/2steps_pu_learning/surface_peau.npy"

MEDIAN_FILTER_SIZE = 40
PERCENTILE_PLAFOND = 50  # Conserve les 93% les moins lumineux, aveugle l'Argmax au top 7%

class SkinValidationViewer(QtWidgets.QWidget):
    def __init__(self, vol_hard, vol_brut, vol_display, skin_hist):
        super().__init__()
        self.setWindowTitle(f"Validation Peau : Plafond {PERCENTILE_PLAFOND}% + Moyenne (Verte)")
        
        self.vol_hard = vol_hard
        self.vol_brut = vol_brut
        self.vol_display = vol_display
        self.skin_hist = skin_hist
        self.nx, self.ny, self.nz = self.vol_hard.shape
        
        # --- Calcul de la surface Hard avec Plafond ---
        print(f"Calcul de la nouvelle surface (Plafond à {PERCENTILE_PLAFOND}%)...")
        self.surface_z_new = np.zeros((self.nx, self.ny), dtype=np.float32)
        
        for y in range(self.ny):
            slice_hard = self.vol_hard[:, y, :].T
            slice_brut = self.vol_brut[:, y, :].T
            
            # 1. Masque issu du volume Hard
            fond_plat = np.min(slice_hard, axis=0) 
            mask_signal = slice_hard > (fond_plat + 1e-5)
            
            # 2. Restauration des vraies valeurs
            slice_restored = np.where(mask_signal, slice_brut, -np.inf)
            
            # 3. LE PLAFOND INTELLIGENT (Ton idée)
            pixels_signal = slice_restored[mask_signal]
            if len(pixels_signal) > 0:
                plafond = np.percentile(pixels_signal, PERCENTILE_PLAFOND)
                # On élimine le top 7% le plus brillant en le poussant à -infini
                slice_restored[slice_restored > plafond] = -np.inf
            
            # 4. Argmax sur les 93% restants
            z_raw = np.argmax(slice_restored, axis=0).astype(np.float32)
            
            vide = np.max(mask_signal, axis=0) == False
            z_raw[vide] = self.nz 
            
            # 5. Filtre médian
            z_smooth = median_filter(z_raw, size=MEDIAN_FILTER_SIZE).astype(np.float32)
            z_smooth[vide] = np.nan
            z_smooth[z_smooth >= self.nz - 1] = np.nan
            
            self.surface_z_new[:, y] = z_smooth

        print("Initialisation de l'interface...")

        # --- Interface ---
        layout = QtWidgets.QVBoxLayout(self)
        
        self.view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.view, stretch=1)
        
        self.plot_xz = self.view.addPlot(title="Comparaison des Surfaces")
        self.plot_xz.invertY(True)
        self.plot_xz.setAspectLocked(False)
        self.plot_xz.setLabel('left', 'Z (Profondeur)')
        self.plot_xz.setLabel('bottom', 'X')
        
        self.img_xz = pg.ImageItem()
        self.plot_xz.addItem(self.img_xz)
        
        self.curve_hist = pg.PlotCurveItem(pen=pg.mkPen('y', width=2), connect='finite')
        self.plot_xz.addItem(self.curve_hist)

        self.curve_new = pg.PlotCurveItem(pen=pg.mkPen('c', width=2), connect='finite')
        self.plot_xz.addItem(self.curve_new)

        self.curve_avg = pg.PlotCurveItem(pen=pg.mkPen('g', width=3), connect='finite')
        self.plot_xz.addItem(self.curve_avg)
        
        # Contrôles
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y.setRange(0, self.ny - 1)
        self.slider_y.setValue(self.ny // 2)
        self.slider_y.valueChanged.connect(self.update_view)
        controls.addWidget(self.slider_y, stretch=1)
        
        self.label_y = QtWidgets.QLabel(f"Y: {self.ny // 2}")
        controls.addWidget(self.label_y)

        self.disp_lo = np.percentile(self.vol_display, 1)
        self.disp_hi = np.percentile(self.vol_display, 99)
        if self.disp_hi <= self.disp_lo: self.disp_hi = self.disp_lo + 1e-6

        self.update_view()

    def update_view(self):
        y = self.slider_y.value()
        self.label_y.setText(f"Y: {y}")
        
        slice_display = self.vol_display[:, y, :].T
        norm_slice = np.clip((slice_display - self.disp_lo) / (self.disp_hi - self.disp_lo), 0, 1)
        self.img_xz.setImage((255 * norm_slice).astype(np.uint8), autoLevels=False)
        
        x_vals = np.arange(self.nx)
        z_hist = self.skin_hist[:, y]
        z_new = self.surface_z_new[:, y]
        
        # Calcul de la moyenne simple
        z_avg = (z_hist + z_new) / 2.0
        
        # Mise à jour
        self.curve_hist.setData(x_vals, z_hist)
        self.curve_new.setData(x_vals, z_new)
        self.curve_avg.setData(x_vals, z_avg)

def main():
    if not os.path.exists(FICHIER_HARD) or not os.path.exists(FICHIER_BRUT):
        print("Erreur : Volume Hard ou Brut introuvable.")
        return
    if not os.path.exists(FICHIER_PEAU_HIST):
        print(f"Erreur : Historique de la peau introuvable à {FICHIER_PEAU_HIST}.")
        return

    vol_hard = np.load(FICHIER_HARD)["volume"].astype(np.float32)
    vol_brut = np.load(FICHIER_BRUT)["volume"].astype(np.float32)
    skin_hist = np.load(FICHIER_PEAU_HIST)

    if os.path.exists(FICHIER_DOUX):
        vol_display = np.load(FICHIER_DOUX)["volume"].astype(np.float32)
    else:
        vol_display = vol_brut

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = SkinValidationViewer(vol_hard, vol_brut, vol_display, skin_hist)
    win.resize(1800, 900)
    win.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()