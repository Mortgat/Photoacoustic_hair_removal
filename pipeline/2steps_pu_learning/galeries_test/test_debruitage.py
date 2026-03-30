import sys
import os
import time
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets
from skimage.restoration import denoise_nl_means, estimate_sigma

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_NORMAL = r"dicom_data/dicom/4261_fromdcm.npz"

class PercentileCutoffViewerExt(QtWidgets.QWidget):
    def __init__(self, vol_brut):
        super().__init__()
        self.setWindowTitle("Laboratoire : Coupure Haute par Percentile (80% - 96%)")
        
        # Extraction de la tranche centrale
        ny = vol_brut.shape[1]
        idx = ny // 2
        self.img_raw = vol_brut[:, idx, :].T
        
        self.nz, self.nx = self.img_raw.shape
        
        layout = QtWidgets.QVBoxLayout(self)
        self.view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.view)
        
        # 9 niveaux de percentile dans la fourchette haute
        params_percentiles = [80, 82, 84, 86, 88, 90, 92, 94, 96]
        
        print(f"Tranche {idx} sélectionnée.")
        print("Étape 1/2 : Calcul du NL-Means (h=5.0)...")
        
        try:
            sigma_est = np.mean(estimate_sigma(self.img_raw))
        except:
            sigma_est = np.std(self.img_raw) * 0.1

        t0 = time.time()
        self.img_nlm = denoise_nl_means(
            self.img_raw, 
            h=5.0 * sigma_est, 
            fast_mode=True, 
            patch_size=5, 
            patch_distance=6
        )
        print(f"NL-Means terminé en {time.time()-t0:.1f}s")
        
        print("Étape 2/2 : Calcul des coupures hautes...")

        # Calcul et affichage en grille 3x3
        for i, perc in enumerate(params_percentiles):
            valeur_coupe = np.percentile(self.img_nlm, perc)
            
            img_cut = np.copy(self.img_nlm)
            img_cut[img_cut < valeur_coupe] = valeur_coupe 
            
            self.add_image_plot(f"NLM + Coupure {perc}%", img_cut, valeur_coupe)

            if (i + 1) % 3 == 0 and i < 8:
                self.view.nextRow()

        print("✅ Calculs terminés.")

    def add_image_plot(self, title, data, cut_val):
        p = self.view.addPlot(title=title)
        p.invertY(True)
        p.setAspectLocked(False)
        img_item = pg.ImageItem()
        p.addItem(img_item)
        
        lo, hi = np.percentile(data, 1), np.percentile(data, 99)
        if hi <= lo: hi = lo + 1e-6
        norm_data = np.clip((data - lo) / (hi - lo), 0, 1)
        img_uint8 = (255 * norm_data).astype(np.uint8)
        
        img_item.setImage(img_uint8, autoLevels=False)
        
        label = pg.TextItem(text=f"Cut Val: {cut_val:.2f}", color='w', anchor=(1,1))
        p.addItem(label)
        label.setPos(self.nx, self.nz)

def main():
    if not os.path.exists(FICHIER_NORMAL):
        print(f"Erreur : {FICHIER_NORMAL} introuvable.")
        return
        
    print("📂 Chargement du volume...")
    data = np.load(FICHIER_NORMAL)
    vol = data["volume"].astype(np.float32)
    
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = PercentileCutoffViewerExt(vol)
    win.resize(1800, 1000)
    win.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()