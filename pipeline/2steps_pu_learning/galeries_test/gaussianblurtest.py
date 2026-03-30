import os
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi

# ==========================================
# CONFIGURATION
# ==========================================
DOSSIER_DATA = r"pipeline/2steps_pu_learning"
SIGMAS_FRANGI = [0.5, 1.0, 1.5]
BLURS_TO_TEST = [0, 0.5, 1.0, 2.0]

class BlurBenchmarkApp(QtWidgets.QWidget):
    def __init__(self, mip_2d):
        super().__init__()
        self.setWindowTitle("Benchmark Interactif (Visibilité Accrue) : Impact du Flou sur Frangi 2D")
        self.mip_2d = mip_2d
        self.nx, self.ny = mip_2d.shape
        
        self.plots = []
        self.images = []
        
        self.init_ui()
        self.compute_and_display()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        
        # Définition du Colormap Magma (plus sensible)
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([[0,0,3,255], [80,18,123,255], [182,50,102,255], [251,135,13,255], [251,252,191,255]], dtype=np.uint8)
        self.cmap_magma = pg.ColorMap(pos, color)

        for i in range(4):
            plot_widget = pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            plot_widget.invertY(True)
            plot_widget.setMenuEnabled(False) # Désactive le menu contextuel par défaut
            
            img_item = pg.ImageItem()
            img_item.setColorMap(self.cmap_magma) # Application du colormap
            plot_widget.addItem(img_item)
            
            # Synchronisation
            if i > 0:
                plot_widget.setXLink(self.plots[0])
                plot_widget.setYLink(self.plots[0])
            
            # --- CORRECTION : BLOCAGE DU ZOOM (STRICT) ---
            vb = plot_widget.getViewBox()
            vb.setLimits(
                xMin=0, xMax=self.nx, 
                yMin=0, yMax=self.ny,
                maxXRange=self.nx, maxYRange=self.ny
            )
            
            grid.addWidget(plot_widget, i // 2, i % 2)
            self.plots.append(plot_widget)
            self.images.append(img_item)

        label = QtWidgets.QLabel("MODE HAUTE SENSIBILITÉ (0-5%). Objectif : Voir si la 'neige' se connecte pour former des lignes.")
        layout.addWidget(label)

    def compute_and_display(self):
        print("Calcul de la carte Frangi initiale sur le bruit...")
        # L'image d'entrée étant le MIP_NORM (normalisé 0-1), Frangi réagira aux micro-gradients du bruit
        f_base = frangi(self.mip_2d, sigmas=SIGMAS_FRANGI, black_ridges=False)
        
        for i, sigma_blur in enumerate(BLURS_TO_TEST):
            if sigma_blur > 0:
                data = gaussian_filter(f_base, sigma=sigma_blur)
            else:
                data = f_base
            
            # Normalisation locale stricte
            if data.max() > 0:
                data = (data - data.min()) / (data.max() - data.min())
            
            # --- CORRECTION : SEUILLAGE VISUEL DRASTIQUE ---
            # On force l'affichage à saturer à 5% d'intensité.
            # Tout pixel > 0.05 sera affiché comme "chaud" (blanc/jaune).
            # Cela permet de voir si le bruit se connecte.
            self.images[i].setImage(data.T, autoLevels=False, levels=[0, 0.05])
            self.plots[i].setTitle(f"Flou Gaussien: {sigma_blur}")

def main():
    import sys
    chemin_mip = os.path.join(DOSSIER_DATA, "mip_2d.npy")
    
    if not os.path.exists(chemin_mip):
        print(f"Erreur : Lancez d'abord le script de génération. {chemin_mip} introuvable.")
        return

    mip_2d = np.load(chemin_mip)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    gui = BlurBenchmarkApp(mip_2d)
    gui.resize(1200, 1000)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()