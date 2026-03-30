import os
import sys
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi

# ==========================================
# CONFIGURATION
# ==========================================
DOSSIER_MASQUES = r"pipeline/2steps_pu_learning"
FICHIER_NORMAL = r"dicom_data/dicom/4261_fromdcm.npz"
FICHIER_PEAU = os.path.join(DOSSIER_MASQUES, "surface_peau.npy")
FICHIER_PRED = os.path.join(DOSSIER_MASQUES, "prediction_pu.npy")

GAUSSIAN_SMOOTH_FRANGI = 0.7 
SEUIL_FRANGI = 0.05

def normalize_image(img):
    """Normalise une image 2D pour l'affichage (0-255)."""
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo: hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (255 * out).astype(np.uint8)

def create_rgba_overlay(mask_2d, color=(255, 0, 0), alpha=150):
    """Crée un overlay RGBA à partir d'un masque booléen 2D."""
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(mask_2d, alpha, 0).astype(np.uint8)
    return rgba

class ComparateurMIP4Vues(QtWidgets.QWidget):
    def __init__(self, vol_brut, surf_safe, pred_vol, nx, ny, nz):
        super().__init__()
        self.setWindowTitle("Comparaison des Stratégies d'Effacement (MIP 2D)")
        
        self.vol_brut = vol_brut
        self.surf_safe = surf_safe
        self.pred_vol = pred_vol
        self.nx, self.ny, self.nz = nx, ny, nz
        
        self.plots = []
        self.init_ui()
        self.compute_and_display()

    def init_ui(self):
        layout = QtWidgets.QGridLayout(self)
        
        # Titres des 4 vues
        titres = [
            "1. Témoin (MIP Original)", 
            "2. Modèle PU (Prédictions en Rouge)",
            "3. Coupe Brutale (Tout l'air mis à 0)", 
            "4. Gommage Ciblé (Colonnes Frangi de l'air à 0)"
        ]
        
        for i in range(4):
            plot = pg.PlotWidget(title=titres[i])
            plot.invertY(True)
            plot.setAspectLocked(True)
            
            # Blocage strict du dézoom
            vb = plot.getViewBox()
            vb.setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.ny, 
                         maxXRange=self.nx, maxYRange=self.ny)
            
            # Synchronisation de toutes les vues sur la première
            if i > 0:
                plot.setXLink(self.plots[0])
                plot.setYLink(self.plots[0])
                
            self.plots.append(plot)
            layout.addWidget(plot, i // 2, i % 2)

    def compute_and_display(self):
        print("Calcul des données de base...")
        mip_temoin = np.max(self.vol_brut, axis=2)
        
        # ---------------------------------------------------------
        # VUE 1 : TÉMOIN
        # ---------------------------------------------------------
        img_temoin = pg.ImageItem(normalize_image(mip_temoin).T)
        self.plots[0].addItem(img_temoin)
        
        # ---------------------------------------------------------
        # VUE 2 : PRÉDICTION DU MODÈLE PU
        # ---------------------------------------------------------
        # On affiche le témoin en fond
        self.plots[1].addItem(pg.ImageItem(normalize_image(mip_temoin).T))
        
        if self.pred_vol is not None:
            # MIP des prédictions (Probabilité maximale sur l'axe Z)
            mip_pred = np.max(self.pred_vol, axis=2) > 0.5
            overlay = create_rgba_overlay(mip_pred.T, color=(255, 0, 0), alpha=150)
            self.plots[1].addItem(pg.ImageItem(overlay))
        else:
            self.plots[1].setTitle("2. Modèle PU (Fichier introuvable)")

        # ---------------------------------------------------------
        # PRÉPARATION MASQUES 3D POUR VUES 3 ET 4
        # ---------------------------------------------------------
        print("Calcul des masques spatiaux...")
        Z_grid = np.arange(self.nz).reshape(1, 1, self.nz)
        air_mask_3d = Z_grid < self.surf_safe[:, :, None]

        # ---------------------------------------------------------
        # VUE 3 : COUPE BRUTALE DE L'AIR
        # ---------------------------------------------------------
        vol_v1 = self.vol_brut.copy()
        vol_v1[air_mask_3d] = 0
        mip_cut = np.max(vol_v1, axis=2)
        self.plots[2].addItem(pg.ImageItem(normalize_image(mip_cut).T))

        # ---------------------------------------------------------
        # VUE 4 : GOMMAGE CIBLÉ FRANGI
        # ---------------------------------------------------------
        print("Calcul de Frangi pour le gommage ciblé...")
        z_max = np.argmax(self.vol_brut, axis=2)
        
        lo, hi = np.percentile(mip_temoin, 1), np.percentile(mip_temoin, 99)
        mip_norm = np.clip((mip_temoin - lo) / (hi - lo), 0, 1)
        
        f2d = frangi(mip_norm, sigmas=[0.5, 1.0, 1.5], black_ridges=False)
        f2d = gaussian_filter(f2d, sigma=GAUSSIAN_SMOOTH_FRANGI)
        f2d = (f2d - f2d.min()) / (f2d.max() - f2d.min() + 1e-8)
        
        # Tubes validés situés dans l'air
        poils_air_2d = (f2d > SEUIL_FRANGI) & (z_max < self.surf_safe)
        
        # Masque 3D : L'intersection entre le tube 2D et la zone d'air
        poils_air_3d_mask = poils_air_2d[:, :, None] & air_mask_3d

        vol_v2 = self.vol_brut.copy()
        vol_v2[poils_air_3d_mask] = 0
        mip_gommage = np.max(vol_v2, axis=2)
        self.plots[3].addItem(pg.ImageItem(normalize_image(mip_gommage).T))

def main():
    print("Chargement des volumes...")
    vol_brut = np.load(FICHIER_NORMAL)["volume"].astype(np.float32)
    nx, ny, nz = vol_brut.shape
    
    if not os.path.exists(FICHIER_PEAU):
        print(f"Erreur critique: {FICHIER_PEAU} introuvable.")
        return
    surface_peau = np.load(FICHIER_PEAU)
    surf_safe = np.nan_to_num(surface_peau, nan=nz)
    
    pred_vol = None
    if os.path.exists(FICHIER_PRED):
        pred_vol = np.load(FICHIER_PRED)
    else:
        print(f"Avertissement: Prédiction non trouvée ({FICHIER_PRED}). La vue 2 sera vide.")

    print("Ouverture de l'interface...")
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = ComparateurMIP4Vues(vol_brut, surf_safe, pred_vol, nx, ny, nz)
    win.resize(1800, 1000)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()