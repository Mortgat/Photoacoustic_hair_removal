import os
import sys
import traceback
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy import QtWidgets, QtCore
from scipy.ndimage import maximum_filter1d

# ==========================================
# CONFIGURATION
# ==========================================
FICHIER_BRUT = r"dicom_data/dicom/4261_fromdcm.npz"
DOSSIER_MASQUES = r"pipeline/2steps_pu_learning"

FICHIER_FRANGI = os.path.join(DOSSIER_MASQUES, "carte_frangi.npy")
FICHIER_PEAU = os.path.join(DOSSIER_MASQUES, "surface_peau.npy")
FICHIER_PRED_PU = os.path.join(DOSSIER_MASQUES, "prediction_pu.npy")
FICHIER_POILS_SURS = os.path.join(DOSSIER_MASQUES, "masque_poils_surs.npy")

DILATATION_RADIUS = 1  

class SniperDiagnosticTool(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diagnostic PU Learning : MIP & Extraction 3D Locale (Log-Scale)")
        
        self.load_data()
        self.init_ui()
        self.update_mip_display()

    def load_data(self):
        print("📂 Chargement du volume brut...")
        self.vol_brut = np.load(FICHIER_BRUT)["volume"].astype(np.float32)
        self.nx, self.ny, self.nz = self.vol_brut.shape
        
        print("🔍 Filtrage Frangi (Dilatation 1 pixel)...")
        frangi_3d = np.load(FICHIER_FRANGI)
        masque_frangi = (frangi_3d > 0).astype(np.uint8)
        del frangi_3d
        
        tf = DILATATION_RADIUS * 2 + 1
        masque_frangi = maximum_filter1d(masque_frangi, size=tf, axis=0)
        masque_frangi = maximum_filter1d(masque_frangi, size=tf, axis=1)
        masque_frangi = maximum_filter1d(masque_frangi, size=tf, axis=2)
        
        self.vol_filtre = self.vol_brut * masque_frangi
        del masque_frangi

        # Calcul du maximum absolu pour la normalisation logarithmique ultérieure
        self.hi_3d = np.percentile(self.vol_filtre, 99.9)

        print("🟠 Chargement Peau...")
        self.surface_peau = np.load(FICHIER_PEAU)
        
        print("🗺️ Calcul du MIP global...")
        self.mip_brut = np.max(self.vol_brut, axis=2)
        lo, hi = np.percentile(self.mip_brut, 1), np.percentile(self.mip_brut, 99)
        self.mip_norm = np.clip((self.mip_brut - lo) / (hi - lo), 0, 1)

        print("📊 Chargement des Masques d'audit...")
        self.mip_pu = np.zeros((self.nx, self.ny), dtype=bool)
        if os.path.exists(FICHIER_PRED_PU):
            pu = np.load(FICHIER_PRED_PU)
            # Seuil strict à 0.5 au lieu de 0 pour éviter le cyan global si probabilités
            self.mip_pu = np.max(pu, axis=2) > 0.5 if pu.ndim == 3 else pu > 0.5

        self.mip_surs = np.zeros((self.nx, self.ny), dtype=bool)
        if os.path.exists(FICHIER_POILS_SURS):
            surs = np.load(FICHIER_POILS_SURS)
            self.mip_surs = np.max(surs, axis=2) > 0

    def init_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        # --- PANNEAU GAUCHE : MIP 2D ---
        left_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)
        
        self.plot_mip = pg.PlotWidget(title="Vue Globale (MIP) - Cliquez pour extraire")
        self.plot_mip.invertY(True)
        self.plot_mip.setAspectLocked(True)
        self.img_mip = pg.ImageItem()
        self.plot_mip.addItem(self.img_mip)
        
        # Interception directe du clic sur l'image
        self.img_mip.mouseClickEvent = self.on_mip_click
        
        # Ajout des traits de repère (Croix de visée)
        pen_crosshair = pg.mkPen('r', style=QtCore.Qt.DashLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen_crosshair)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen_crosshair)
        self.plot_mip.addItem(self.v_line)
        self.plot_mip.addItem(self.h_line)
        # Positionnement initial hors champ
        self.v_line.setPos(-100)
        self.h_line.setPos(-100)
        
        # Contrôles des calques
        controls = QtWidgets.QHBoxLayout()
        left_panel.addLayout(controls)
        
        self.chk_pu = QtWidgets.QCheckBox("Prédictions PU (Cyan)")
        self.chk_pu.setChecked(True)
        self.chk_pu.stateChanged.connect(self.update_mip_display)
        controls.addWidget(self.chk_pu)
        
        self.chk_surs = QtWidgets.QCheckBox("Poils Sûrs (Jaune)")
        self.chk_surs.setChecked(True)
        self.chk_surs.stateChanged.connect(self.update_mip_display)
        controls.addWidget(self.chk_surs)
        
        self.chk_peau = QtWidgets.QCheckBox("Zone Peau Valide (Vert)")
        self.chk_peau.stateChanged.connect(self.update_mip_display)
        controls.addWidget(self.chk_peau)

        # --- PANNEAU DROIT : RENDU 3D LOCAL ---
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)
        
        self.label_3d = QtWidgets.QLabel("Volume 3D Local (Aplatissement sur la Peau | Échelle Log)")
        self.label_3d.setAlignment(QtCore.Qt.AlignCenter)
        right_panel.addWidget(self.label_3d)

        self.gl_view = gl.GLViewWidget()
        right_panel.addWidget(self.gl_view)
        
        # Blocage absolu des contrôles de zoom et de translation
        self.gl_view.pan = lambda *args, **kwargs: None
        self.gl_view.wheelEvent = lambda *args, **kwargs: None

        self.vol_item = gl.GLVolumeItem(np.zeros((10,10,10,4), dtype=np.ubyte))
        self.gl_view.addItem(self.vol_item)
        
        self.grid = gl.GLGridItem()
        self.gl_view.addItem(self.grid)

    def update_mip_display(self):
        rgb = np.zeros((self.nx, self.ny, 3), dtype=np.float32)
        base = self.mip_norm
        
        rgb[..., 0] = base 
        rgb[..., 1] = base 
        rgb[..., 2] = base 
        
        if self.chk_pu.isChecked():
            mask = self.mip_pu
            rgb[mask, 0] = 0           
            rgb[mask, 1] = np.clip(rgb[mask, 1] + 0.5, 0, 1) 
            rgb[mask, 2] = 1.0         
            
        if self.chk_surs.isChecked():
            mask = self.mip_surs
            rgb[mask, 0] = 1.0         
            rgb[mask, 1] = 1.0         
            rgb[mask, 2] = 0           
            
        if self.chk_peau.isChecked():
            mask = ~np.isnan(self.surface_peau)
            rgb[mask, 1] = np.clip(rgb[mask, 1] + 0.2, 0, 1) 

        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        self.img_mip.setImage(rgb, autoLevels=False)

    def on_mip_click(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
            
        try:
            pos = event.pos()
            cx, cy = int(pos.x()), int(pos.y())
            
            if not (0 <= cx < self.nx and 0 <= cy < self.ny):
                return

            # Déplacement de la croix de visée
            self.v_line.setPos(cx)
            self.h_line.setPos(cy)

            # Fenêtre de 5% de la dimension maximale
            patch_size = int(max(self.nx, self.ny) * 0.05)
            half = patch_size // 2
            
            x_min, x_max = max(0, cx - half), min(self.nx, cx + half)
            y_min, y_max = max(0, cy - half), min(self.ny, cy + half)
            
            self.extract_and_render_3d_patch(x_min, x_max, y_min, y_max)
        
        except Exception as e:
            print("Erreur lors de l'extraction du patch 3D :")
            traceback.print_exc()

    def extract_and_render_3d_patch(self, xmin, xmax, ymin, ymax):
        vol_patch = self.vol_filtre[xmin:xmax, ymin:ymax, :]
        skin_patch = self.surface_peau[xmin:xmax, ymin:ymax]
        
        px, py, pz = vol_patch.shape
        flat_vol = np.zeros_like(vol_patch)
        
        target_z = pz // 2
        
        # Aplatissement topographique
        for i in range(px):
            for j in range(py):
                z_skin = skin_patch[i, j]
                if not np.isnan(z_skin):
                    shift = target_z - int(z_skin)
                    flat_vol[i, j, :] = np.roll(vol_patch[i, j, :], shift)
                    
        # Application du Log-scale
        flat_vol_pos = np.clip(flat_vol, 0, None)
        log_vol = np.log1p(flat_vol_pos)
        
        global_log_hi = np.log1p(max(0, self.hi_3d))
        norm_vol = np.clip(log_vol / (global_log_hi + 1e-6), 0, 1)
        
        # Conversion RGBA
        val = (norm_vol * 255).astype(np.ubyte)
        
        rgba = np.zeros((px, py, pz, 4), dtype=np.ubyte)
        rgba[..., 0] = val 
        rgba[..., 1] = val 
        rgba[..., 2] = val 
        
        # Opacité quadratique pour filtrer le bruit de fond remonté par le log
        alpha = (norm_vol**2 * 255).astype(np.ubyte) 
        rgba[..., 3] = alpha
        
        self.vol_item.setData(rgba)
        
        # Verrouillage absolu de la caméra autour du centre du patch cliqué
        self.gl_view.opts['center'] = pg.Vector(px/2, py/2, target_z)
        self.gl_view.opts['distance'] = max(px, py) * 1.5
        
        # Placement de la grille
        self.grid.translate(0, 0, target_z - self.grid.transform().m33())
        self.grid.resetTransform()
        self.grid.translate(px//2, py//2, target_z)

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = SniperDiagnosticTool()
    win.resize(1800, 900)
    win.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()