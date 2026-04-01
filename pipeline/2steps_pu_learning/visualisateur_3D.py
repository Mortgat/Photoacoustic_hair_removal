import os
import sys
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
FICHIER_VAISSEAUX = os.path.join(DOSSIER_MASQUES, "masque_vaisseaux_surs.npy")
FICHIER_MIP = os.path.join(DOSSIER_MASQUES, "mip_2d.npy")

DILATATION_RADIUS = 1  

# ==========================================
# MOTEUR 3D INTELLIGENT
# ==========================================
class SmartGLViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_dist = None
        self.base_center = None

    def set_base_state(self, distance, center):
        self.max_dist = distance
        self.base_center = center
        self.opts['distance'] = distance
        self.opts['center'] = pg.Vector(*center)

    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        if self.max_dist is not None and self.opts['distance'] >= self.max_dist:
            self.opts['distance'] = self.max_dist
            if self.base_center is not None:
                self.opts['center'] = pg.Vector(*self.base_center)
        self.update()

    def mouseMoveEvent(self, ev):
        is_panning = (ev.buttons() == QtCore.Qt.MiddleButton)
        if is_panning and self.max_dist is not None:
            if self.opts['distance'] >= self.max_dist - 0.1: return 
        super().mouseMoveEvent(ev)

def rgba_overlay(mask_2d, color_rgb, alpha=100):
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color_rgb
    rgba[..., 3] = np.where(mask_2d, alpha, 0).astype(np.uint8)
    return rgba

def rgba_overlay_prob(pred_2d, color_rgb, prob_mode):
    h, w = pred_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color_rgb
    if prob_mode:
        alpha = (pred_2d * 200).astype(np.uint8)
        rgba[..., 3] = np.where(pred_2d > 0.05, alpha, 0)
    else:
        rgba[..., 3] = np.where(pred_2d > 0.5, 150, 0).astype(np.uint8)
    return rgba

class SniperDiagnosticTool(QtWidgets.QWidget):
    def __init__(self, vol, surf, pos, pred_map, mip_norm, vol_filtre, vaisseaux_mask):
        super().__init__()
        self.setWindowTitle("Diagnostic PU : MIP Original & Extraction 3D Colorisée")
        
        self.vol_brut = vol
        self.surface_peau = surf.astype(np.float32)
        self.pos_mask = pos
        self.pred_map = pred_map
        self.vol_filtre = vol_filtre
        self.vaisseaux_mask = vaisseaux_mask
        self.has_vaisseaux = np.any(self.vaisseaux_mask)
        
        self.nx, self.ny, self.nz = self.vol_brut.shape
        self.mip_xy_u8 = (mip_norm * 255).astype(np.uint8).T
        self.pos_xy = np.any(self.pos_mask, axis=2).T
        self.vs_xy = np.any(self.vaisseaux_mask, axis=2).T
        
        if self.pred_map.ndim == 3: self.pred_xy = np.max(self.pred_map, axis=2).T
        else: self.pred_xy = self.pred_map.T

        self.init_ui()
        self.update_mip_display()

    def init_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        left_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)
        
        self.plot_xy = pg.PlotWidget(title="Vue de dessus XY (MIP)")
        self.plot_xy.invertY(True)
        self.plot_xy.setAspectLocked(True)
        left_panel.addWidget(self.plot_xy, stretch=1)
        
        self.xy_base = pg.ImageItem()
        self.xy_vs = pg.ImageItem()
        self.xy_pos = pg.ImageItem()
        self.xy_pred = pg.ImageItem()
        
        for item in [self.xy_base, self.xy_vs, self.xy_pos, self.xy_pred]:
            self.plot_xy.addItem(item)
            
        vb = self.plot_xy.getViewBox()
        vb.setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.ny, maxXRange=self.nx, maxYRange=self.ny)
        
        pen_crosshair = pg.mkPen('cyan', width=1, style=QtCore.Qt.DashLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen_crosshair)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen_crosshair)
        self.plot_xy.addItem(self.v_line)
        self.plot_xy.addItem(self.h_line)
        self.v_line.setPos(-100); self.h_line.setPos(-100)
        
        self.plot_xy.scene().sigMouseClicked.connect(self.on_xy_mouse_clicked)
        
        controls = QtWidgets.QHBoxLayout()
        left_panel.addLayout(controls)
        
        self.cb_pos = QtWidgets.QCheckBox("Poils (Jaune)")
        self.cb_pos.setChecked(True)
        self.cb_vs = QtWidgets.QCheckBox("Vaisseaux (Magenta)")
        self.cb_vs.setChecked(True)
        
        controls.addWidget(self.cb_pos)
        controls.addWidget(self.cb_vs)
        
        if not self.has_vaisseaux:
            self.cb_vs.hide()
            self.cb_vs.setChecked(False)

        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.VLine)
        controls.addWidget(line)

        self.radio_prob = QtWidgets.QRadioButton("Pred: Probas (Rouge)")
        self.radio_bin = QtWidgets.QRadioButton("Pred: Binaire (>0.5)")
        self.radio_off = QtWidgets.QRadioButton("Pred: Masquées")
        self.radio_prob.setChecked(True)

        for r in [self.radio_prob, self.radio_bin, self.radio_off]: controls.addWidget(r)

        self.cb_pos.stateChanged.connect(self.update_mip_display)
        self.cb_vs.stateChanged.connect(self.update_mip_display)
        self.radio_prob.toggled.connect(self.update_mip_display)
        self.radio_bin.toggled.connect(self.update_mip_display)
        self.radio_off.toggled.connect(self.update_mip_display)

        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)
        
        self.label_3d = QtWidgets.QLabel("Volume 3D Local (Régression Linéaire Peau + Masques)")
        self.label_3d.setAlignment(QtCore.Qt.AlignCenter)
        right_panel.addWidget(self.label_3d)

        self.gl_view = SmartGLViewWidget()
        right_panel.addWidget(self.gl_view, stretch=1)
        
        self.vol_item = gl.GLVolumeItem(np.zeros((10,10,10,4), dtype=np.ubyte))
        self.gl_view.addItem(self.vol_item)
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=200, y=200, z=0)
        self.gl_view.addItem(self.grid)
        
        self.info_panel = QtWidgets.QLabel("Cliquez sur le MIP gauche pour extraire un patch 3D.")
        self.info_panel.setStyleSheet("background-color: #222; color: #fff; padding: 10px; font-size: 14px;")
        self.info_panel.setAlignment(QtCore.Qt.AlignCenter)
        right_panel.addWidget(self.info_panel)

    def update_mip_display(self):
        show_pred = not self.radio_off.isChecked()
        prob_mode = self.radio_prob.isChecked()
        empty_xy_rgba = np.zeros((self.nx, self.ny, 4), dtype=np.uint8)

        self.xy_base.setImage(self.mip_xy_u8, autoLevels=False)
        self.xy_pos.setImage(rgba_overlay(self.pos_xy, (255, 255, 0), 150) if self.cb_pos.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_vs.setImage(rgba_overlay(self.vs_xy, (255, 0, 255), 180) if self.cb_vs.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_pred.setImage(rgba_overlay_prob(self.pred_xy, (255, 0, 0), prob_mode) if show_pred else empty_xy_rgba, autoLevels=False)

    def on_xy_mouse_clicked(self, event):
        if event.button() != QtCore.Qt.LeftButton: return
        pos = self.plot_xy.getViewBox().mapSceneToView(event.scenePos())
        cx, cy = int(pos.x()), int(pos.y())
        if not (0 <= cx < self.nx and 0 <= cy < self.ny): return
        self.v_line.setPos(cx); self.h_line.setPos(cy)
        patch_size = int(max(self.nx, self.ny) * 0.05)
        half = patch_size // 2
        x_min, x_max = max(0, cx - half), min(self.nx, cx + half)
        y_min, y_max = max(0, cy - half), min(self.ny, cy + half)
        self.extract_and_render_3d_patch(x_min, x_max, y_min, y_max, cx, cy)

    def extract_and_render_3d_patch(self, xmin, xmax, ymin, ymax, cx, cy):
        vol_patch = self.vol_filtre[xmin:xmax, ymin:ymax, :]
        skin_patch = self.surface_peau[xmin:xmax, ymin:ymax]
        surs_patch = self.pos_mask[xmin:xmax, ymin:ymax, :] > 0
        vs_patch = self.vaisseaux_mask[xmin:xmax, ymin:ymax, :] > 0
        
        if self.pred_map.ndim == 3: pu_patch = self.pred_map[xmin:xmax, ymin:ymax, :] > 0.5
        else: pu_patch = np.zeros_like(surs_patch)
            
        px, py, pz = vol_patch.shape
        target_z = pz // 2

        skin_3d_patch = np.zeros_like(vol_patch, dtype=bool)
        mask_valide = ~np.isnan(skin_patch)
        if mask_valide.any():
            vx, vy = np.where(mask_valide)
            vz = np.clip(skin_patch[vx, vy], 0, pz-1).astype(int)
            skin_3d_patch[vx, vy, vz] = True

        flat_vol, flat_surs, flat_pu, flat_skin, flat_vs = [np.zeros_like(x) for x in (vol_patch, surs_patch, pu_patch, skin_3d_patch, vs_patch)]

        if np.sum(mask_valide) >= 3:
            X, Y = np.meshgrid(np.arange(px), np.arange(py), indexing='ij')
            x_val, y_val, z_val = X[mask_valide], Y[mask_valide], skin_patch[mask_valide]
            A = np.c_[x_val, y_val, np.ones_like(x_val)]
            C, _, _, _ = np.linalg.lstsq(A, z_val, rcond=None)
            plan_z = C[0] * X + C[1] * Y + C[2]
            
            for i in range(px):
                for j in range(py):
                    shift = target_z - int(np.round(plan_z[i, j]))
                    flat_vol[i, j, :] = np.roll(vol_patch[i, j, :], shift)
                    flat_surs[i, j, :] = np.roll(surs_patch[i, j, :], shift)
                    flat_pu[i, j, :] = np.roll(pu_patch[i, j, :], shift)
                    flat_skin[i, j, :] = np.roll(skin_3d_patch[i, j, :], shift)
                    flat_vs[i, j, :] = np.roll(vs_patch[i, j, :], shift)
        else:
            flat_vol, flat_surs, flat_pu, flat_skin, flat_vs = vol_patch.copy(), surs_patch.copy(), pu_patch.copy(), skin_3d_patch.copy(), vs_patch.copy()

        flat_vol_pos = np.clip(flat_vol, 0, None)
        local_hi = np.percentile(flat_vol_pos, 99.9)
        if local_hi <= 0: local_hi = 1e-6
        norm_vol = np.clip(flat_vol_pos / local_hi, 0, 1)
        val = (norm_vol * 255).astype(np.ubyte)
        
        rgba = np.zeros((px, py, pz, 4), dtype=np.ubyte)
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = val 
        
        rgba[flat_surs] = [255, 255, 0, 0]
        rgba[flat_pu] = [255, 0, 0, 0]
        if self.cb_vs.isChecked(): rgba[flat_vs] = [255, 0, 255, 0]
        rgba[flat_skin] = [0, 255, 0, 0]

        alpha = (norm_vol**2 * 150).astype(np.ubyte) 
        mask_any = flat_surs | flat_pu | flat_skin
        if self.cb_vs.isChecked(): mask_any = mask_any | flat_vs
        alpha[mask_any] = np.maximum(alpha[mask_any], 180)
        rgba[..., 3] = alpha
        
        rgba = rgba[:, :, ::-1, :]
        self.vol_item.setData(rgba)
        
        base_center = (px/2, py/2, target_z)
        max_distance = max(px, py) * 1.5
        self.gl_view.set_base_state(max_distance, base_center)
        
        self.grid.translate(0, 0, target_z - self.grid.transform().m33())
        self.grid.resetTransform()
        self.grid.translate(px//2, py//2, target_z)
        
        max_val, mean_val = np.max(vol_patch), np.mean(vol_patch[vol_patch > 0]) if np.any(vol_patch > 0) else 0
        z_peau_local = self.surface_peau[cx, cy]
        
        self.info_panel.setText(f"<b>Cible :</b> X={cx}, Y={cy} | <b>Z Peau :</b> {z_peau_local:.1f} voxels<br><b>Intensité Brut Max :</b> {max_val:.2f} | <b>Moyenne (Hors vide) :</b> {mean_val:.2f}")

def main():
    vol = np.load(FICHIER_BRUT)["volume"].astype(np.float32)
    surf = np.load(FICHIER_PEAU)
    pos = np.load(FICHIER_POILS_SURS)
    mip_norm = np.load(FICHIER_MIP)
    pred_map = np.load(FICHIER_PRED_PU)
    
    vs = np.load(FICHIER_VAISSEAUX) if os.path.exists(FICHIER_VAISSEAUX) else np.zeros_like(pos)
    
    if os.path.exists(FICHIER_FRANGI):
        frangi_3d = np.load(FICHIER_FRANGI)
        masque_frangi = (frangi_3d > 0).astype(np.uint8)
        del frangi_3d
        tf = DILATATION_RADIUS * 2 + 1
        masque_frangi = maximum_filter1d(maximum_filter1d(maximum_filter1d(masque_frangi, size=tf, axis=0), size=tf, axis=1), size=tf, axis=2)
        vol_filtre = vol * masque_frangi
    else: vol_filtre = vol.copy()

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    win = SniperDiagnosticTool(vol, surf, pos, pred_map, mip_norm, vol_filtre, vs)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()