import os
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore

def load_npz_array(npz_path, key="volume"):
    return np.load(npz_path)[key]

def normalize_to_uint8_slice(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo: hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (255 * out).astype(np.uint8)

def rgba_overlay(mask_2d, color_rgb, alpha=100):
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.where(mask_2d, alpha, 0).astype(np.uint8)
    return rgba

def rgba_overlay_prob(pred_2d, color_rgb, prob_mode):
    h, w = pred_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    
    if prob_mode:
        alpha = (pred_2d * 200).astype(np.uint8)
        rgba[..., 3] = np.where(pred_2d > 0.05, alpha, 0)
    else:
        rgba[..., 3] = np.where(pred_2d > 0.5, 150, 0).astype(np.uint8)
        
    return rgba

class VisualiseurPUPrediction(QtWidgets.QWidget):
    def __init__(self, volume, surface_z, pos_mask, pred_map, mip_norm):
        super().__init__()
        self.setWindowTitle("Inspection des Prédictions : PU Learning")

        self.volume = volume
        self.surface_z = surface_z.astype(np.float32)
        self.pos_mask = pos_mask
        self.pred_map = pred_map

        self.nx, self.ny, self.nz = self.volume.shape
        self.current_y = self.ny // 2

        print("Préparation des projections 2D...")
        self.mip_xy_u8 = (mip_norm * 255).astype(np.uint8).T
        self.pos_xy = np.any(self.pos_mask, axis=2).T
        self.pred_xy = np.max(self.pred_map, axis=2).T

        self.init_ui()
        self.init_top_view()
        self.init_xz_view()
        self.update_all()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        self.label_xz = QtWidgets.QLabel("Coupe XZ")
        self.label_xz.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.label_xz)
        self.plot_xz = pg.PlotWidget()
        left_layout.addWidget(self.plot_xz, stretch=1)
        split.addWidget(left_widget)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        self.label_xy = QtWidgets.QLabel("Vue de dessus XY (MIP)")
        self.label_xy.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.label_xy)
        self.plot_xy = pg.PlotWidget()
        right_layout.addWidget(self.plot_xy, stretch=1)
        split.addWidget(right_widget)
        
        split.setSizes([1000, 800])

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y.setMinimum(0)
        self.slider_y.setMaximum(self.ny - 1)
        self.slider_y.setValue(self.current_y)
        controls.addWidget(QtWidgets.QLabel("Y :"))
        controls.addWidget(self.slider_y, stretch=1)
        self.label_y_value = QtWidgets.QLabel(str(self.current_y))
        controls.addWidget(self.label_y_value)

        # Séparateur visuel
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        controls.addWidget(line)

        # Contrôles de base
        self.cb_bg = QtWidgets.QCheckBox("Fond Volume")
        self.cb_bg.setChecked(True)
        self.cb_pos = QtWidgets.QCheckBox("Poils GT (Vert)")
        self.cb_pos.setChecked(True)
        controls.addWidget(self.cb_bg)
        controls.addWidget(self.cb_pos)

        # Séparateur visuel
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.VLine)
        controls.addWidget(line2)

        # Boutons Radio exclusifs pour la prédiction
        self.radio_prob = QtWidgets.QRadioButton("Pred: Probabilités")
        self.radio_bin = QtWidgets.QRadioButton("Pred: Binaire (>0.5)")
        self.radio_off = QtWidgets.QRadioButton("Pred: Masquées")
        self.radio_prob.setChecked(True)

        controls.addWidget(self.radio_prob)
        controls.addWidget(self.radio_bin)
        controls.addWidget(self.radio_off)

        # Connexions
        self.cb_bg.stateChanged.connect(self.update_all)
        self.cb_pos.stateChanged.connect(self.update_all)
        self.radio_prob.toggled.connect(self.update_all)
        self.radio_bin.toggled.connect(self.update_all)
        self.radio_off.toggled.connect(self.update_all)
        self.slider_y.valueChanged.connect(self.on_y_changed)

    def init_top_view(self):
        self.plot_xy.setLabel('bottom', 'X')
        self.plot_xy.setLabel('left', 'Y')
        self.plot_xy.invertY(True)
        
        self.xy_base = pg.ImageItem()
        self.xy_pos = pg.ImageItem()
        self.xy_pred = pg.ImageItem()
        
        for item in [self.xy_base, self.xy_pos, self.xy_pred]:
            self.plot_xy.addItem(item)

        self.xy_y_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=2))
        self.plot_xy.addItem(self.xy_y_line)
        self.plot_xy.scene().sigMouseClicked.connect(self.on_xy_mouse_clicked)
        
        vb = self.plot_xy.getViewBox()
        vb.setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.ny, maxXRange=self.nx, maxYRange=self.ny)

    def init_xz_view(self):
        self.plot_xz.setLabel('bottom', 'X')
        self.plot_xz.setLabel('left', 'Z (Profondeur)')
        self.plot_xz.invertY(True)
        
        self.xz_base = pg.ImageItem()
        self.xz_pos = pg.ImageItem()
        self.xz_pred = pg.ImageItem()
        
        for item in [self.xz_base, self.xz_pos, self.xz_pred]:
            self.plot_xz.addItem(item)

        self.skin_curve = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
        self.limit_curve = pg.PlotCurveItem(pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
        self.plot_xz.addItem(self.skin_curve)
        self.plot_xz.addItem(self.limit_curve)
        
        vb = self.plot_xz.getViewBox()
        vb.setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.nz, maxXRange=self.nx, maxYRange=self.nz)

    def on_y_changed(self, value):
        self.current_y = int(value)
        self.label_y_value.setText(str(self.current_y))
        self.update_all()

    def update_all(self):
        y = self.current_y
        
        show_pred = not self.radio_off.isChecked()
        prob_mode = self.radio_prob.isChecked()
        
        empty_xy_rgba = np.zeros((self.ny, self.nx, 4), dtype=np.uint8)
        empty_xz_rgba = np.zeros((self.nz, self.nx, 4), dtype=np.uint8)
        empty_xy_base = np.zeros((self.ny, self.nx), dtype=np.uint8)
        empty_xz_base = np.zeros((self.nz, self.nx), dtype=np.uint8)

        # MAJ VUE XY
        self.xy_base.setImage(self.mip_xy_u8 if self.cb_bg.isChecked() else empty_xy_base, autoLevels=False)
        self.xy_pos.setImage(rgba_overlay(self.pos_xy, (0, 255, 0), 120) if self.cb_pos.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_pred.setImage(rgba_overlay_prob(self.pred_xy, (255, 0, 0), prob_mode) if show_pred else empty_xy_rgba, autoLevels=False)
        self.xy_y_line.setPos(y)

        # MAJ VUE XZ
        slice_xz = self.volume[:, y, :].T
        self.xz_base.setImage(normalize_to_uint8_slice(slice_xz) if self.cb_bg.isChecked() else empty_xz_base, autoLevels=False)
        
        self.xz_pos.setImage(rgba_overlay(self.pos_mask[:, y, :].T, (0, 255, 0), 150) if self.cb_pos.isChecked() else empty_xz_rgba, autoLevels=False)
        
        pred_slice = self.pred_map[:, y, :].T
        self.xz_pred.setImage(rgba_overlay_prob(pred_slice, (255, 0, 0), prob_mode) if show_pred else empty_xz_rgba, autoLevels=False)

        x_vals = np.arange(self.nx)
        z_skin = self.surface_z[:, y]
        valid = ~np.isnan(z_skin)
        
        if np.any(valid):
            self.skin_curve.setData(x_vals[valid], z_skin[valid])
            self.limit_curve.setData(x_vals[valid], z_skin[valid] - 40)
        else:
            self.skin_curve.clear()
            self.limit_curve.clear()

    def on_xy_mouse_clicked(self, event):
        if event.button() != QtCore.Qt.LeftButton: return
        pos = self.plot_xy.getViewBox().mapSceneToView(event.scenePos())
        if 0 <= pos.y() < self.ny:
            self.slider_y.setValue(int(round(pos.y())))

def main():
    import sys
    dossier = r"pipeline/2steps_pu_learning"
    f_vol = r"dicom_data/dicom/4261_fromdcm.npz"
    
    print("Chargement des données...")
    vol = np.load(f_vol)["volume"]
    surf = np.load(os.path.join(dossier, "surface_peau.npy"))
    pos = np.load(os.path.join(dossier, "masque_poils_surs.npy"))
    mip_norm = np.load(os.path.join(dossier, "mip_2d.npy"))
    
    f_pred = os.path.join(dossier, "prediction_pu.npy")
    if not os.path.exists(f_pred):
        print(f"Erreur : Le fichier de prédiction {f_pred} est introuvable.")
        return
    pred_map = np.load(f_pred)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = VisualiseurPUPrediction(vol, surf, pos, pred_map, mip_norm)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()