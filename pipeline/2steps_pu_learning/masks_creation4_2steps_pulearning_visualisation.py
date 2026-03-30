import os
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore

def load_npz_array(npz_path, key="volume"):
    return np.load(npz_path)[key]

def normalize_to_uint8(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo: hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (255 * out).astype(np.uint8)

def rgba_overlay(mask_2d, color_rgb, alpha=100):
    """Crée un calque RGBA aux dimensions exactes de PyQtGraph (row-major)"""
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.where(mask_2d, alpha, 0).astype(np.uint8)
    return rgba


class VisualiseurPU(QtWidgets.QWidget):
    def __init__(self, volume, surface_z, pos_mask, hn_mask, frangi_map, mip_2d):
        super().__init__()
        self.setWindowTitle("Inspection Parfaite V2 : XZ par Y + MIP XY")

        self.volume = volume
        self.surface_z = surface_z.astype(np.float32)
        self.pos_mask = pos_mask
        self.hn_mask = hn_mask
        self.frangi_map = frangi_map

        self.nx, self.ny, self.nz = self.volume.shape
        self.current_y = self.ny // 2

        print("Préparation des projections 2D...")
        # L'image XY s'affiche avec X horizontal et Y vertical. En row-major, shape = (ny, nx)
        self.mip_xy_u8 = normalize_to_uint8(mip_2d).T
        self.pos_xy = np.any(self.pos_mask, axis=2).T
        self.hn_xy = np.any(self.hn_mask, axis=2).T
        self.frangi_xy = np.any(self.frangi_map > 0.1, axis=2).T

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

        controls.addWidget(QtWidgets.QLabel("Y :"))
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y.setMinimum(0)
        self.slider_y.setMaximum(self.ny - 1)
        self.slider_y.setValue(self.current_y)
        controls.addWidget(self.slider_y, stretch=1)
        self.label_y_value = QtWidgets.QLabel(str(self.current_y))
        controls.addWidget(self.label_y_value)

        self.cb_bg = QtWidgets.QCheckBox("Fond Volume")
        self.cb_bg.setChecked(True)
        self.cb_pos = QtWidgets.QCheckBox("Poils Sûrs (Vert)")
        self.cb_pos.setChecked(True)
        self.cb_hn = QtWidgets.QCheckBox("Hard Negatives (Rouge)")
        self.cb_hn.setChecked(True)
        self.cb_fr = QtWidgets.QCheckBox("Frangi (Bleu)")
        self.cb_fr.setChecked(True)
        
        for cb in [self.cb_bg, self.cb_pos, self.cb_hn, self.cb_fr]:
            controls.addWidget(cb)
            cb.stateChanged.connect(self.update_all)
        self.slider_y.valueChanged.connect(self.on_y_changed)

    def init_top_view(self):
        self.plot_xy.setLabel('bottom', 'X')
        self.plot_xy.setLabel('left', 'Y')
        self.plot_xy.invertY(True)
        
        self.xy_base = pg.ImageItem()
        self.xy_fr = pg.ImageItem()
        self.xy_hn = pg.ImageItem()
        self.xy_pos = pg.ImageItem()
        
        for item in [self.xy_base, self.xy_fr, self.xy_hn, self.xy_pos]:
            self.plot_xy.addItem(item)

        self.xy_y_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=2))
        self.plot_xy.addItem(self.xy_y_line)
        self.plot_xy.scene().sigMouseClicked.connect(self.on_xy_mouse_clicked)
        self.plot_xy.getViewBox().setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.ny, maxXRange=self.nx, maxYRange=self.ny)

    def init_xz_view(self):
        self.plot_xz.setLabel('bottom', 'X')
        self.plot_xz.setLabel('left', 'Z (Profondeur)')
        self.plot_xz.invertY(True)
        
        self.xz_base = pg.ImageItem()
        self.xz_fr = pg.ImageItem()
        self.xz_hn = pg.ImageItem()
        self.xz_pos = pg.ImageItem()
        
        for item in [self.xz_base, self.xz_fr, self.xz_hn, self.xz_pos]:
            self.plot_xz.addItem(item)

        self.skin_curve = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
        self.limit_curve = pg.PlotCurveItem(pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
        self.plot_xz.addItem(self.skin_curve)
        self.plot_xz.addItem(self.limit_curve)
        
        self.plot_xz.getViewBox().setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.nz, maxXRange=self.nx, maxYRange=self.nz)

    def on_y_changed(self, value):
        self.current_y = int(value)
        self.label_y_value.setText(str(self.current_y))
        self.update_all()

    def update_all(self):
        y = self.current_y
        
        # Dimensions row-major pour des images vides parfaites
        empty_xy_rgba = np.zeros((self.ny, self.nx, 4), dtype=np.uint8)
        empty_xz_rgba = np.zeros((self.nz, self.nx, 4), dtype=np.uint8)
        empty_xy_base = np.zeros((self.ny, self.nx), dtype=np.uint8)
        empty_xz_base = np.zeros((self.nz, self.nx), dtype=np.uint8)

        # MAJ VUE XY (Dessus)
        self.xy_base.setImage(self.mip_xy_u8 if self.cb_bg.isChecked() else empty_xy_base, autoLevels=False)
        self.xy_pos.setImage(rgba_overlay(self.pos_xy, (0, 255, 0), 120) if self.cb_pos.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_hn.setImage(rgba_overlay(self.hn_xy, (255, 0, 0), 80) if self.cb_hn.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_fr.setImage(rgba_overlay(self.frangi_xy, (0, 100, 255), 60) if self.cb_fr.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_y_line.setPos(y)

        # MAJ VUE XZ (Coupe)
        slice_xz = self.volume[:, y, :].T  # Shape (nz, nx)
        self.xz_base.setImage(normalize_to_uint8(slice_xz) if self.cb_bg.isChecked() else empty_xz_base, autoLevels=False)
        
        pos_mask = self.pos_mask[:, y, :].T
        hn_mask = self.hn_mask[:, y, :].T
        fr_mask = self.frangi_map[:, y, :].T > 0.1
        
        self.xz_pos.setImage(rgba_overlay(pos_mask, (0, 255, 0), 150) if self.cb_pos.isChecked() else empty_xz_rgba, autoLevels=False)
        self.xz_hn.setImage(rgba_overlay(hn_mask, (255, 0, 0), 80) if self.cb_hn.isChecked() else empty_xz_rgba, autoLevels=False)
        self.xz_fr.setImage(rgba_overlay(fr_mask, (0, 100, 255), 60) if self.cb_fr.isChecked() else empty_xz_rgba, autoLevels=False)

        # MAJ Lignes Z-Gating
        x_vals = np.arange(self.nx)
        z_skin = self.surface_z[:, y]
        valid = ~np.isnan(z_skin)
        
        if np.any(valid):
            self.skin_curve.setData(x_vals[valid], z_skin[valid])
            self.limit_curve.setData(x_vals[valid], z_skin[valid] - 40) # 40 voxels = 5.0 mm
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
    hn = np.load(os.path.join(dossier, "masque_hard_negatives.npy"))
    frangi = np.load(os.path.join(dossier, "carte_frangi.npy"))
    mip_2d = np.max(vol, axis=2)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = VisualiseurPU(vol, surf, pos, hn, frangi, mip_2d)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()