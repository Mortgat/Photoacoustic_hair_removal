import os
import argparse
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore


def load_npz_array(npz_path, key="volume"):
    data = np.load(npz_path)
    if key not in data:
        raise KeyError(f"Clé '{key}' absente de {npz_path}")
    return data[key]


def normalize_to_uint8(img, low=1, high=99):
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (255 * out).astype(np.uint8)


def rgba_overlay_from_mask(mask_2d, color_rgb, alpha=80):
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.where(mask_2d, alpha, 0).astype(np.uint8)
    return rgba


class Viewer(QtWidgets.QWidget):
    def __init__(self, volume, skin, hairs, vessels):
        super().__init__()
        self.setWindowTitle("Vue de dessus XY — peau, poils, vaisseaux")

        self.volume = volume
        self.skin = skin
        self.hairs = hairs
        self.vessels = vessels

        self.nx, self.ny, self.nz = self.volume.shape

        self.bg = normalize_to_uint8(np.max(self.volume, axis=2)).T
        self.skin_xy = (self.skin < self.nz).T
        self.hairs_xy = np.any(self.hairs, axis=2).T
        self.vessels_xy = np.any(self.vessels, axis=2).T

        self.show_bg = True
        self.show_skin = True
        self.show_hairs = True
        self.show_vessels = True

        self._build_ui()
        self._update()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

        self.plot = pg.PlotWidget()
        self.plot.invertY(True)
        self.plot.setLabel("bottom", "X")
        self.plot.setLabel("left", "Y")
        layout.addWidget(self.plot, stretch=1)

        self.base_item = pg.ImageItem(np.zeros_like(self.bg))
        self.skin_item = pg.ImageItem(rgba_overlay_from_mask(self.skin_xy, (0, 255, 0), alpha=30))
        self.hair_item = pg.ImageItem(rgba_overlay_from_mask(self.hairs_xy, (0, 0, 255), alpha=120))
        self.vessel_item = pg.ImageItem(rgba_overlay_from_mask(self.vessels_xy, (255, 0, 0), alpha=90))

        self.plot.addItem(self.base_item)
        self.plot.addItem(self.skin_item)
        self.plot.addItem(self.hair_item)
        self.plot.addItem(self.vessel_item)

        vb = self.plot.getViewBox()
        vb.setLimits(
            xMin=0, xMax=self.nx,
            yMin=0, yMax=self.ny,
            maxXRange=self.nx,
            maxYRange=self.ny,
            minXRange=20,
            minYRange=20
        )

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.cb_bg = QtWidgets.QCheckBox("Fond")
        self.cb_bg.setChecked(True)
        self.cb_skin = QtWidgets.QCheckBox("Peau")
        self.cb_skin.setChecked(True)
        self.cb_hair = QtWidgets.QCheckBox("Poils sûrs")
        self.cb_hair.setChecked(True)
        self.cb_vessel = QtWidgets.QCheckBox("Vaisseaux")
        self.cb_vessel.setChecked(True)

        self.btn_reset = QtWidgets.QPushButton("Reset vue")

        for w in [self.cb_bg, self.cb_skin, self.cb_hair, self.cb_vessel, self.btn_reset]:
            controls.addWidget(w)

        self.cb_bg.stateChanged.connect(self._update)
        self.cb_skin.stateChanged.connect(self._update)
        self.cb_hair.stateChanged.connect(self._update)
        self.cb_vessel.stateChanged.connect(self._update)
        self.btn_reset.clicked.connect(self._reset)

    def _update(self):
        self.show_bg = self.cb_bg.isChecked()
        self.show_skin = self.cb_skin.isChecked()
        self.show_hairs = self.cb_hair.isChecked()
        self.show_vessels = self.cb_vessel.isChecked()

        self.base_item.setImage(self.bg if self.show_bg else np.zeros_like(self.bg), autoLevels=False)
        self.skin_item.setVisible(self.show_skin)
        self.hair_item.setVisible(self.show_hairs)
        self.vessel_item.setVisible(self.show_vessels)

        self.label.setText(
            f"Peau={np.count_nonzero(self.skin_xy)} px | "
            f"Poils={np.count_nonzero(self.hairs_xy)} px | "
            f"Vaisseaux={np.count_nonzero(self.vessels_xy)} px"
        )

    def _reset(self):
        self.plot.getViewBox().setRange(xRange=(0, self.nx), yRange=(0, self.ny), padding=0.0)


def main(args):
    volume = load_npz_array(args.fichier_volume, key=args.cle_volume)
    skin = np.load(args.fichier_peau)
    hairs = np.load(args.fichier_poils)
    vessels = np.load(args.fichier_vaisseaux)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    pg.setConfigOptions(imageAxisOrder='row-major')

    win = Viewer(volume, skin, hairs, vessels)
    win.resize(1300, 950)
    win.show()
    app.exec_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fichier_volume", type=str, default=r"dicom_data/dicom/4261_fromdcm.npz")
    parser.add_argument("--cle_volume", type=str, default="volume")
    parser.add_argument("--fichier_peau", type=str, default=r"pipeline/body_hair_extraction_methods/surface_peau.npy")
    parser.add_argument("--fichier_poils", type=str, default=r"pipeline/body_hair_extraction_methods/masque_poils_surs.npy")
    parser.add_argument("--fichier_vaisseaux", type=str, default=r"pipeline/body_hair_extraction_methods/masque_vaisseaux_potentiels.npy")
    args = parser.parse_args()
    main(args)