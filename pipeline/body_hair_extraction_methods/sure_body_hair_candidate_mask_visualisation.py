import os
import argparse
import numpy as np
import pyqtgraph as pg

from qtpy import QtWidgets, QtCore


def load_npz_array(npz_path, key="volume"):
    data = np.load(npz_path)
    if key not in data:
        raise KeyError(
            f"Clé '{key}' introuvable dans {npz_path}. "
            f"Clés disponibles: {list(data.keys())}"
        )
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


class TopXYViewer(QtWidgets.QWidget):
    def __init__(
        self,
        volume,
        surface_z,
        masque_positifs,
        projection_mode="max",
        opacity_poils=120,
        opacity_peau=40,
    ):
        super().__init__()

        self.setWindowTitle("Validation du masque PU — vue de dessus XY")

        self.volume = volume
        self.surface_z = surface_z.astype(np.float32)
        self.masque_positifs = masque_positifs > 0

        self.nx, self.ny, self.nz = self.volume.shape

        if self.surface_z.shape != (self.nx, self.ny):
            raise ValueError(
                f"surface_z.shape={self.surface_z.shape} incompatible avec volume.shape[:2]={self.volume.shape[:2]}"
            )
        if self.masque_positifs.shape != self.volume.shape:
            raise ValueError(
                f"masque_positifs.shape={self.masque_positifs.shape} incompatible avec volume.shape={self.volume.shape}"
            )

        profondeur_max = np.nanmax(self.surface_z)
        self.surface_z = self.surface_z.copy()
        self.surface_z[self.surface_z == profondeur_max] = np.nan
        self.skin_valid = ~np.isnan(self.surface_z)

        self.opacity_poils = opacity_poils
        self.opacity_peau = opacity_peau
        self.show_background = True

        if projection_mode == "max":
            top_xy = np.max(self.volume, axis=2)   # (X, Y)
        elif projection_mode == "mean":
            top_xy = np.mean(self.volume, axis=2)
        else:
            raise ValueError("projection_mode doit être 'max' ou 'mean'")

        self.top_xy_u8 = normalize_to_uint8(top_xy).T          # (Y, X)
        self.poils_xy = np.any(self.masque_positifs, axis=2).T # (Y, X)
        self.skin_xy = self.skin_valid.T                       # (Y, X)

        self.init_ui()
        self.update_view()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.checkbox_bg = QtWidgets.QCheckBox("Fond ON/OFF")
        self.checkbox_bg.setChecked(True)
        controls.addWidget(self.checkbox_bg)

        self.btn_reset = QtWidgets.QPushButton("Reset vue")
        controls.addWidget(self.btn_reset)

        controls.addStretch()

        self.plot.setLabel('bottom', 'X')
        self.plot.setLabel('left', 'Y')
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(True)

        vb = self.plot.getViewBox()
        vb.setDefaultPadding(0.0)

        self.base_item = pg.ImageItem(np.zeros_like(self.top_xy_u8, dtype=np.uint8))
        self.plot.addItem(self.base_item)

        self.skin_item = pg.ImageItem(
            rgba_overlay_from_mask(self.skin_xy, (0, 255, 0), alpha=self.opacity_peau)
        )
        self.plot.addItem(self.skin_item)

        self.poils_item = pg.ImageItem(
            rgba_overlay_from_mask(self.poils_xy, (0, 0, 255), alpha=self.opacity_poils)
        )
        self.plot.addItem(self.poils_item)

        vb.setLimits(
            xMin=0,
            xMax=self.nx,
            yMin=0,
            yMax=self.ny,
            maxXRange=self.nx,
            maxYRange=self.ny,
            minXRange=20,
            minYRange=20
        )

        self.checkbox_bg.stateChanged.connect(self.on_bg_changed)
        self.btn_reset.clicked.connect(self.reset_view)

    def get_background(self):
        if self.show_background:
            return self.top_xy_u8
        return np.zeros_like(self.top_xy_u8, dtype=np.uint8)

    def update_view(self):
        self.base_item.setImage(self.get_background(), autoLevels=False)

        nb_poils_xy = int(np.count_nonzero(self.poils_xy))
        nb_skin_xy = int(np.count_nonzero(self.skin_xy))

        self.label.setText(
            f"Vue de dessus XY projetée sur Z — Peau valide={nb_skin_xy} px | "
            f"Positifs projetés={nb_poils_xy} px"
        )

    def on_bg_changed(self, state):
        self.show_background = self.checkbox_bg.isChecked()
        self.update_view()

    def reset_view(self):
        vb = self.plot.getViewBox()
        vb.setRange(
            xRange=(0, self.nx),
            yRange=(0, self.ny),
            padding=0.0
        )


def main(args):
    print("--- VIEWER XY UNIQUEMENT POUR VALIDATION DU MASQUE PU ---")

    if not os.path.exists(args.fichier_volume):
        raise FileNotFoundError(f"Volume introuvable : {args.fichier_volume}")
    if not os.path.exists(args.fichier_poils):
        raise FileNotFoundError(f"Masque poils introuvable : {args.fichier_poils}")
    if not os.path.exists(args.fichier_peau):
        raise FileNotFoundError(f"Surface peau introuvable : {args.fichier_peau}")

    print("Chargement des données...")
    volume = load_npz_array(args.fichier_volume, key=args.cle_volume)
    masque_positifs = np.load(args.fichier_poils)
    surface_z = np.load(args.fichier_peau)

    print(f"volume.shape         = {volume.shape}")
    print(f"masque_positifs.shape= {masque_positifs.shape}")
    print(f"surface_z.shape      = {surface_z.shape}")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    pg.setConfigOptions(imageAxisOrder='row-major')

    win = TopXYViewer(
        volume=volume,
        surface_z=surface_z,
        masque_positifs=masque_positifs,
        projection_mode=args.projection_mode,
        opacity_poils=args.opacity_poils,
        opacity_peau=args.opacity_peau,
    )

    win.resize(1200, 900)
    win.show()
    app.exec_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vue de dessus XY projetée sur Z pour valider le masque PU."
    )

    parser.add_argument(
        "--fichier_volume",
        type=str,
        default=r"dicom_data/dicom/4261_fromdcm.npz"
    )
    parser.add_argument(
        "--cle_volume",
        type=str,
        default="volume"
    )
    parser.add_argument(
        "--fichier_poils",
        type=str,
        default=r"pipeline/body_hair_extraction_methods/masque_poils_surs_v3.npy"
    )
    parser.add_argument(
        "--fichier_peau",
        type=str,
        default=r"pipeline/body_hair_extraction_methods/surface_peau.npy"
    )
    parser.add_argument(
        "--projection_mode",
        type=str,
        choices=["max", "mean"],
        default="max"
    )
    parser.add_argument(
        "--opacity_poils",
        type=int,
        default=120
    )
    parser.add_argument(
        "--opacity_peau",
        type=int,
        default=40
    )

    args = parser.parse_args()
    main(args)