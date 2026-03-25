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


def robust_limits(img, low=1, high=99):
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def normalize_to_uint8(img, low=1, high=99):
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (255 * out).astype(np.uint8)


def rgba_overlay_from_mask(mask_2d, color_rgb, alpha=80):
    """
    mask_2d: shape (H, W), bool ou 0/1
    color_rgb: tuple 0..255
    retourne RGBA uint8
    """
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.where(mask_2d, alpha, 0).astype(np.uint8)
    return rgba


def build_skin_xz_mask(surface_z, y, nz):
    """
    surface_z shape = (X, Y)
    Retourne un masque 2D shape = (Z, X)
    avec un pixel à z = surface_z[x, y] pour chaque x valide.
    """
    nx = surface_z.shape[0]
    mask = np.zeros((nz, nx), dtype=bool)

    z_col = surface_z[:, y]
    valid = ~np.isnan(z_col)
    xs = np.where(valid)[0]
    zs = np.rint(z_col[valid]).astype(np.int32)
    zs = np.clip(zs, 0, nz - 1)

    mask[zs, xs] = True
    return mask


class HairInspectionApp(QtWidgets.QWidget):
    def __init__(
        self,
        volume,
        surface_z,
        masque_predit,
        masque_surs=None,
        projection_mode="max",
        opacity_pred=90,
        opacity_sure=120,
        opacity_skin=220,
    ):
        super().__init__()

        self.setWindowTitle("Inspection XZ par Y + vue dessus XY")

        self.volume = volume
        self.surface_z = surface_z.astype(np.float32)
        self.masque_predit = masque_predit > 0
        self.masque_surs = (masque_surs > 0) if masque_surs is not None else None

        self.nx, self.ny, self.nz = self.volume.shape

        if self.surface_z.shape != (self.nx, self.ny):
            raise ValueError(
                f"surface_z.shape={self.surface_z.shape} incompatible avec volume.shape[:2]={self.volume.shape[:2]}"
            )
        if self.masque_predit.shape != self.volume.shape:
            raise ValueError(
                f"masque_predit.shape={self.masque_predit.shape} incompatible avec volume.shape={self.volume.shape}"
            )
        if self.masque_surs is not None and self.masque_surs.shape != self.volume.shape:
            raise ValueError(
                f"masque_surs.shape={self.masque_surs.shape} incompatible avec volume.shape={self.volume.shape}"
            )

        # Nettoyage du fond de surface_z
        profondeur_max = np.nanmax(self.surface_z)
        self.surface_z = self.surface_z.copy()
        self.surface_z[self.surface_z == profondeur_max] = np.nan
        self.skin_valid = ~np.isnan(self.surface_z)

        self.opacity_pred = opacity_pred
        self.opacity_sure = opacity_sure
        self.opacity_skin = opacity_skin

        self.current_y = self.ny // 2
        self.show_background = True

        self.vmin, self.vmax = robust_limits(self.volume)

        # Projection XY fixe
        if projection_mode == "max":
            top_xy = np.max(self.volume, axis=2)   # (X, Y)
        elif projection_mode == "mean":
            top_xy = np.mean(self.volume, axis=2)
        else:
            raise ValueError("projection_mode doit être 'max' ou 'mean'")

        self.top_xy_u8 = normalize_to_uint8(top_xy).T
        self.pred_xy = np.any(self.masque_predit, axis=2).T
        self.sure_xy = np.any(self.masque_surs, axis=2).T if self.masque_surs is not None else None
        self.skin_xy = self.skin_valid.T

        self.init_ui()
        self.init_top_view()
        self.init_xz_view()
        self.update_all()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        # Panneau gauche : XZ
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        self.label_xz = QtWidgets.QLabel("Coupe XZ")
        self.label_xz.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.label_xz)

        self.plot_xz = pg.PlotWidget()
        left_layout.addWidget(self.plot_xz, stretch=1)

        split.addWidget(left_widget)

        # Panneau droit : XY
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        self.label_xy = QtWidgets.QLabel("Vue de dessus XY")
        self.label_xy.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.label_xy)

        self.plot_xy = pg.PlotWidget()
        right_layout.addWidget(self.plot_xy, stretch=1)

        split.addWidget(right_widget)
        split.setSizes([1100, 700])

        # Contrôles
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QtWidgets.QLabel("Y"))

        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y.setMinimum(0)
        self.slider_y.setMaximum(self.ny - 1)
        self.slider_y.setValue(self.current_y)
        controls.addWidget(self.slider_y, stretch=1)

        self.label_y_value = QtWidgets.QLabel(str(self.current_y))
        controls.addWidget(self.label_y_value)

        self.checkbox_fit_xz = QtWidgets.QCheckBox("Déformer XZ pour remplir")
        self.checkbox_fit_xz.setChecked(True)
        controls.addWidget(self.checkbox_fit_xz)

        self.checkbox_bg = QtWidgets.QCheckBox("Fond ON/OFF")
        self.checkbox_bg.setChecked(True)
        controls.addWidget(self.checkbox_bg)

        self.btn_reset_xz = QtWidgets.QPushButton("Reset vue XZ")
        controls.addWidget(self.btn_reset_xz)

        self.btn_reset_xy = QtWidgets.QPushButton("Reset vue XY")
        controls.addWidget(self.btn_reset_xy)

        self.slider_y.valueChanged.connect(self.on_y_changed)
        self.checkbox_fit_xz.stateChanged.connect(self.on_fit_xz_changed)
        self.checkbox_bg.stateChanged.connect(self.on_bg_changed)
        self.btn_reset_xz.clicked.connect(self.reset_xz_view)
        self.btn_reset_xy.clicked.connect(self.reset_xy_view)

    def init_top_view(self):
        self.plot_xy.setLabel('bottom', 'X')
        self.plot_xy.setLabel('left', 'Y')
        self.plot_xy.showGrid(x=False, y=False)

        vb = self.plot_xy.getViewBox()
        vb.setDefaultPadding(0.0)

        # Convention image : y=0 en haut
        self.plot_xy.invertY(True)

        self.xy_base_item = pg.ImageItem(self.get_xy_base())
        self.plot_xy.addItem(self.xy_base_item)

        self.xy_skin_item = pg.ImageItem(
            rgba_overlay_from_mask(self.skin_xy, (0, 255, 0), alpha=35)
        )
        self.plot_xy.addItem(self.xy_skin_item)

        self.xy_pred_item = pg.ImageItem(
            rgba_overlay_from_mask(self.pred_xy, (255, 0, 0), alpha=self.opacity_pred)
        )
        self.plot_xy.addItem(self.xy_pred_item)

        self.xy_sure_item = None
        if self.sure_xy is not None:
            self.xy_sure_item = pg.ImageItem(
                rgba_overlay_from_mask(self.sure_xy, (0, 0, 255), alpha=self.opacity_sure)
            )
            self.plot_xy.addItem(self.xy_sure_item)

        self.xy_y_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen((255, 255, 0), width=2)
        )
        self.plot_xy.addItem(self.xy_y_line)

        self.plot_xy.scene().sigMouseClicked.connect(self.on_xy_mouse_clicked)

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

    def init_xz_view(self):
        self.plot_xz.setLabel('bottom', 'X')
        self.plot_xz.setLabel('left', 'Z')
        self.plot_xz.showGrid(x=False, y=False)

        vb = self.plot_xz.getViewBox()
        vb.setDefaultPadding(0.0)

        # Convention photoacoustique : z=0 en haut
        self.plot_xz.invertY(True)

        empty_base = np.zeros((self.nz, self.nx), dtype=np.uint8)
        self.xz_base_item = pg.ImageItem(empty_base)
        self.plot_xz.addItem(self.xz_base_item)

        self.xz_skin_item = pg.ImageItem(np.zeros((self.nz, self.nx, 4), dtype=np.uint8))
        self.plot_xz.addItem(self.xz_skin_item)

        self.xz_pred_item = pg.ImageItem(np.zeros((self.nz, self.nx, 4), dtype=np.uint8))
        self.plot_xz.addItem(self.xz_pred_item)

        self.xz_sure_item = pg.ImageItem(np.zeros((self.nz, self.nx, 4), dtype=np.uint8))
        self.plot_xz.addItem(self.xz_sure_item)

        vb.setLimits(
            xMin=0,
            xMax=self.nx,
            yMin=0,
            yMax=self.nz,
            maxXRange=self.nx,
            maxYRange=self.nz,
            minXRange=20,
            minYRange=5
        )

    def get_xz_base(self, y):
        slice_xz = self.volume[:, y, :].T  # (Z, X)
        if self.show_background:
            return normalize_to_uint8(slice_xz)
        return np.zeros_like(slice_xz, dtype=np.uint8)

    def get_xy_base(self):
        if self.show_background:
            return self.top_xy_u8
        return np.zeros_like(self.top_xy_u8, dtype=np.uint8)

    def get_xz_pred_mask(self, y):
        return self.masque_predit[:, y, :].T  # (Z, X)

    def get_xz_sure_mask(self, y):
        if self.masque_surs is None:
            return None
        return self.masque_surs[:, y, :].T

    def get_xz_skin_mask(self, y):
        return build_skin_xz_mask(self.surface_z, y, self.nz)

    def update_xz_view(self):
        y = self.current_y

        base = self.get_xz_base(y)
        pred = self.get_xz_pred_mask(y)
        sure = self.get_xz_sure_mask(y)
        skin = self.get_xz_skin_mask(y)

        self.xz_base_item.setImage(base, autoLevels=False)
        self.xz_pred_item.setImage(
            rgba_overlay_from_mask(pred, (255, 0, 0), alpha=self.opacity_pred),
            autoLevels=False
        )
        self.xz_skin_item.setImage(
            rgba_overlay_from_mask(skin, (0, 255, 0), alpha=self.opacity_skin),
            autoLevels=False
        )

        if sure is not None:
            self.xz_sure_item.setImage(
                rgba_overlay_from_mask(sure, (0, 0, 255), alpha=self.opacity_sure),
                autoLevels=False
            )
        else:
            self.xz_sure_item.setImage(
                np.zeros((self.nz, self.nx, 4), dtype=np.uint8),
                autoLevels=False
            )

        n_pred = int(np.count_nonzero(pred))
        n_sure = int(np.count_nonzero(sure)) if sure is not None else 0
        n_skin = int(np.count_nonzero(skin))

        self.label_xz.setText(
            f"Coupe XZ à y={y} — Peau={n_skin} px | Sûrs={n_sure} px | Prédits={n_pred} px"
        )

    def update_xy_view(self):
        y = self.current_y
        self.xy_base_item.setImage(self.get_xy_base(), autoLevels=False)
        self.xy_y_line.setPos(y)
        self.label_xy.setText(f"Vue de dessus XY projetée sur Z — coupe XZ à y={y}")

    def update_all(self):
        self.update_xz_view()
        self.update_xy_view()
        self.label_y_value.setText(str(self.current_y))

    def on_y_changed(self, value):
        self.current_y = int(value)
        self.update_all()

    def on_fit_xz_changed(self, state):
        fit = self.checkbox_fit_xz.isChecked()

        # True => on autorise la déformation pour remplir
        # False => aspect 1:1
        self.plot_xz.setAspectLocked(not fit, ratio=1)
        self.reset_xz_view()

    def on_bg_changed(self, state):
        self.show_background = self.checkbox_bg.isChecked()
        self.update_xz_view()
        self.update_xy_view()

    def reset_xz_view(self):
        vb = self.plot_xz.getViewBox()
        vb.setRange(
            xRange=(0, self.nx),
            yRange=(0, self.nz),
            padding=0.0
        )

    def reset_xy_view(self):
        vb = self.plot_xy.getViewBox()
        vb.setRange(
            xRange=(0, self.nx),
            yRange=(0, self.ny),
            padding=0.0
        )

    def on_xy_mouse_clicked(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return

        scene_pos = event.scenePos()
        vb = self.plot_xy.getViewBox()
        mouse_point = vb.mapSceneToView(scene_pos)

        x = mouse_point.x()
        y = mouse_point.y()

        if 0 <= x < self.nx and 0 <= y < self.ny:
            self.slider_y.setValue(int(round(y)))


def main(args):
    print("--- VIEWER XZ / XY AVEC PYQTGRAPH ---")

    if not os.path.exists(args.fichier_volume_original):
        raise FileNotFoundError(f"Volume original introuvable : {args.fichier_volume_original}")
    if not os.path.exists(args.fichier_peau):
        raise FileNotFoundError(f"Surface peau introuvable : {args.fichier_peau}")
    if not os.path.exists(args.fichier_poils_predits):
        raise FileNotFoundError(f"Masque poils prédits introuvable : {args.fichier_poils_predits}")

    print("Chargement des données...")
    volume = load_npz_array(args.fichier_volume_original, key=args.cle_volume)
    surface_z = np.load(args.fichier_peau)
    masque_predit = np.load(args.fichier_poils_predits)

    masque_surs = None
    if args.fichier_poils_surs and os.path.exists(args.fichier_poils_surs):
        masque_surs = np.load(args.fichier_poils_surs)

    print(f"volume.shape        = {volume.shape}")
    print(f"surface_z.shape     = {surface_z.shape}")
    print(f"masque_predit.shape = {masque_predit.shape}")
    if masque_surs is not None:
        print(f"masque_surs.shape   = {masque_surs.shape}")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    pg.setConfigOptions(imageAxisOrder='row-major')

    win = HairInspectionApp(
        volume=volume,
        surface_z=surface_z,
        masque_predit=masque_predit,
        masque_surs=masque_surs,
        projection_mode=args.projection_mode,
        opacity_pred=args.opacity_predits,
        opacity_sure=args.opacity_surs,
        opacity_skin=args.opacity_peau,
    )

    win.resize(1700, 900)
    win.show()

    app.exec_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspection interactive : coupe XZ par Y + vue XY projetée"
    )

    parser.add_argument(
        "--fichier_volume_original",
        type=str,
        default=r"dicom_data/dicom/4261_fromdcm.npz"
    )
    parser.add_argument(
        "--cle_volume",
        type=str,
        default="volume"
    )
    parser.add_argument(
        "--fichier_peau",
        type=str,
        default=r"pipeline/body_hair_extraction_methods/surface_peau.npy"
    )
    parser.add_argument(
        "--fichier_poils_predits",
        type=str,
        default=r"pipeline/body_hair_extraction_methods/masque_predit_hybrid_v2.npy"
    )
    parser.add_argument(
        "--fichier_poils_surs",
        type=str,
        default=r"pipeline/body_hair_extraction_methods/masque_poils_surs.npy"
    )
    parser.add_argument(
        "--projection_mode",
        type=str,
        choices=["max", "mean"],
        default="max"
    )
    parser.add_argument(
        "--opacity_predits",
        type=int,
        default=90
    )
    parser.add_argument(
        "--opacity_surs",
        type=int,
        default=120
    )
    parser.add_argument(
        "--opacity_peau",
        type=int,
        default=220
    )

    args = parser.parse_args()
    main(args)