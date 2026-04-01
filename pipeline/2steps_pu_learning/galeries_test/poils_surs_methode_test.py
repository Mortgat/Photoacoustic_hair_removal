import os
import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy import QtWidgets, QtCore
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.measure import label, regionprops

# ==========================================
# CONFIGURATION 
# ==========================================
FICHIER_BRUT = r"dicom_data/dicom/4261_fromdcm.npz"
DOSSIER_MASQUES = r"pipeline/2steps_pu_learning"
FICHIER_PEAU = os.path.join(DOSSIER_MASQUES, "surface_peau.npy")

DX_MM, DY_MM, DZ_MM = 0.125, 0.125, 0.125

# --- SEUILS POILS (Nouvelle Méthode) ---
PERCENTILE_POILS_INT_GRAINE = 95.0    
PERCENTILE_POILS_INT_TEST = 75.0      
PERCENTILE_FRANGI_POILS = 75.0        

LONGUEUR_MIN_POIL_MM = 0.8          
LIMITE_LONGUEUR_POIL_MM = 5.0       
OFFSET_PEAU_STANDARD_VOXELS = 0     
PROFONDEUR_FORET_MAX_MM = 1.5       

# --- SEUILS VAISSEAUX (Ancienne Méthode) ---
PERCENTILE_FRANGI_VAISSEAUX = 90.0    
LONGUEUR_MIN_VAISSEAU_MM = 1.5      
PROFONDEUR_MIN_VAISSEAU_MM = 1.0      # Rejet strict de la surface
RATIO_PLATITUDE_MAX = 0.3           

# --- RECONSTRUCTION 3D (Z-EXPANSION POUR TOUS) ---
Z_EXPANSION_MAX_VOXELS = 10         
Z_DIFF_INTENSITE_MAX = 0.30         

# ==========================================
# CALCULATEUR DES MASQUES
# ==========================================
def calculer_nouveaux_masques(vol, surf):
    print("⚙️ Calcul des primitives (MIP, Frangi)...")
    nx, ny, nz = vol.shape
    mip_brut = np.max(vol, axis=2)
    z_max = np.argmax(vol, axis=2)
    
    lo, hi = np.percentile(mip_brut, 1), np.percentile(mip_brut, 99)
    mip_norm = np.clip((mip_brut - lo) / (hi - lo), 0, 1)
    
    f2d = frangi(mip_norm, sigmas=[0.5, 1.0, 1.5], black_ridges=False)
    f2d = (f2d - f2d.min()) / (f2d.max() - f2d.min() + 1e-8)
    
    seuil_p_graine = np.percentile(mip_norm, PERCENTILE_POILS_INT_GRAINE)
    seuil_p_test = np.percentile(mip_norm, PERCENTILE_POILS_INT_TEST)
    seuil_f_poils = np.percentile(f2d, PERCENTILE_FRANGI_POILS)
    seuil_f_vs = np.percentile(f2d, PERCENTILE_FRANGI_VAISSEAUX)

    # ---------------------------------------------------------
    # 1. GÉNÉRATION DES POILS (Nouvelle méthode avec Hystérésis)
    # ---------------------------------------------------------
    print("🟢 Extraction des Poils (Nouvelle Méthode : Forêt & Hystérésis)...")
    graines_poils = (mip_norm > seuil_p_graine) & (f2d > seuil_f_poils)
    masque_test_poils = (mip_norm > seuil_p_test) & (f2d > seuil_f_poils)
    
    poils_2d = np.zeros_like(mip_norm, dtype=bool)
    labels_test_poils = label(masque_test_poils)
    
    for reg in regionprops(labels_test_poils):
        xs, ys = reg.coords[:, 0], reg.coords[:, 1]
        
        if np.any(graines_poils[xs, ys]):
            longueur_test_mm = max(xs.max() - xs.min(), ys.max() - ys.min()) * DX_MM
            
            masque_graines_locales = graines_poils[xs, ys]
            g_xs = xs[masque_graines_locales]
            g_ys = ys[masque_graines_locales]
            
            longueur_graine_mm = max(g_xs.max() - g_xs.min(), g_ys.max() - g_ys.min()) * DX_MM
            
            if longueur_graine_mm >= LONGUEUR_MIN_POIL_MM:
                z_skin = surf[g_xs, g_ys]
                valid = ~np.isnan(z_skin)
                
                if np.any(valid):
                    skin_med = np.median(z_skin[valid])
                    z_med = np.median(z_max[g_xs, g_ys])
                    
                    if z_med <= skin_med + OFFSET_PEAU_STANDARD_VOXELS:
                        poils_2d[g_xs, g_ys] = True
                    elif skin_med < z_med <= skin_med + (PROFONDEUR_FORET_MAX_MM / DZ_MM):
                        if longueur_test_mm <= LIMITE_LONGUEUR_POIL_MM:
                            poils_2d[g_xs, g_ys] = True

    # ---------------------------------------------------------
    # 2. GÉNÉRATION DES VAISSEAUX (Ancienne méthode restaurée)
    # ---------------------------------------------------------
    print("🔴 Extraction des Vaisseaux Sûrs (Ancienne Méthode)...")
    vaisseaux_2d = np.zeros_like(mip_norm, dtype=bool)
    
    cand_vs = (f2d > seuil_f_vs) & (~poils_2d)
    labels_vs = label(cand_vs)
    profondeur_min_voxels = PROFONDEUR_MIN_VAISSEAU_MM / DZ_MM
    
    for reg in regionprops(labels_vs):
        if reg.major_axis_length >= (LONGUEUR_MIN_VAISSEAU_MM / DX_MM):
            xs, ys = reg.coords[:, 0], reg.coords[:, 1]
            z_vals = z_max[xs, ys]
            
            z_skin_valid = surf[xs, ys]
            mask_nan = ~np.isnan(z_skin_valid)
            
            if np.sum(mask_nan) > 0:
                skin_median = np.median(z_skin_valid[mask_nan])
                z_median = np.median(z_vals[mask_nan])
                
                # Rejet global de la surface par la médiane
                if z_median >= (skin_median + profondeur_min_voxels):
                    dx = xs.max() - xs.min()
                    dy = ys.max() - ys.min()
                    dz = z_vals.max() - z_vals.min()
                    longueur_xy = max(1, max(dx, dy))
                    
                    if (dz / longueur_xy) <= RATIO_PLATITUDE_MAX:
                        vaisseaux_2d[xs, ys] = True

    # ---------------------------------------------------------
    # 3. Z-EXPANSION (Reconstruction volumique pour tous)
    # ---------------------------------------------------------
    print("🧊 Épaississement 3D des masques (avec plafond de verre vasculaire)...")
    def expand_3d(mask_2d, is_vaisseau=False):
        mask_3d = np.zeros_like(vol, dtype=bool)
        xs, ys = np.where(mask_2d)
        
        for x, y in zip(xs, ys):
            zc = z_max[x, y]
            
            # --- LE MUR DE SÉCURITÉ ---
            limite_haute_z = 0
            if is_vaisseau:
                z_skin = surf[x, y]
                if np.isnan(z_skin):
                    continue 
                limite_haute_z = z_skin + (PROFONDEUR_MIN_VAISSEAU_MM / DZ_MM)
                
                if zc < limite_haute_z:
                    continue # On coupe net la racine si le coeur est trop haut
            
            val_base = vol[x, y, zc]
            mask_3d[x, y, zc] = True
            
            for direction in [-1, 1]:
                for step in range(1, Z_EXPANSION_MAX_VOXELS + 1):
                    cz = zc + (direction * step)
                    
                    if 0 <= cz < nz:
                        # Plafond de verre pour la Z-expansion
                        if is_vaisseau and cz < limite_haute_z:
                            continue
                            
                        val_curr = vol[x, y, cz]
                        if abs(val_base - val_curr) / (abs(val_base) + 1e-8) <= Z_DIFF_INTENSITE_MAX:
                            mask_3d[x, y, cz] = True
                        else: 
                            break
        return mask_3d

    pos_mask_3d = expand_3d(poils_2d, is_vaisseau=False)
    vs_mask_3d = expand_3d(vaisseaux_2d, is_vaisseau=True)
    
    return pos_mask_3d, vs_mask_3d, mip_norm

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

# ==========================================
# INTERFACE PRINCIPALE
# ==========================================
class SniperDiagnosticTool(QtWidgets.QWidget):
    def __init__(self, vol, surf, pos_mask, vs_mask, mip_norm):
        super().__init__()
        self.setWindowTitle("Validation Hybride : Poils V6 & Vaisseaux V2.9 (Sécurisés)")
        
        self.vol_brut = vol
        self.surface_peau = surf.astype(np.float32)
        self.pos_mask = pos_mask
        self.vaisseaux_mask = vs_mask
        self.has_vaisseaux = np.any(self.vaisseaux_mask)
        
        self.nx, self.ny, self.nz = self.vol_brut.shape
        self.mip_xy_u8 = (mip_norm * 255).astype(np.uint8).T
        self.pos_xy = np.any(self.pos_mask, axis=2).T
        self.vs_xy = np.any(self.vaisseaux_mask, axis=2).T

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
        
        for item in [self.xy_base, self.xy_vs, self.xy_pos]:
            self.plot_xy.addItem(item)
            
        vb = self.plot_xy.getViewBox()
        vb.setLimits(xMin=0, xMax=self.nx, yMin=0, yMax=self.ny, maxXRange=self.nx, maxYRange=self.ny)
        
        pen_crosshair = pg.mkPen('cyan', width=1, style=QtCore.Qt.DashLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen_crosshair)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen_crosshair)
        self.plot_xy.addItem(self.v_line); self.plot_xy.addItem(self.h_line)
        self.v_line.setPos(-100); self.h_line.setPos(-100)
        
        self.plot_xy.scene().sigMouseClicked.connect(self.on_xy_mouse_clicked)
        
        controls = QtWidgets.QHBoxLayout()
        left_panel.addLayout(controls)
        
        self.cb_pos = QtWidgets.QCheckBox("Poils Sûrs (Jaune)")
        self.cb_pos.setChecked(True)
        self.cb_vs = QtWidgets.QCheckBox("Vaisseaux Sûrs (Magenta)")
        self.cb_vs.setChecked(True)
        
        controls.addWidget(self.cb_pos); controls.addWidget(self.cb_vs)
        
        if not self.has_vaisseaux:
            self.cb_vs.hide()
            self.cb_vs.setChecked(False)

        self.cb_pos.stateChanged.connect(self.update_mip_display)
        self.cb_vs.stateChanged.connect(self.update_mip_display)

        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)
        
        self.label_3d = QtWidgets.QLabel("Volume 3D Local")
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
        self.info_panel.setStyleSheet("background-color: #222; color: #fff; padding: 10px;")
        self.info_panel.setAlignment(QtCore.Qt.AlignCenter)
        right_panel.addWidget(self.info_panel)

    def update_mip_display(self):
        empty_xy_rgba = np.zeros((self.nx, self.ny, 4), dtype=np.uint8)
        self.xy_base.setImage(self.mip_xy_u8, autoLevels=False)
        self.xy_pos.setImage(rgba_overlay(self.pos_xy, (255, 255, 0), 150) if self.cb_pos.isChecked() else empty_xy_rgba, autoLevels=False)
        self.xy_vs.setImage(rgba_overlay(self.vs_xy, (255, 0, 255), 180) if self.cb_vs.isChecked() else empty_xy_rgba, autoLevels=False)

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
        vol_patch = self.vol_brut[xmin:xmax, ymin:ymax, :]
        skin_patch = self.surface_peau[xmin:xmax, ymin:ymax]
        surs_patch = self.pos_mask[xmin:xmax, ymin:ymax, :] > 0
        vs_patch = self.vaisseaux_mask[xmin:xmax, ymin:ymax, :] > 0
            
        px, py, pz = vol_patch.shape
        target_z = pz // 2

        skin_3d_patch = np.zeros_like(vol_patch, dtype=bool)
        mask_valide = ~np.isnan(skin_patch)
        if mask_valide.any():
            vx, vy = np.where(mask_valide)
            vz = np.clip(skin_patch[vx, vy], 0, pz-1).astype(int)
            skin_3d_patch[vx, vy, vz] = True

        flat_vol, flat_surs, flat_skin, flat_vs = [np.zeros_like(x) for x in (vol_patch, surs_patch, skin_3d_patch, vs_patch)]

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
                    flat_skin[i, j, :] = np.roll(skin_3d_patch[i, j, :], shift)
                    flat_vs[i, j, :] = np.roll(vs_patch[i, j, :], shift)
        else:
            flat_vol, flat_surs, flat_skin, flat_vs = vol_patch.copy(), surs_patch.copy(), skin_3d_patch.copy(), vs_patch.copy()

        flat_vol_pos = np.clip(flat_vol, 0, None)
        local_hi = np.percentile(flat_vol_pos, 99.9)
        if local_hi <= 0: local_hi = 1e-6
        norm_vol = np.clip(flat_vol_pos / local_hi, 0, 1)
        val = (norm_vol * 255).astype(np.ubyte)
        
        rgba = np.zeros((px, py, pz, 4), dtype=np.ubyte)
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = val 
        
        rgba[flat_surs] = [255, 255, 0, 0]
        if self.cb_vs.isChecked(): rgba[flat_vs] = [255, 0, 255, 0]
        rgba[flat_skin] = [0, 255, 0, 0]

        alpha = (norm_vol**2 * 150).astype(np.ubyte) 
        mask_any = flat_surs | flat_skin
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
    print("📂 Chargement des données brutes...")
    vol = np.load(FICHIER_BRUT)["volume"].astype(np.float32)
    surf = np.load(FICHIER_PEAU)

    pos_mask_new, vs_mask_new, mip_norm = calculer_nouveaux_masques(vol, surf)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    win = SniperDiagnosticTool(vol, surf, pos_mask_new, vs_mask_new, mip_norm)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()