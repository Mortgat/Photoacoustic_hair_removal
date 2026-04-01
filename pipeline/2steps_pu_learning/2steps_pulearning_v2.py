import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal.windows import hann

# ==========================================
# CONFIGURATION V5
# ==========================================
FICHIER_VOL = r"dicom_data/dicom/4261_fromdcm.npz"
DOSSIER_MASQUES = r"pipeline/2steps_pu_learning"
DOSSIER_MODELE = r"pipeline/modele_pu"
DZ_MM = 0.125
ZONE_CONFLIT_MAX_MM = 5.0
PROFONDEUR_EASY_NEG_MM = 1.5 

EPOCHS = 15 
LR = 1e-4

# Champ récepteur élargi pour capturer les bifurcations
PATCH_XY = 96
PATCH_Z = 24

# ==========================================
# FONCTIONS DE PERTE
# ==========================================
class PUFocalLoss(nn.Module):
    def __init__(self, alpha=0.98, gamma=2.0):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, logits, targets, valid_mask):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        return (focal_loss * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)

class ConditionalDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask):
        probs = torch.sigmoid(logits)
        probs_valid = probs * valid_mask
        targets_valid = targets * valid_mask
        if targets_valid.sum() < 1.0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        intersection = (probs_valid * targets_valid).sum()
        union = probs_valid.sum() + targets_valid.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

# ==========================================
# ARCHITECTURE U-NET 3D (2 Canaux)
# ==========================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), groups=8):
        super().__init__()
        pad = tuple(k // 2 for k in kernel_size)
        g = min(groups, out_c)
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(g, out_c), nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(g, out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class StandardUNet3DAniso(nn.Module):
    def __init__(self, in_channels=2, base=32): 
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.enc4 = ConvBlock3D(base * 4, base * 8)
        
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        d3 = self.up3(e4)
        if d3.shape[-3:] != e3.shape[-3:]: d3 = F.interpolate(d3, size=e3.shape[-3:])
        d3 = self.dec3(torch.cat([e3, d3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[-3:] != e2.shape[-3:]: d2 = F.interpolate(d2, size=e2.shape[-3:])
        d2 = self.dec2(torch.cat([e2, d2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]: d1 = F.interpolate(d1, size=e1.shape[-3:])
        return self.out(self.dec1(torch.cat([e1, d1], dim=1)))

# ==========================================
# DATALOADER STRICT
# ==========================================
class PUStrictDataset(Dataset):
    def __init__(self, vol, dist, pos, hn, vs, easy_neg, p_xy=PATCH_XY, p_z=PATCH_Z, iters=400):
        self.vol, self.dist = vol, dist
        self.pos, self.hn, self.vs, self.easy_neg = pos, hn, vs, easy_neg
        self.p_xy, self.p_z, self.iters = p_xy, p_z, iters
        self.nx, self.ny, self.nz = vol.shape
        
        self.pos_coords = np.argwhere(self.pos)
        self.hn_coords = np.argwhere(self.hn)
        self.vs_coords = np.argwhere(self.vs)
        self.en_coords = np.argwhere(self.easy_neg)

    def __len__(self): return self.iters

    def __getitem__(self, idx):
        b_x, b_y, b_val = [], [], []
        
        def extract(c_list):
            if len(c_list) == 0: return self.nx//2, self.ny//2, self.nz//2
            return c_list[np.random.randint(len(c_list))]

        def get_patch(x, y, z):
            x0 = np.clip(x - self.p_xy//2, 0, self.nx - self.p_xy)
            y0 = np.clip(y - self.p_xy//2, 0, self.ny - self.p_xy)
            z0 = np.clip(z - self.p_z//2, 0, self.nz - self.p_z)
            xs, ys, zs = slice(x0, x0+self.p_xy), slice(y0, y0+self.p_xy), slice(z0, z0+self.p_z)
            
            x_p = np.stack([self.vol[xs, ys, zs], self.dist[xs, ys, zs]], axis=0)
            target = np.zeros((1, self.p_xy, self.p_xy, self.p_z), dtype=np.float32)
            valid = np.zeros((1, self.p_xy, self.p_xy, self.p_z), dtype=np.float32)
            
            # Application des masques
            target[0][self.pos[xs, ys, zs]] = 1.0; valid[0][self.pos[xs, ys, zs]] = 1.0
            target[0][self.hn[xs, ys, zs]] = 0.0; valid[0][self.hn[xs, ys, zs]] = 1.0
            target[0][self.vs[xs, ys, zs]] = 0.0; valid[0][self.vs[xs, ys, zs]] = 1.0
            target[0][self.easy_neg[xs, ys, zs]] = 0.0; valid[0][self.easy_neg[xs, ys, zs]] = 1.0
            
            return x_p, target, valid

        # Construction du batch local : 2 Poils, 1 Bruit HN, 1 Vaisseau HN (ou Air)
        for _ in range(2):
            px, py, pz = extract(self.pos_coords)
            dx, dy, dv = get_patch(px, py, pz)
            b_x.append(dx); b_y.append(dy); b_val.append(dv)
            
        hx, hy, hz = extract(self.hn_coords)
        dx, dy, dv = get_patch(hx, hy, hz)
        b_x.append(dx); b_y.append(dy); b_val.append(dv)
        
        if len(self.vs_coords) > 0:
            vx, vy, vz = extract(self.vs_coords)
            dx, dy, dv = get_patch(vx, vy, vz)
        else:
            ex, ey, ez = extract(self.en_coords)
            dx, dy, dv = get_patch(ex, ey, ez)
            
        b_x.append(dx); b_y.append(dy); b_val.append(dv)

        return torch.from_numpy(np.stack(b_x)).float(), torch.from_numpy(np.stack(b_y)).float(), torch.from_numpy(np.stack(b_val)).float()

# ==========================================
# INFÉRENCE GLISSANTE RAM-SAFE
# ==========================================
def get_hann_window_3d(shape):
    wx = hann(shape[0], sym=True)
    wy = hann(shape[1], sym=True)
    wz = hann(shape[2], sym=True)
    return (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float32)

def sliding_window_inference_ram_safe(model, vol, dist, device, p_xy=PATCH_XY, p_z=PATCH_Z):
    model.eval()
    nx, ny, nz = vol.shape
    out_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    weight_map = np.zeros((nx, ny, nz), dtype=np.float32)
    window = get_hann_window_3d((p_xy, p_xy, p_z))
    
    step_xy, step_z = p_xy // 2, p_z // 2
    
    with torch.no_grad():
        for x in tqdm(range(0, max(1, nx - p_xy + 1), step_xy), desc="Inférence X"):
            for y in range(0, max(1, ny - p_xy + 1), step_xy):
                for z in range(0, max(1, nz - p_z + 1), step_z):
                    x0, y0, z0 = min(x, nx - p_xy), min(y, ny - p_xy), min(z, nz - p_z)
                    xs, ys, zs = slice(x0, x0+p_xy), slice(y0, y0+p_xy), slice(z0, z0+p_z)
                    
                    patch_in = np.stack([vol[xs, ys, zs], dist[xs, ys, zs]], axis=0)
                    tensor_in = torch.from_numpy(patch_in).unsqueeze(0).float().to(device)
                    
                    probs = torch.sigmoid(model(tensor_in)).squeeze().cpu().numpy()
                    out_mask[xs, ys, zs] += probs * window
                    weight_map[xs, ys, zs] += window

    weight_map[weight_map == 0] = 1.0
    out_mask /= weight_map 
    
    del weight_map, window; gc.collect()
    out_mask_f16 = out_mask.astype(np.float16)
    del out_mask; gc.collect()
    
    return out_mask_f16

# ==========================================
# BOUCLE PRINCIPALE
# ==========================================
def main():
    os.makedirs(DOSSIER_MODELE, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vol = np.load(FICHIER_VOL)["volume"].astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8) 
    
    surf = np.load(os.path.join(DOSSIER_MASQUES, "surface_peau.npy"))
    pos = np.load(os.path.join(DOSSIER_MASQUES, "masque_poils_surs.npy"))
    hn = np.load(os.path.join(DOSSIER_MASQUES, "masque_hard_negatives.npy"))
    
    f_vs = os.path.join(DOSSIER_MASQUES, "masque_vaisseaux_surs.npy")
    vs = np.load(f_vs) if os.path.exists(f_vs) else np.zeros_like(pos)

    nz = vol.shape[2]
    surf_propre = np.nan_to_num(surf, nan=nz) 
    dist_mm = (np.arange(nz).reshape(1, 1, nz) - surf_propre[:, :, None]) * DZ_MM
    dist_mm = dist_mm.astype(np.float32)
    
    easy_neg = dist_mm > PROFONDEUR_EASY_NEG_MM
    dist_norm = np.clip(dist_mm / ZONE_CONFLIT_MAX_MM, -1.0, 1.0)
    
    del surf_propre, dist_mm; gc.collect()

    dataset = PUStrictDataset(vol, dist_norm, pos, hn, vs, easy_neg)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = StandardUNet3DAniso(in_channels=2, base=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    criterion_focal = PUFocalLoss(alpha=0.98, gamma=2.0)
    criterion_dice = ConditionalDiceLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y, batch_valid in pbar:
            batch_x = batch_x.squeeze(0).to(device)
            batch_y = batch_y.squeeze(0).to(device)
            batch_valid = batch_valid.squeeze(0).to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            
            loss_focal = criterion_focal(logits, batch_y, batch_valid)
            loss_dice = criterion_dice(logits, batch_y, batch_valid)
            loss = loss_focal + (0.5 * loss_dice)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
    torch.save(model.state_dict(), os.path.join(DOSSIER_MODELE, "unet_pu_weights.pth"))

    del dataset, loader, optimizer, loss, logits; gc.collect()
    torch.cuda.empty_cache()

    masque_predit = sliding_window_inference_ram_safe(model, vol, dist_norm, device)
    np.save(os.path.join(DOSSIER_MASQUES, "prediction_pu.npy"), masque_predit)

if __name__ == "__main__":
    main()