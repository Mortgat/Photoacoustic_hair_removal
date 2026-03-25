import os
import math
import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# Utils
# =========================================================

def zscore_norm(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-8)


def build_signed_distance_channel(
    surface_z: np.ndarray,
    shape_3d: Tuple[int, int, int],
    dz_mm: float,
    clip_mm: float = 2.0,
) -> np.ndarray:
    """
    Retourne un canal 3D de distance signée à la peau.
    Convention:
      distance = (z - surface_z[x,y]) * dz_mm
      < 0 : au-dessus de la peau
      > 0 : sous la peau
    Puis clipping et normalisation dans [-1, 1].
    """
    nx, ny, nz = shape_3d
    z_grid = np.arange(nz, dtype=np.float32)[None, None, :]
    surf = surface_z.astype(np.float32)[:, :, None]

    dist_mm = (z_grid - surf) * dz_mm
    dist_mm = np.clip(dist_mm, -clip_mm, clip_mm)
    dist_mm = dist_mm / clip_mm

    # Colonnes invalides codées à nz dans ta pipeline -> distance 0 neutre
    invalid = surface_z >= nz
    if np.any(invalid):
        dist_mm[invalid, :] = 0.0

    return dist_mm.astype(np.float32)


def hann_window_3d(px: int, py: int, pz: int) -> np.ndarray:
    wx = np.hanning(px)
    wy = np.hanning(py)
    wz = np.hanning(pz)
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    w = w.astype(np.float32)
    # éviter les zéros stricts aux bords
    w = np.maximum(w, 1e-3)
    return w


# =========================================================
# Architecture
# =========================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), num_groups=8):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet3DAniso(nn.Module):
    def __init__(self, in_channels=2, base=16):
        super().__init__()

        # Encodeur
        self.enc1 = ConvBlock3D(in_channels, base, kernel_size=(3, 3, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        self.enc2 = ConvBlock3D(base, base * 2, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.enc3 = ConvBlock3D(base * 2, base * 4, kernel_size=(3, 3, 3))

        # Décodeur
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = ConvBlock3D(base * 4, base * 2, kernel_size=(3, 3, 3))

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = ConvBlock3D(base * 2, base, kernel_size=(3, 3, 1))

        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = d2[:, :, :e2.shape[2], :e2.shape[3], :e2.shape[4]]
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = d1[:, :, :e1.shape[2], :e1.shape[3], :e1.shape[4]]
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# =========================================================
# nnPU loss
# =========================================================

class nnPULoss(nn.Module):
    def __init__(self, prior: float):
        super().__init__()
        self.prior = prior
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, labels):
        pos_loss = self.loss_func(predictions, torch.ones_like(predictions))
        unl_loss = self.loss_func(predictions, torch.zeros_like(predictions))
        pos_as_neg_loss = self.loss_func(predictions, torch.zeros_like(predictions))

        pos_mask = (labels == 1).float()
        unl_mask = (labels == 0).float()

        n_pos = torch.clamp(pos_mask.sum(), min=1.0)
        n_unl = torch.clamp(unl_mask.sum(), min=1.0)

        risk_p = (pos_loss * pos_mask).sum() / n_pos
        risk_u = (unl_loss * unl_mask).sum() / n_unl
        risk_p_neg = (pos_as_neg_loss * pos_mask).sum() / n_pos

        risk_n_estim = risk_u - self.prior * risk_p_neg
        risk_n = torch.clamp(risk_n_estim, min=0.0)

        return self.prior * risk_p + risk_n


# =========================================================
# Dataset avec sampling guidé
# =========================================================

class GuidedPatchDataset(Dataset):
    def __init__(
        self,
        volume_norm: np.ndarray,
        dist_channel: np.ndarray,
        mask_pos: np.ndarray,
        surface_z: np.ndarray,
        patch_xy: int,
        patch_z: int,
        iterations: int,
        dz_mm: float,
        positive_patch_ratio: float = 0.5,
        skin_patch_ratio: float = 0.3,
    ):
        self.volume = volume_norm
        self.dist = dist_channel
        self.mask = mask_pos.astype(np.uint8)
        self.surface_z = surface_z.astype(np.float32)

        self.patch_xy = patch_xy
        self.patch_z = patch_z
        self.iterations = iterations
        self.dz_mm = dz_mm

        self.positive_patch_ratio = positive_patch_ratio
        self.skin_patch_ratio = skin_patch_ratio
        self.random_patch_ratio = 1.0 - positive_patch_ratio - skin_patch_ratio

        self.nx, self.ny, self.nz = self.volume.shape

        self.pos_coords = np.argwhere(self.mask > 0)
        self.skin_valid = np.argwhere(self.surface_z < self.nz)

        if len(self.skin_valid) == 0:
            raise ValueError("Aucune colonne de peau valide trouvée.")

    def __len__(self):
        return self.iterations

    def _clip_start(self, c, size, max_size):
        c0 = int(c - size // 2)
        c0 = max(0, min(c0, max_size - size))
        return c0

    def _sample_positive_centered(self):
        if len(self.pos_coords) == 0:
            return self._sample_random()

        x, y, z = self.pos_coords[np.random.randint(len(self.pos_coords))]
        x0 = self._clip_start(x, self.patch_xy, self.nx)
        y0 = self._clip_start(y, self.patch_xy, self.ny)
        z0 = self._clip_start(z, self.patch_z, self.nz)
        return x0, y0, z0

    def _sample_near_skin(self):
        x, y = self.skin_valid[np.random.randint(len(self.skin_valid))]
        z_skin = int(np.clip(round(self.surface_z[x, y]), 0, self.nz - 1))

        # centre z aléatoire dans une bande autour de la peau
        z_center = z_skin + np.random.randint(-8, 12)
        x0 = self._clip_start(x, self.patch_xy, self.nx)
        y0 = self._clip_start(y, self.patch_xy, self.ny)
        z0 = self._clip_start(z_center, self.patch_z, self.nz)
        return x0, y0, z0

    def _sample_random(self):
        x0 = np.random.randint(0, self.nx - self.patch_xy + 1)
        y0 = np.random.randint(0, self.ny - self.patch_xy + 1)
        z0 = np.random.randint(0, self.nz - self.patch_z + 1)
        return x0, y0, z0

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.positive_patch_ratio:
            x0, y0, z0 = self._sample_positive_centered()
        elif r < self.positive_patch_ratio + self.skin_patch_ratio:
            x0, y0, z0 = self._sample_near_skin()
        else:
            x0, y0, z0 = self._sample_random()

        vol_patch = self.volume[x0:x0+self.patch_xy, y0:y0+self.patch_xy, z0:z0+self.patch_z]
        dist_patch = self.dist[x0:x0+self.patch_xy, y0:y0+self.patch_xy, z0:z0+self.patch_z]
        mask_patch = self.mask[x0:x0+self.patch_xy, y0:y0+self.patch_xy, z0:z0+self.patch_z]

        x = np.stack([vol_patch, dist_patch], axis=0).astype(np.float32)
        y = mask_patch[None, ...].astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


# =========================================================
# Main
# =========================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"🚀 Device: {device}")

    print("Chargement des données...")
    volume = np.load(args.fichier_volume)["volume"].astype(np.float32)
    mask_pos = np.load(args.fichier_poils).astype(np.uint8)
    surface_z = np.load(args.fichier_peau).astype(np.float32)

    nx, ny, nz = volume.shape
    print(f"Volume shape: {volume.shape}")
    print(f"Positifs shape: {mask_pos.shape}")
    print(f"Surface shape: {surface_z.shape}")

    if mask_pos.shape != volume.shape:
        raise ValueError("masque positifs incompatible avec le volume")
    if surface_z.shape != volume.shape[:2]:
        raise ValueError("surface peau incompatible avec le volume")

    print("Préparation des canaux d'entrée...")
    volume_norm = zscore_norm(volume)
    dist_channel = build_signed_distance_channel(
        surface_z=surface_z,
        shape_3d=volume.shape,
        dz_mm=args.dz_mm,
        clip_mm=args.distance_clip_mm,
    )

    dataset = GuidedPatchDataset(
        volume_norm=volume_norm,
        dist_channel=dist_channel,
        mask_pos=mask_pos,
        surface_z=surface_z,
        patch_xy=args.patch_xy,
        patch_z=args.patch_z,
        iterations=args.iterations_par_epoch,
        dz_mm=args.dz_mm,
        positive_patch_ratio=args.positive_patch_ratio,
        skin_patch_ratio=args.skin_patch_ratio,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = MiniUNet3DAniso(in_channels=2, base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nnPULoss(prior=args.prior_pi)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("--- 🏋️ Entraînement ---")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1}: loss moyenne = {total_loss / len(dataloader):.4f}")

    print("--- 🔬 Inférence avec chevauchement pondéré ---")
    model.eval()

    px, py, pz = args.patch_xy, args.patch_xy, args.patch_z
    sx, sy, sz = args.step_xy, args.step_xy, args.step_z

    window = hann_window_3d(px, py, pz)

    sum_pred = np.zeros((nx, ny, nz), dtype=np.float32)
    sum_w = np.zeros((nx, ny, nz), dtype=np.float32)

    def start_positions(n, patch, step):
        pos = list(range(0, max(1, n - patch + 1), step))
        if len(pos) == 0 or pos[-1] != n - patch:
            pos.append(max(0, n - patch))
        return pos

    xs = start_positions(nx, px, sx)
    ys = start_positions(ny, py, sy)
    zs = start_positions(nz, pz, sz)

    total_steps = len(xs) * len(ys) * len(zs)

    with torch.no_grad():
        pbar = tqdm(total=total_steps, desc="Inférence")
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    vol_patch = volume_norm[x0:x0+px, y0:y0+py, z0:z0+pz]
                    dist_patch = dist_channel[x0:x0+px, y0:y0+py, z0:z0+pz]

                    x_patch = np.stack([vol_patch, dist_patch], axis=0)[None, ...].astype(np.float32)
                    x_tensor = torch.from_numpy(x_patch).to(device, non_blocking=True)

                    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        logits = model(x_tensor)
                        prob = torch.sigmoid(logits).squeeze().float().cpu().numpy()

                    sum_pred[x0:x0+px, y0:y0+py, z0:z0+pz] += prob * window
                    sum_w[x0:x0+px, y0:y0+py, z0:z0+pz] += window

                    pbar.update(1)
        pbar.close()

    avg_pred = sum_pred / np.maximum(sum_w, 1e-8)
    mask_pred = (avg_pred > args.threshold).astype(np.uint8)

    os.makedirs(os.path.dirname(args.fichier_sortie), exist_ok=True)

    print(f"💾 Sauvegarde score map : {args.fichier_sortie_scores}")
    np.save(args.fichier_sortie_scores, avg_pred.astype(np.float32))

    print(f"💾 Sauvegarde masque binaire : {args.fichier_sortie}")
    np.save(args.fichier_sortie, mask_pred)

    print("✅ Terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PU learning v2 pour segmentation de poils.")

    # fichiers
    parser.add_argument("--fichier_volume", type=str, default=r"dicom_data/dicom/4261_fromdcm.npz")
    parser.add_argument("--fichier_poils", type=str, default=r"pipeline/body_hair_extraction_methods/masque_positifs_pu.npy")
    parser.add_argument("--fichier_peau", type=str, default=r"pipeline/body_hair_extraction_methods/surface_peau.npy")
    parser.add_argument("--fichier_sortie", type=str, default=r"pipeline/body_hair_extraction_methods/masque_predit_pu_v2.npy")
    parser.add_argument("--fichier_sortie_scores", type=str, default=r"pipeline/body_hair_extraction_methods/scores_predit_pu_v2.npy")

    # géométrie / canaux
    parser.add_argument("--dz_mm", type=float, default=0.125)
    parser.add_argument("--distance_clip_mm", type=float, default=2.0)

    # prior / seuil
    parser.add_argument("--prior_pi", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.35)

    # réseau
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # patchs
    parser.add_argument("--patch_xy", type=int, default=96)
    parser.add_argument("--patch_z", type=int, default=32)

    # sampling
    parser.add_argument("--positive_patch_ratio", type=float, default=0.5)
    parser.add_argument("--skin_patch_ratio", type=float, default=0.3)

    # train
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iterations_par_epoch", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=4)

    # inférence
    parser.add_argument("--step_xy", type=int, default=48)
    parser.add_argument("--step_z", type=int, default=16)

    args = parser.parse_args()
    main(args)