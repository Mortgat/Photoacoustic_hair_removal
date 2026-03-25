import os
import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def zscore_norm(volume: np.ndarray, out_dtype=np.float16) -> np.ndarray:
    v = volume.astype(np.float32, copy=False)
    v = (v - v.mean()) / (v.std() + 1e-8)
    return v.astype(out_dtype, copy=False)


def build_signed_distance_channel(
    surface_z: np.ndarray,
    shape_3d: Tuple[int, int, int],
    dz_mm: float,
    clip_mm: float = 2.0,
    out_dtype=np.float16,
) -> np.ndarray:
    nx, ny, nz = shape_3d
    z_grid = np.arange(nz, dtype=np.float32)[None, None, :]
    surf = surface_z.astype(np.float32, copy=False)[:, :, None]

    dist_mm = (z_grid - surf) * dz_mm
    dist_mm = np.clip(dist_mm, -clip_mm, clip_mm)
    dist_mm = dist_mm / clip_mm

    invalid = surface_z >= nz
    if np.any(invalid):
        dist_mm[invalid, :] = 0.0

    return dist_mm.astype(out_dtype, copy=False)


def build_easy_negative_masks(
    surface_z: np.ndarray,
    shape_3d: Tuple[int, int, int],
    dz_mm: float,
    air_far_mm: float,
    deep_tissue_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = shape_3d
    z_grid = np.arange(nz, dtype=np.float32)[None, None, :]
    surf = surface_z.astype(np.float32, copy=False)[:, :, None]

    valid = (surface_z < nz)[:, :, None]

    air_far = (z_grid <= (surf - air_far_mm / dz_mm)) & valid
    deep_tissue = (z_grid >= (surf + deep_tissue_mm / dz_mm)) & valid

    return air_far.astype(bool), deep_tissue.astype(bool)


def hann_window_3d(px: int, py: int, pz: int) -> np.ndarray:
    wx = np.hanning(px)
    wy = np.hanning(py)
    wz = np.hanning(pz)
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    w = np.maximum(w.astype(np.float32), 1e-3)
    return w


class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), groups=8):
        super().__init__()
        pad = tuple(k // 2 for k in kernel_size)
        g = min(groups, out_c)
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(g, out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(g, out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet3DAniso(nn.Module):
    def __init__(self, in_channels=2, base=16):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base, kernel_size=(3, 3, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 1))
        self.enc2 = ConvBlock3D(base, base * 2, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3 = ConvBlock3D(base * 2, base * 4, kernel_size=(3, 3, 3))

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
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = d1[:, :, :e1.shape[2], :e1.shape[3], :e1.shape[4]]
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class HybridMaskedLoss(nn.Module):
    def __init__(
        self,
        pos_weight=1.0,
        easy_neg_weight=0.15,
    ):
        super().__init__()
        self.register_buffer("pos_weight_buf", torch.tensor([pos_weight], dtype=torch.float32))
        self.easy_neg_weight = easy_neg_weight

    def forward(self, logits, target, strong_valid, easy_neg_mask):
        bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.pos_weight_buf)

        loss_map = bce(logits, target)
        strong_valid = strong_valid.float()
        strong_denom = torch.clamp(strong_valid.sum(), min=1.0)
        strong_loss = (loss_map * strong_valid).sum() / strong_denom

        easy_target = torch.zeros_like(target)
        easy_loss_map = bce(logits, easy_target)
        easy_neg_mask = easy_neg_mask.float()
        easy_denom = torch.clamp(easy_neg_mask.sum(), min=1.0)
        easy_loss = (easy_loss_map * easy_neg_mask).sum() / easy_denom

        return strong_loss + self.easy_neg_weight * easy_loss


class GuidedHybridDataset(Dataset):
    def __init__(
        self,
        volume_norm,
        dist_channel,
        pos_mask,
        vessel_mask,
        easy_air_mask,
        easy_deep_mask,
        surface_z,
        patch_xy,
        patch_z,
        iterations,
        positive_patch_ratio=0.2,
        vessel_patch_ratio=0.5,
        skin_patch_ratio=0.3,
    ):
        self.volume = volume_norm
        self.dist = dist_channel

        self.pos = pos_mask.astype(bool, copy=False)
        self.vessel = vessel_mask.astype(bool, copy=False)
        self.easy_air = easy_air_mask.astype(bool, copy=False)
        self.easy_deep = easy_deep_mask.astype(bool, copy=False)
        self.surface = surface_z.astype(np.float32, copy=False)

        self.patch_xy = patch_xy
        self.patch_z = patch_z
        self.iterations = iterations

        self.nx, self.ny, self.nz = self.volume.shape

        self.pos_coords = np.argwhere(self.pos)
        self.vessel_coords = np.argwhere(self.vessel)
        self.skin_valid = np.argwhere(self.surface < self.nz)

        self.positive_patch_ratio = positive_patch_ratio
        self.vessel_patch_ratio = vessel_patch_ratio
        self.skin_patch_ratio = skin_patch_ratio

    def __len__(self):
        return self.iterations

    def _clip_start(self, c, size, max_size):
        c0 = int(c - size // 2)
        return max(0, min(c0, max_size - size))

    def _sample_random(self):
        x0 = np.random.randint(0, self.nx - self.patch_xy + 1)
        y0 = np.random.randint(0, self.ny - self.patch_xy + 1)
        z0 = np.random.randint(0, self.nz - self.patch_z + 1)
        return x0, y0, z0

    def _sample_from_coords(self, coords):
        if len(coords) == 0:
            return self._sample_random()
        x, y, z = coords[np.random.randint(len(coords))]
        return (
            self._clip_start(x, self.patch_xy, self.nx),
            self._clip_start(y, self.patch_xy, self.ny),
            self._clip_start(z, self.patch_z, self.nz),
        )

    def _sample_near_skin(self):
        x, y = self.skin_valid[np.random.randint(len(self.skin_valid))]
        z_skin = int(np.clip(round(self.surface[x, y]), 0, self.nz - 1))
        z_center = z_skin + np.random.randint(-10, 14)
        return (
            self._clip_start(x, self.patch_xy, self.nx),
            self._clip_start(y, self.patch_xy, self.ny),
            self._clip_start(z_center, self.patch_z, self.nz),
        )

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.positive_patch_ratio:
            x0, y0, z0 = self._sample_from_coords(self.pos_coords)
        elif r < self.positive_patch_ratio + self.vessel_patch_ratio:
            x0, y0, z0 = self._sample_from_coords(self.vessel_coords)
        else:
            x0, y0, z0 = self._sample_near_skin()

        xs = slice(x0, x0 + self.patch_xy)
        ys = slice(y0, y0 + self.patch_xy)
        zs = slice(z0, z0 + self.patch_z)

        vol_patch = self.volume[xs, ys, zs]
        dist_patch = self.dist[xs, ys, zs]

        pos_patch = self.pos[xs, ys, zs]
        vessel_patch = self.vessel[xs, ys, zs]
        easy_air_patch = self.easy_air[xs, ys, zs]
        easy_deep_patch = self.easy_deep[xs, ys, zs]

        x = np.stack([vol_patch, dist_patch], axis=0).astype(np.float32, copy=False)

        target = np.zeros_like(pos_patch, dtype=np.float32)
        strong_valid = np.zeros_like(pos_patch, dtype=np.float32)

        target[pos_patch] = 1.0
        strong_valid[pos_patch] = 1.0

        target[vessel_patch] = 0.0
        strong_valid[vessel_patch] = 1.0

        easy_neg = (easy_air_patch | easy_deep_patch) & (~pos_patch) & (~vessel_patch)
        easy_neg = easy_neg.astype(np.float32, copy=False)

        return (
            torch.from_numpy(x),
            torch.from_numpy(target[None, ...]),
            torch.from_numpy(strong_valid[None, ...]),
            torch.from_numpy(easy_neg[None, ...]),
        )


def inference_by_y_blocks(
    model,
    device,
    volume_norm,
    dist_channel,
    out_scores_path,
    patch_xy,
    patch_z,
    step_xy,
    step_z,
    y_block_size=128,
):
    nx, ny, nz = volume_norm.shape
    px, py, pz = patch_xy, patch_xy, patch_z
    sx, sy, sz = step_xy, step_xy, step_z

    window = hann_window_3d(px, py, pz)

    scores_memmap = np.lib.format.open_memmap(
        out_scores_path, mode="w+", dtype=np.float32, shape=(nx, ny, nz)
    )

    def starts(n, patch, step):
        pts = list(range(0, max(1, n - patch + 1), step))
        if len(pts) == 0 or pts[-1] != n - patch:
            pts.append(max(0, n - patch))
        return pts

    xs = starts(nx, px, sx)
    ys = starts(ny, py, sy)
    zs = starts(nz, pz, sz)

    model.eval()

    with torch.no_grad():
        for y_block_start in range(0, ny, y_block_size):
            y_block_end = min(ny, y_block_start + y_block_size)

            block_len = y_block_end - y_block_start
            sum_pred = np.zeros((nx, block_len, nz), dtype=np.float32)
            sum_w = np.zeros((nx, block_len, nz), dtype=np.float32)

            relevant_ys = [y0 for y0 in ys if not (y0 + py <= y_block_start or y0 >= y_block_end)]
            total_steps = len(xs) * len(relevant_ys) * len(zs)
            pbar = tqdm(total=total_steps, desc=f"Inférence Y[{y_block_start}:{y_block_end}]")

            for x0 in xs:
                for y0 in relevant_ys:
                    for z0 in zs:
                        vpatch = volume_norm[x0:x0+px, y0:y0+py, z0:z0+pz]
                        dpatch = dist_channel[x0:x0+px, y0:y0+py, z0:z0+pz]

                        x_patch = np.stack([vpatch, dpatch], axis=0)[None, ...].astype(np.float32, copy=False)
                        x_tensor = torch.from_numpy(x_patch).to(device, non_blocking=True)

                        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                            logits = model(x_tensor)
                            prob = torch.sigmoid(logits).squeeze().float().cpu().numpy()

                        yy0 = max(y0, y_block_start)
                        yy1 = min(y0 + py, y_block_end)

                        local_patch_y0 = yy0 - y0
                        local_patch_y1 = yy1 - y0
                        local_block_y0 = yy0 - y_block_start
                        local_block_y1 = yy1 - y_block_start

                        sum_pred[x0:x0+px, local_block_y0:local_block_y1, z0:z0+pz] += (
                            prob[:, local_patch_y0:local_patch_y1, :] *
                            window[:, local_patch_y0:local_patch_y1, :]
                        )
                        sum_w[x0:x0+px, local_block_y0:local_block_y1, z0:z0+pz] += (
                            window[:, local_patch_y0:local_patch_y1, :]
                        )

                        pbar.update(1)

            pbar.close()

            block_scores = sum_pred / np.maximum(sum_w, 1e-8)
            scores_memmap[:, y_block_start:y_block_end, :] = block_scores.astype(np.float32, copy=False)
            scores_memmap.flush()

            del sum_pred, sum_w, block_scores

    return scores_memmap


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"🚀 Device: {device}")

    print("Chargement des données...")
    volume = np.load(args.fichier_volume)["volume"].astype(np.float32, copy=False)
    pos_mask = np.load(args.fichier_poils).astype(bool, copy=False)
    vessel_mask = np.load(args.fichier_vaisseaux).astype(bool, copy=False)
    surface_z = np.load(args.fichier_peau).astype(np.float32, copy=False)

    print("Préparation des canaux...")
    volume_norm = zscore_norm(volume, out_dtype=np.float16)
    del volume

    dist_channel = build_signed_distance_channel(
        surface_z=surface_z,
        shape_3d=volume_norm.shape,
        dz_mm=args.dz_mm,
        clip_mm=args.distance_clip_mm,
        out_dtype=np.float16,
    )

    print("Construction des easy negatives...")
    easy_air_mask, easy_deep_mask = build_easy_negative_masks(
        surface_z=surface_z,
        shape_3d=volume_norm.shape,
        dz_mm=args.dz_mm,
        air_far_mm=args.easy_air_far_mm,
        deep_tissue_mm=args.easy_deep_mm,
    )

    dataset = GuidedHybridDataset(
        volume_norm=volume_norm,
        dist_channel=dist_channel,
        pos_mask=pos_mask,
        vessel_mask=vessel_mask,
        easy_air_mask=easy_air_mask,
        easy_deep_mask=easy_deep_mask,
        surface_z=surface_z,
        patch_xy=args.patch_xy,
        patch_z=args.patch_z,
        iterations=args.iterations_par_epoch,
        positive_patch_ratio=args.positive_patch_ratio,
        vessel_patch_ratio=args.vessel_patch_ratio,
        skin_patch_ratio=args.skin_patch_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = MiniUNet3DAniso(in_channels=2, base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = HybridMaskedLoss(
        pos_weight=args.pos_weight,
        easy_neg_weight=args.easy_neg_weight,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("--- Entraînement ---")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_x, batch_target, batch_valid, batch_easy in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_valid = batch_valid.to(device, non_blocking=True)
            batch_easy = batch_easy.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(batch_x)
                loss = criterion(logits, batch_target, batch_valid, batch_easy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1}: loss moyenne = {total_loss / len(loader):.4f}")

    print("--- Inférence mémoire-safe ---")
    os.makedirs(os.path.dirname(args.fichier_sortie), exist_ok=True)

    scores_memmap = inference_by_y_blocks(
        model=model,
        device=device,
        volume_norm=volume_norm,
        dist_channel=dist_channel,
        out_scores_path=args.fichier_sortie_scores,
        patch_xy=args.patch_xy,
        patch_z=args.patch_z,
        step_xy=args.step_xy,
        step_z=args.step_z,
        y_block_size=args.y_block_size,
    )

    print("Binarisation finale...")
    pred = (np.asarray(scores_memmap) > args.threshold).astype(np.uint8)
    np.save(args.fichier_sortie, pred)

    print("✅ Terminé.")
    print(f"Scores : {args.fichier_sortie_scores}")
    print(f"Masque : {args.fichier_sortie}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fichier_volume", type=str, default=r"dicom_data/dicom/4261_fromdcm.npz")
    parser.add_argument("--fichier_peau", type=str, default=r"pipeline/body_hair_extraction_methods/surface_peau.npy")
    parser.add_argument("--fichier_poils", type=str, default=r"pipeline/body_hair_extraction_methods/masque_poils_surs.npy")
    parser.add_argument("--fichier_vaisseaux", type=str, default=r"pipeline/body_hair_extraction_methods/masque_vaisseaux_potentiels.npy")
    parser.add_argument("--fichier_sortie", type=str, default=r"pipeline/body_hair_extraction_methods/masque_predit_hybrid_v2.npy")
    parser.add_argument("--fichier_sortie_scores", type=str, default=r"pipeline/body_hair_extraction_methods/scores_predit_hybrid_v2.npy")

    parser.add_argument("--dz_mm", type=float, default=0.125)
    parser.add_argument("--distance_clip_mm", type=float, default=2.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=16)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iterations_par_epoch", type=int, default=400)

    parser.add_argument("--patch_xy", type=int, default=64)
    parser.add_argument("--patch_z", type=int, default=24)

    parser.add_argument("--positive_patch_ratio", type=float, default=0.2)
    parser.add_argument("--vessel_patch_ratio", type=float, default=0.5)
    parser.add_argument("--skin_patch_ratio", type=float, default=0.3)

    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--easy_neg_weight", type=float, default=0.15)

    parser.add_argument("--easy_air_far_mm", type=float, default=3.0)
    parser.add_argument("--easy_deep_mm", type=float, default=6.0)

    parser.add_argument("--threshold", type=float, default=0.70)

    parser.add_argument("--step_xy", type=int, default=32)
    parser.add_argument("--step_z", type=int, default=12)
    parser.add_argument("--y_block_size", type=int, default=128)

    args = parser.parse_args()
    main(args)