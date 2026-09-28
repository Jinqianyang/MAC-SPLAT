import json
import os
import sys
import math
import inspect
import einops
import lightning as L
import lpips
import omegaconf
import torch
import torch.nn.functional as F
import wandb
import logging

sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.scannetpp.scannetpp as scannetpp
import src.mast3r_src.mast3r.model_v3 as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import workspace
import utils.geometry as geometry
from utils.fast_matching import find_matches_fast_reciprocal

try:
    from gsplat.rendering import rasterization as gsplat_rasterization
    _GSP_AVAILABLE = True
except Exception:
    gsplat_rasterization = None
    _GSP_AVAILABLE = False


def _to_h(p):
    return torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)

def _from_h(p):
    return p[..., :3] / p[..., 3:].clamp_min(1e-6)

def _invert_se3(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    Rinv = R.transpose(-1, -2)
    tinv = -Rinv @ t
    out = torch.eye(4, device=T.device, dtype=T.dtype).expand_as(T).clone()
    out[..., :3, :3] = Rinv
    out[..., :3, 3:4] = tinv
    return out

def _transform_points(p, T):
    Ph = _to_h(p)
    Qh = Ph @ T.transpose(-1, -2)
    return _from_h(Qh)

def _ensure_batched_T(T, B, device):
    T = torch.as_tensor(T, device=device, dtype=torch.float32)
    if T.ndim == 2:
        if T.shape == (3, 4):
            T = torch.cat([T, torch.tensor([[0, 0, 0, 1.]], device=device)], dim=0)
        T = T.unsqueeze(0).expand(B, -1, -1).contiguous()
    elif T.ndim == 3:
        if T.shape[1:] == (3, 4):
            pad = torch.zeros((T.shape[0], 1, 4), device=device)
            pad[..., 0, 3] = 1.
            T = torch.cat([T, pad], dim=1)
        assert T.shape[0] == B, f"Pose batch {T.shape[0]} != data batch {B}"
    else:
        raise ValueError(f"Unexpected pose shape: {T.shape}")
    return T

def _px_to_grid(uv_px, H, W, *, align_corners=False, device=None, dtype=torch.float32):
    if device is None:
        device = uv_px.device
    uv_px = uv_px.to(dtype=dtype, device=device)
    if align_corners:
        norm = torch.tensor([W - 1, H - 1], device=device, dtype=dtype)
        return uv_px / norm * 2 - 1
    norm = torch.tensor([W, H], device=device, dtype=dtype)
    return (uv_px + 0.5) / norm * 2 - 1

def _fetch_T_wc(ctx, *, device, B, default_conv='c2w'):
    def as_4x4(M):
        return _ensure_batched_T(M, B, device)
    for k in ['camera_pose', 'T_w_c', 'Twc', 'T_wc', 'c2w', 'cam2world', 'pose', 'T_cam_world', 'cam_T_world']:
        if k in ctx:
            return as_4x4(ctx[k])
    for k in ['T_c_w', 'Tcw', 'w2c', 'world2cam', 'T_world_cam', 'world_T_cam']:
        if k in ctx:
            T_c_w = as_4x4(ctx[k])
            return _invert_se3(T_c_w)
    if 'R' in ctx and ('t' in ctx or 'T' in ctx):
        R = torch.as_tensor(ctx['R'], device=device, dtype=torch.float32)
        t = torch.as_tensor(ctx.get('t', ctx.get('T')), device=device, dtype=torch.float32)
        if t.ndim == R.ndim - 1:
            t = t.unsqueeze(-1)
        if R.ndim == 2:
            M = torch.eye(4, device=device); M[:3, :3] = R; M[:3, 3:4] = t
        else:
            M = torch.eye(4, device=device).repeat(R.shape[0], 1, 1)
            M[:, :3, :3] = R; M[:, :3, 3:4] = t
        return as_4x4(M)
    raise KeyError("Context missing extrinsics (cam2world/world2cam).")

def _fetch_K(ctx, *, device):
    for k in ['K', 'camera_intrinsics', 'intrinsics', 'kalib']:
        if k in ctx:
            K = torch.as_tensor(ctx[k], device=device, dtype=torch.float32)
            if K.ndim == 2:
                K = K.unsqueeze(0)
            K = K[..., :3, :3]
            return K
    raise KeyError("Context missing intrinsics (K).")


class MAST3RGaussians(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = {}

        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
            dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
            two_confs=True, use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        self.encoder.requires_grad_(False)
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)

        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )
        self.benchmarker = benchmarker.Benchmarker()

        spatial_lpips = config.loss.get('average_over_mask', True)
        self.lpips_criterion = lpips.LPIPS('vgg', spatial=spatial_lpips)

        if self.config.loss.get('mast3r_loss_weight') is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()


    def setup(self, stage: str):
        logging.info(f"--- Running setup() on rank {self.global_rank} for stage: {stage} ---")

        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                logging.info("Setting up train dataset...")
                self.train_dataset = scannetpp.get_scannet_dataset(
                    self.config.data.root,
                    'train',
                    self.config.data.resolution,
                    num_epochs_per_epoch=self.config.data.epochs_per_train_epoch,
                )
            if self.val_dataset is None:
                logging.info("Setting up validation dataset...")
                self.val_dataset = scannetpp.get_scannet_test_dataset(
                    self.config.data.root,
                    alpha=0.5,
                    beta=0.5,
                    resolution=self.config.data.resolution,
                    use_every_n_sample=100,
                )

        if stage == 'test':
            if not self.test_datasets:
                logging.info("Setting up test datasets...")
                for alpha, beta in ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5), (0.3, 0.3)):
                    key = f"alpha_{alpha}_beta_{beta}"
                    self.test_datasets[key] = scannetpp.get_scannet_test_dataset(
                        self.config.data.root,
                        alpha=alpha,
                        beta=beta,
                        resolution=self.config.data.resolution,
                        use_every_n_sample=10
                    )

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn_skip_corrupted,
            sampler=sampler,
            shuffle=shuffle,
            pin_memory=True
        )

    def val_dataloader(self):
        sampler = None
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=True)

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn_skip_corrupted,
            sampler=sampler,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        dataloaders = []
        for test_dataset in self.test_datasets.values():
            sampler = None
            if self.trainer is not None and self.trainer.world_size > 1:
                sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

            loader = DataLoader(
                test_dataset,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                collate_fn=collate_fn_skip_corrupted,
                sampler=sampler,
                shuffle=False
            )
            dataloaders.append(loader)
        return dataloaders


    def _cfg(self, key, default=None):
        v = self.config.loss.get(key, None)
        if v is not None:
            return v
        if key.startswith('fewview_'):
            v = self.config.loss.get(key.replace('fewview_', 'novel_view_'), None)
        elif key.startswith('novel_view_'):
            v = self.config.loss.get(key.replace('novel_view_', 'fewview_'), None)
        return default if v is None else v

    def _post_process_predictions(self, pred):
        max_scale_limit = self.config.get('max_scale_limit', 0.5)
        max_logit_limit = math.log(max_scale_limit)
        if 'scale_logits' in pred:
            if self.training:
                self.log('train/debug_logits_unclamped_max', pred['scale_logits'].detach().max(), on_step=True, on_epoch=False)
            clamped_logits = pred['scale_logits'].clamp(max=max_logit_limit)
            pred['scales'] = torch.exp(clamped_logits)
        return pred

    def debug_gaussian_params(self, gaussians, prefix=""):
        log_data = {}
        if 'pts3d' in gaussians:
            positions = gaussians['pts3d'].detach()
            log_data[f'{prefix}/pos_max'] = positions.max()
            log_data[f'{prefix}/pos_min'] = positions.min()
        if 'scales' in gaussians:
            scales = gaussians['scales'].detach()
            log_data[f'{prefix}/scale_max'] = scales.max()
            log_data[f'{prefix}/scale_mean'] = scales.mean()
            log_data[f'{prefix}/scale_min'] = scales.min()
        if 'opacities' in gaussians:
            opacities = gaussians['opacities'].detach()
            log_data[f'{prefix}/opacity_max'] = opacities.max()
            log_data[f'{prefix}/opacity_mean'] = opacities.mean()
            log_data[f'{prefix}/opacity_min'] = opacities.min()
        if 'sh' in gaussians:
            sh = gaussians['sh'].detach()
            log_data[f'{prefix}/sh_max'] = sh.max()
            log_data[f'{prefix}/sh_min'] = sh.min()
        self.log_dict(log_data, on_step=True, on_epoch=False)

    def _histogram_summary(self, tensor: torch.Tensor, prefix: str, bins: int = 20):
        x = tensor.detach().reshape(-1)
        if x.numel() == 0: return
        q = torch.quantile(x, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x.device))
        self.log_dict({
            f"{prefix}/q0": q[0], f"{prefix}/q25": q[1],
            f"{prefix}/median": q[2], f"{prefix}/q75": q[3], f"{prefix}/q100": q[4],
        }, on_step=True, on_epoch=False)

    def _track_scale_drift(self, pred1, pred2, prefix="train"):
        s1 = pred1.get('scales', None); s2 = pred2.get('scales', None)
        if s1 is None or s2 is None: return
        s1m = s1.detach().mean(); s2m = s2.detach().mean()
        self.log(f"{prefix}/scale_mean_v1", s1m, on_step=True, on_epoch=False)
        self.log(f"{prefix}/scale_mean_v2", s2m, on_step=True, on_epoch=False)
        self.log(f"{prefix}/scale_mean_gap", (s1m - s2m).abs(), on_step=True, on_epoch=False)
        if not hasattr(self, "_prev_scale_mean_v1"):
            self._prev_scale_mean_v1, self._prev_scale_mean_v2 = s1m, s2m
        v1_speed = (s1m - self._prev_scale_mean_v1).abs()
        v2_speed = (s2m - self._prev_scale_mean_v2).abs()
        self._prev_scale_mean_v1, self._prev_scale_mean_v2 = s1m, s2m
        self.log(f"{prefix}/scale_drift_speed_v1", v1_speed, on_step=True, on_epoch=False)
        self.log(f"{prefix}/scale_drift_speed_v2", v2_speed, on_step=True, on_epoch=False)

    def _check_grad_norms(self, named_params, prefix="train/grad"):
        total = 0.0
        for n, p in named_params:
            if p.grad is not None:
                g = p.grad.detach()
                g2 = (g * g).sum()
                total += g2
        self.log(f"{prefix}_global_l2", torch.sqrt(torch.tensor(total, device=self.device).detach() + 1e-12),
                 on_step=True, on_epoch=False)

    def _check_nan_inf(self, *tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    raise RuntimeError("NaN/Inf detected in loss or intermediates.")

    def _render_with_gsplat_single(self, gauss, w2c, K_norm, image_shape):
        assert _GSP_AVAILABLE, "gsplat is not installed; novel-view consistency is unavailable."
        B = gauss['means'].shape[0]
        H, W = image_shape
        rgb_out, alpha_out = [], []

        near_plane = float(self._cfg('fewview_near', 0.1))
        far_plane  = float(self._cfg('fewview_far', 1000.0))

        for b in range(B):
            means_b = einops.rearrange(gauss['means'][b:b+1], 'b h w c -> (b h w) c')
            covars_b = einops.rearrange(gauss['covariances'][b:b+1], 'b h w i j -> (b h w) i j')
            sh_b = gauss['sh'][b:b+1]
            if sh_b.ndim == 5:
                sh_b = einops.rearrange(sh_b, 'b h w c d -> (b h w) c d')
            else:
                deg = int(math.sqrt(sh_b.shape[-1] // 3) - 1)
                sh_b = einops.rearrange(sh_b, 'b h w (c d) -> (b h w) c d', c=3, d=(deg + 1) ** 2)
            colors = sh_b.permute(0, 2, 1).contiguous()
            opac_b = einops.rearrange(gauss['opacities'][b:b+1], 'b h w 1 -> (b h w)')

            Kn = K_norm[b:b+1].clone()
            Kn[..., 0, 0] *= W; Kn[..., 1, 1] *= H
            Kn[..., 0, 2] *= W; Kn[..., 1, 2] *= H

            backgrounds = torch.zeros((1, 3), device=means_b.device, dtype=means_b.dtype)
            sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
            render, alpha, _meta = gsplat_rasterization(
                means=means_b, covars=covars_b, colors=colors, opacities=opac_b,
                viewmats=w2c[b:b+1], Ks=Kn, width=W, height=H, sh_degree=sh_degree,
                backgrounds=backgrounds, near_plane=near_plane, far_plane=far_plane,
                packed=False, quats=None, scales=None, render_mode="RGB",
            )
            rgb_out.append(einops.rearrange(render, 'b h w c -> b c h w'))
            alpha_out.append(einops.rearrange(alpha, 'b h w 1 -> b 1 h w'))
        return torch.cat(rgb_out, 0), torch.cat(alpha_out, 0)

    def _image_grad_mag(self, img_bchw: torch.Tensor) -> torch.Tensor:
        dx = img_bchw[:, :, :, 1:] - img_bchw[:, :, :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = img_bchw[:, :, 1:, :] - img_bchw[:, :, :-1, :]
        dy = F.pad(dy, (0, 0, 0, 1))
        grad = torch.sqrt(
            (dx**2).mean(dim=1, keepdim=True) +
            (dy**2).mean(dim=1, keepdim=True) + 1e-12
        )
        assert dx.shape == dy.shape == img_bchw.shape, f"grad shape mismatch: dx={dx.shape}, dy={dy.shape}, img={img_bchw.shape}"
        return grad

    def calculate_novel_view_consistency_loss(self, batch, pred1, pred2_aligned, *, image_shape, matches_info):
        device = self.device
        H, W = image_shape

        def _cfg(key, default):
            loss_cfg = getattr(self.config, "loss", {})
            if key in loss_cfg: return loss_cfg[key]
            if key.startswith("fewview_"):
                return loss_cfg.get(key.replace("fewview_", "novel_view_"), default)
            if key.startswith("novel_view_"):
                return loss_cfg.get(key.replace("novel_view_", "fewview_"), default)
            return default

        if matches_info is None or matches_info["matches_px"].numel() == 0:
            return torch.tensor(0.0, device=device), {"nvc/no_matches": torch.tensor(1.0, device=device)}

        B = pred1["means"].shape[0]
        ctx1, ctx2 = batch["context"]
        try:
            T_w_c1 = _fetch_T_wc(ctx1, device=device, B=B)
            T_w_c2 = _fetch_T_wc(ctx2, device=device, B=B)
            K1_px  = _fetch_K(ctx1, device=device)
            K2_px  = _fetch_K(ctx2, device=device)
        except Exception:
            return torch.tensor(0.0, device=device), {"nvc/cam_missing": torch.tensor(1.0, device=device)}

        t_mid = float(_cfg("fewview_nvc_t", 0.5))

        def _mat_log(R):
            cos_th = ((R.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) * 0.5).clamp(-1.0, 1.0)
            th = torch.acos(cos_th)
            A = (R - R.transpose(1, 2)) * 0.5
            s = torch.sin(th).clamp_min(1e-6)
            return A * (th / s).view(-1, 1, 1)

        def _mat_exp(L):
            L2 = torch.bmm(L, L)
            th2 = (-0.5 * L2.diagonal(dim1=1, dim2=2).sum(dim=1)).clamp_min(0)
            th = torch.sqrt(th2)
            I = torch.eye(3, device=L.device, dtype=L.dtype).unsqueeze(0).repeat(L.shape[0], 1, 1)
            s, c, t = torch.sin(th).view(-1,1,1), torch.cos(th).view(-1,1,1), th.view(-1,1,1).clamp_min(1e-6)
            return I + (s/t) * L + ((1 - c) / (t*t)) * L2

        R1, t1 = T_w_c1[:, :3, :3], T_w_c1[:, :3, 3]
        R2, t2 = T_w_c2[:, :3, :3], T_w_c2[:, :3, 3]
        delta = torch.bmm(R1.transpose(1, 2), R2)
        Rt = torch.bmm(R1, _mat_exp(_mat_log(delta) * t_mid))
        tt = (1 - t_mid) * t1 + t_mid * t2

        Tt_w_c = torch.eye(4, device=device, dtype=T_w_c1.dtype).unsqueeze(0).repeat(B, 1, 1)
        Tt_w_c[:, :3, :3] = Rt
        Tt_w_c[:, :3, 3]  = tt
        Tt_c_w = _invert_se3(Tt_w_c)

        K1n = geometry.normalize_intrinsics(K1_px, (H, W))[..., :3, :3]
        K2n = geometry.normalize_intrinsics(K2_px, (H, W))[..., :3, :3]
        Ktn = K1n.clone()
        Ktn[..., 0, 0] = (1 - t_mid) * K1n[..., 0, 0] + t_mid * K2n[..., 0, 0]
        Ktn[..., 1, 1] = (1 - t_mid) * K1n[..., 1, 1] + t_mid * K2n[..., 1, 1]
        Ktn[..., 0, 2] = (1 - t_mid) * K1n[..., 0, 2] + t_mid * K2n[..., 0, 2]
        Ktn[..., 1, 2] = (1 - t_mid) * K1n[..., 1, 2] + t_mid * K2n[..., 1, 2]

        def _to_world_gauss(means_bhw3, cov_bhwij, T_w_c):
            Pw = _transform_points(means_bhw3, T_w_c)
            R  = T_w_c[:, :3, :3]
            Rb = R[:, None, None, :, :]
            cov_w = Rb @ cov_bhwij @ Rb.transpose(-1, -2)
            return Pw, cov_w

        means1_w, cov1_w = _to_world_gauss(pred1["means"], pred1["covariances"], T_w_c1)

        pred2_has_cam1 = ("means" in pred2_aligned)
        if pred2_has_cam1:
            means2_w, cov2_w = _to_world_gauss(pred2_aligned["means"], pred2_aligned["covariances"], T_w_c1)
        else:
            means2_w, cov2_w = _to_world_gauss(pred2_aligned["means_in_other_view"], pred2_aligned["covariances"], T_w_c2)

        if not globals().get("_GSP_AVAILABLE", False) or gsplat_rasterization is None:
            return torch.tensor(0.0, device=device), {"nvc/gsplat_unavailable": torch.tensor(1.0, device=device)}

        gauss1 = {"means": means1_w, "covariances": cov1_w, "sh": pred1["sh"], "opacities": pred1["opacities"]}
        gauss2 = {"means": means2_w, "covariances": cov2_w, "sh": pred2_aligned["sh"], "opacities": pred2_aligned["opacities"]}
        rgb1, alpha1 = self._render_with_gsplat_single(gauss1, Tt_c_w, Ktn, (H, W))
        rgb2, alpha2 = self._render_with_gsplat_single(gauss2, Tt_c_w, Ktn, (H, W))

        def _sample_map(map_bchw, uv_px, batch_indices):
            B, C, HH, WW = map_bchw.shape
            out = torch.zeros((uv_px.shape[0], C), device=map_bchw.device, dtype=map_bchw.dtype)
            bi = batch_indices.to(map_bchw.device)
            for b in range(B):
                sel = (bi == b)
                if sel.any():
                    uv_b = uv_px[sel]
                    grid_b = _px_to_grid(uv_b, HH, WW, align_corners=False, device=map_bchw.device).view(1, -1, 1, 2)
                    s = F.grid_sample(map_bchw[b:b+1], grid_b, align_corners=False, mode="bilinear")
                    out[sel] = s.squeeze(0).squeeze(-1).permute(1, 0).contiguous()
            return out

        matches_px = matches_info["matches_px"].to(device)
        batch_idx  = matches_info["batch_indices"].to(device)
        conf_all   = matches_info["confidence"].to(device)

        Hp, Wp = pred1["means"].shape[1:3]
        uv1_px = matches_px[:, :2]
        uv2_px = matches_px[:, 2:]

        p1_map = pred1["means"].permute(0, 3, 1, 2)
        if pred2_has_cam1:
            p2_map = pred2_aligned["means"].permute(0, 3, 1, 2)
            Twc2_used = T_w_c1
        else:
            p2_map = pred2_aligned["means_in_other_view"].permute(0, 3, 1, 2)
            Twc2_used = T_w_c2

        p1_3d = _sample_map(p1_map, uv1_px, batch_idx)
        p2_3d = _sample_map(p2_map, uv2_px, batch_idx)

        Twc1_N = T_w_c1[batch_idx]
        Twc2_N = Twc2_used[batch_idx]
        TcwN   = Tt_c_w[batch_idx]
        Kb     = Ktn[batch_idx]

        def _to_world(PCam, Twc):
            return _from_h(_to_h(PCam).unsqueeze(1) @ Twc.transpose(-1, -2)).squeeze(1)

        Pw1 = _to_world(p1_3d, Twc1_N)
        Pw2 = _to_world(p2_3d, Twc2_N)

        def _to_novel_px(Pw, Tcw, Knorm):
            Pc = _from_h(_to_h(Pw).unsqueeze(1) @ Tcw.transpose(-1, -2)).squeeze(1)
            x = Pc[:, 0] / Pc[:, 2].clamp_min(1e-6)
            y = Pc[:, 1] / Pc[:, 2].clamp_min(1e-6)
            u = Knorm[:, 0, 0] * x + Knorm[:, 0, 2]
            v = Knorm[:, 1, 1] * y + Knorm[:, 1, 2]
            return torch.stack([(u * W).clamp(0, W - 1), (v * H).clamp(0, H - 1)], dim=-1)

        uvn1 = _to_novel_px(Pw1, TcwN, Kb)
        uvn2 = _to_novel_px(Pw2, TcwN, Kb)

        agree_th = float(_cfg("fewview_agree_thresh_px", 2.0))
        proj_dist = (uvn1 - uvn2).norm(dim=-1)
        agree = proj_dist < agree_th
        if agree.sum() == 0:
            return torch.tensor(0.0, device=device), {"nvc/no_agree": torch.tensor(1.0, device=device)}

        def _sample_map_bchw(map_bchw, uv_px):
            return _sample_map(map_bchw, uv_px, batch_idx).view(-1)

        a1 = _sample_map_bchw(alpha1, uvn1)
        a2 = _sample_map_bchw(alpha2, uvn2)
        tau = float(_cfg("fewview_alpha_thresh", 0.3))
        gate_soft = torch.sigmoid(10.0 * (a1 - tau)) * torch.sigmoid(10.0 * (a2 - tau))
        valid = agree & (gate_soft > 1e-4)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device), {"nvc/no_valid": torch.tensor(1.0, device=device)}

        rgb1_s = _sample_map(rgb1, uvn1, batch_idx)
        rgb2_s = _sample_map(rgb2, uvn2, batch_idx)

        img1 = batch["context"][0]["original_img"]
        img2 = batch["context"][1]["original_img"]
        g1 = self._image_grad_mag(img1)
        g2 = self._image_grad_mag(img2)

        scale_u, scale_v = W / float(Wp), H / float(Hp)
        uv1_full = torch.stack([uv1_px[:, 0] * scale_u, uv1_px[:, 1] * scale_v], dim=-1)
        uv2_full = torch.stack([uv2_px[:, 0] * scale_u, uv2_px[:, 1] * scale_v], dim=-1)
        grad1 = _sample_map(g1, uv1_full, batch_idx).view(-1)
        grad2 = _sample_map(g2, uv2_full, batch_idx).view(-1)
        grad_min = torch.minimum(grad1, grad2)
        th_grad = float(_cfg("fewview_grad_thresh", 0.1))
        W_grad = torch.where(grad_min > th_grad, torch.exp(-grad_min), torch.ones_like(grad_min))

        conf_all = conf_all.clamp_min(0)
        conf_sel = conf_all[valid]
        gate_sel = gate_soft[valid]
        w_tot = (conf_sel * gate_sel * W_grad[valid]).clamp_min(1e-8)

        color_res = (rgb1_s[valid] - rgb2_s[valid]).abs().mean(dim=-1)
        color_loss = (color_res * w_tot).sum() / w_tot.sum()

        add_loss = torch.tensor(0.0, device=device)
        w_shape = float(_cfg("fewview_shape_weight", 0.0))
        w_opac  = float(_cfg("fewview_opacity_weight", 0.0))

        if w_shape > 0:
            s1_map = pred1["scales"].permute(0, 3, 1, 2)
            s2_map = pred2_aligned["scales"].permute(0, 3, 1, 2)
            s1 = _sample_map(s1_map, uv1_px, batch_idx)
            s2 = _sample_map(s2_map, uv2_px, batch_idx)
            l_shape = (torch.log(s1.clamp_min(1e-6)) - torch.log(s2.clamp_min(1e-6))).abs().mean(dim=-1)
            add_loss = add_loss + w_shape * (l_shape[valid] * w_tot).sum() / w_tot.sum()

        if w_opac > 0:
            o1_map = pred1["opacities"].permute(0, 3, 1, 2)
            o2_map = pred2_aligned["opacities"].permute(0, 3, 1, 2)
            o1 = _sample_map(o1_map, uv1_px, batch_idx).view(-1)
            o2 = _sample_map(o2_map, uv2_px, batch_idx).view(-1)
            l_opac = (o1 - o2).abs()
            add_loss = add_loss + w_opac * (l_opac[valid] * w_tot).sum() / w_tot.sum()

        total = color_loss + add_loss

        Kn_pix_fx = (Ktn[..., 0, 0] * W).mean().detach()
        Kn_pix_cx = (Ktn[..., 0, 2] * W).mean().detach()
        logs = {
            "nvc/color":        color_loss.detach(),
            "nvc/extra":        add_loss.detach(),
            "nvc/agree_ratio":  agree.float().mean().detach(),
            "nvc/valid_ratio":  valid.float().mean().detach(),
            "nvc/alpha1_mean":  alpha1.mean().detach(),
            "nvc/alpha2_mean":  alpha2.mean().detach(),
            "nvc/gate_mean":    gate_soft.mean().detach(),
            "nvc/K_fx_px":      Kn_pix_fx,
            "nvc/K_cx_px":      Kn_pix_cx,
        }
        return total, logs

    def forward(self, view1, view2):
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        head_method = self.encoder._downstream_head
        _ = inspect.signature(head_method)
        decout1_for_head = [tok.float() for tok in dec1]
        decout2_for_head = [tok.float() for tok in dec2]

        pred1 = self.encoder._downstream_head(1, decout1_for_head, shape1, view1.get('original_img'))
        pred2 = self.encoder._downstream_head(2, decout2_for_head, shape2, view2.get('original_img'))

        pred1 = self._post_process_predictions(pred1)
        pred2 = self._post_process_predictions(pred2)

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        if self.config.get('learn_residual_sh', True):
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def calculate_3d_consistency_loss(self, batch, matches_info, pred1, pred2):
        device = self.device
        loss_cfg = getattr(self.config, "loss", {})

        w_pos     = float(loss_cfg.get("consistency_pos_weight",     1.0))
        w_shape   = float(loss_cfg.get("consistency_shape_weight",   0.1))
        w_opacity = float(loss_cfg.get("consistency_opacity_weight", 0.05))
        w_color   = float(loss_cfg.get("consistency_color_weight",   0.05))
        use_dc_only = bool(loss_cfg.get("consistency_color_use_dc_only", False))

        ctx1, ctx2 = batch["context"]
        B = pred1["means"].shape[0]
        T_w_c1 = _fetch_T_wc(ctx1, device=device, B=B)
        T_w_c2 = _fetch_T_wc(ctx2, device=device, B=B)

        matches_px     = matches_info['matches_px']
        confidence_all = matches_info['confidence']
        batch_indices  = matches_info['batch_indices']

        means1_map = pred1['means'].permute(0, 3, 1, 2).contiguous()
        means2_map = pred2['means_in_other_view'].permute(0, 3, 1, 2).contiguous()
        B, _, H, W = means1_map.shape

        cov1_map = pred1['covariances'].reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()
        cov2_map = pred2['covariances'].reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()

        def _to_bchw_dense(x):
            if x is None:
                return None
            if getattr(x, "is_sparse", False) or (hasattr(torch, "sparse") and x.layout != torch.strided):
                x = x.to_dense()
            if x.ndim == 4:
                return x.permute(0, 3, 1, 2).contiguous()
            elif x.ndim == 3:
                return x.unsqueeze(1).contiguous()
            else:
                return x.contiguous()

        opac1_map = _to_bchw_dense(pred1.get('opacities', None))
        opac2_map = _to_bchw_dense(pred2.get('opacities', None))

        sh1 = pred1.get('sh', None)
        sh2 = pred2.get('sh', None)
        sh1_map = None
        sh2_map = None
        if sh1 is not None:
            if sh1.ndim == 5:
                Bb, Hh, Ww, Cc, Dd = sh1.shape
                sh1_map = sh1.reshape(Bb, Hh, Ww, Cc * Dd).permute(0, 3, 1, 2).contiguous()
            else:
                sh1_map = sh1.permute(0, 3, 1, 2).contiguous()
        if sh2 is not None:
            if sh2.ndim == 5:
                Bb, Hh, Ww, Cc, Dd = sh2.shape
                sh2_map = sh2.reshape(Bb, Hh, Ww, Cc * Dd).permute(0, 3, 1, 2).contiguous()
            else:
                sh2_map = sh2.permute(0, 3, 1, 2).contiguous()

        total_num = torch.tensor(0.0, device=device)
        total_den = torch.tensor(0.0, device=device)

        def _huber(x, delta=0.1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5 * ax * ax / delta, ax - 0.5 * delta)

        chunk_size = int(self.config.get("consistency_chunk_size", 64))
        eps_spd = 1e-6

        for b_idx in range(B):
            sel = (batch_indices == b_idx)
            if not sel.any():
                continue

            item_matches_px = matches_px[sel]
            item_confidence = confidence_all[sel]
            Ni = item_matches_px.shape[0]

            uv1_px_all = item_matches_px[:, :2]
            uv2_px_all = item_matches_px[:, 2:]

            Twc1_b = T_w_c1[b_idx]
            Twc2_b = T_w_c2[b_idx]
            R1_b = Twc1_b[:3, :3]
            R2_b = Twc2_b[:3, :3]

            for i0 in range(0, Ni, chunk_size):
                i1 = min(Ni, i0 + chunk_size)
                uv1_px = uv1_px_all[i0:i1]
                uv2_px = uv2_px_all[i0:i1]
                conf   = item_confidence[i0:i1]
                M = uv1_px.shape[0]
                if M == 0:
                    continue

                norm = torch.tensor([W - 1, H - 1], device=device, dtype=torch.float32)
                uv1_norm = (uv1_px / norm * 2 - 1).view(M, 1, 1, 2)
                uv2_norm = (uv2_px / norm * 2 - 1).view(M, 1, 1, 2)

                if w_pos > 0:
                    p1_cam = F.grid_sample(
                        means1_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv1_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1)

                    p2_cam = F.grid_sample(
                        means2_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv2_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1)

                    z_ok = (p1_cam[:, 2] > 1e-6) & (p2_cam[:, 2] > 1e-6)
                    if z_ok.sum() > 0:
                        p1_cam = p1_cam[z_ok]; p2_cam = p2_cam[z_ok]
                        conf_ok = conf[z_ok]
                        M_ok = p1_cam.shape[0]

                        ones = torch.ones((M_ok, 1), device=device, dtype=p1_cam.dtype)
                        p1_h = torch.cat([p1_cam, ones], dim=-1)
                        p2_h = torch.cat([p2_cam, ones], dim=-1)

                        p1_w = (p1_h @ Twc1_b.transpose(0, 1))[:, :3]
                        p2_w = (p2_h @ Twc2_b.transpose(0, 1))[:, :3]

                        l_pos = (p1_w - p2_w).abs().mean(dim=-1)
                        l_pos = _huber(l_pos, delta=0.05).clamp(max=1.0)
                        total_num += w_pos * (conf_ok * l_pos).sum()
                        total_den += conf_ok.sum()

                if w_shape > 0:
                    cov1_flat = F.grid_sample(
                        cov1_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv1_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1)
                    cov2_flat = F.grid_sample(
                        cov2_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv2_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1)

                    cov1_cam = cov1_flat.view(M, 3, 3)
                    cov2_cam = cov2_flat.view(M, 3, 3)

                    R1 = R1_b.unsqueeze(0).expand(M, 3, 3).contiguous()
                    R2 = R2_b.unsqueeze(0).expand(M, 3, 3).contiguous()

                    cov1_w = R1 @ cov1_cam @ R1.transpose(1, 2)
                    cov2_w = R2 @ cov2_cam @ R2.transpose(1, 2)

                    cov1_w = 0.5 * (cov1_w + cov1_w.transpose(1, 2))
                    cov2_w = 0.5 * (cov2_w + cov2_w.transpose(1, 2))

                    S1, U1 = torch.linalg.eigh(cov1_w)
                    S2, U2 = torch.linalg.eigh(cov2_w)
                    S1 = S1.clamp_min(eps_spd); S2 = S2.clamp_min(eps_spd)

                    s1 = torch.sqrt(S1)
                    s2 = torch.sqrt(S2)

                    l_shape = (torch.log(s1) - torch.log(s2)).abs().mean(dim=-1)
                    l_shape = _huber(l_shape, delta=0.05).clamp(max=1.0)
                    total_num += w_shape * (conf * l_shape).sum()
                    total_den += conf.sum()

                if w_opacity > 0 and (opac1_map is not None) and (opac2_map is not None):
                    o1 = F.grid_sample(
                        opac1_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv1_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1).view(-1)
                    o2 = F.grid_sample(
                        opac2_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv2_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1).view(-1)

                    l_opac = (o1 - o2).abs().clamp(max=1.0)
                    total_num += w_opacity * (conf * l_opac).sum()
                    total_den += conf.sum()

                if w_color > 0 and (sh1_map is not None) and (sh2_map is not None):
                    c1 = F.grid_sample(
                        sh1_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv1_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1)
                    c2 = F.grid_sample(
                        sh2_map[b_idx:b_idx+1].expand(M, -1, -1, -1),
                        uv2_norm, align_corners=True, mode="bilinear",
                    ).squeeze(-1).squeeze(-1)

                    if use_dc_only:
                        c1, c2 = c1[:, :3], c2[:, :3]

                    l_col = (c1 - c2).abs().mean(dim=-1).clamp(max=2.0)
                    total_num += w_color * (conf * l_col).sum()
                    total_den += conf.sum()

        final_loss = total_num / (total_den + 1e-8)

        reg = torch.tensor(0.0, device=device)
        if 'scales' in pred1 and 'scales' in pred2:
            with torch.no_grad():
                s1 = pred1['scales']
                s2 = pred2['scales']
                s_min = float(self.config.loss.get('scale_safe_min', 0.01))
                s_max = float(self.config.loss.get('scale_safe_max', 0.50))
                reg += F.relu(s1 - s_max).mean() + F.relu(s_min - s1).mean()
                reg += F.relu(s2 - s_max).mean() + F.relu(s_min - s2).mean()

                m1 = s1.mean()
                m2 = s2.mean()
                gap = (m1 - m2).abs()
                thresh = float(self.config.loss.get('scale_gap_thresh', 0.05))
                reg += F.relu(gap - thresh)

        lambda_reg = float(self.config.loss.get('consistency_reg_weight', 0.01))
        final_loss = final_loss + lambda_reg * reg
        self.log("train/cons/reg", (lambda_reg * reg).detach(), on_step=True, on_epoch=False)
        self.log("train/cons/den", total_den.detach(), on_step=True, on_epoch=False)

        return final_loss

    def calculate_loss(self, batch, view1, view2, pred1, pred2, matches_info, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):
        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color
        if apply_mask:
            predicted_color = predicted_color * mask[..., None, :, :]
            target_color = target_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        rgb_l2_loss = (predicted_color - target_color) ** 2
        mse_loss = (rgb_l2_loss.sum()) / (mask.sum() * 3 + 1e-8) if average_over_mask else rgb_l2_loss.mean()

        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / (flattened_mask.sum() + 1e-8) if average_over_mask else lpips_loss.mean()

        loss = self.config.loss.mse_loss_weight * mse_loss + self.config.loss.lpips_loss_weight * lpips_loss

        if self.config.loss.get('mast3r_loss_weight') is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        consistency_loss = torch.tensor(0.0, device=self.device)
        consistency_weight = self.config.loss.get('consistency_loss_weight', 0.2)
        if consistency_weight > 0 and matches_info is not None and matches_info['matches_px'].shape[0] > 0:
            consistency_loss = self.calculate_3d_consistency_loss(batch, matches_info, pred1, pred2)
            loss += consistency_weight * consistency_loss

        nvc_w = float(self.config.loss.get('novel_view_loss_weight', 0.0))
        if nvc_w > 0:
            warmup = int(self.config.loss.get('novel_view_warmup_steps', 2000))
            ramp = min(1.0, float(self.global_step) / max(1, warmup))
            nvc_loss, nvc_logs = self.calculate_novel_view_consistency_loss(
                batch, pred1, pred2, image_shape=(mask.shape[-2], mask.shape[-1]),
                matches_info=matches_info
            )
            loss = loss + (nvc_w * ramp) * nvc_loss
            if isinstance(nvc_loss, torch.Tensor):
                self.log('train/nvc_loss' if self.training else 'val/nvc_loss', nvc_loss.detach(), prog_bar=False)
            for k, v in nvc_logs.items():
                self.log(('train/' if self.training else 'val/') + k, v, prog_bar=False)

        ssim_val = None
        if calculate_ssim:
            ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=average_over_mask)
            if average_over_mask:
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / (flattened_mask.sum() + 1e-8)
            else:
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, consistency_loss, ssim_val

        return loss, mse_loss, lpips_loss, consistency_loss

    def _get_matches_info(self, pred1, pred2):
        matches_info = None
        consistency_weight = self.config.loss.get('consistency_loss_weight', 0.0)
        need_nvc = float(self.config.loss.get('novel_view_loss_weight', 0.0)) > 0
        if consistency_weight > 0 or need_nvc:
            with torch.no_grad():
                subsample_stride = self.config.loss.get('match_subsample_stride', 8)
                matches_px, batch_indices = find_matches_fast_reciprocal(
                    pred1['desc'], pred2['desc'],
                    subsample_stride=subsample_stride,
                )

                if matches_px.shape[0] == 0:
                    return None

                B, H, W, _ = pred1['desc'].shape
                conf1 = pred1.get('desc_conf', torch.ones(B, H, W, device=self.device))
                conf2 = pred2.get('desc_conf', torch.ones(B, H, W, device=self.device))

                uv1_px, uv2_px = matches_px[:, :2], matches_px[:, 2:]
                uv1_norm = (uv1_px / torch.tensor([W - 1, H - 1], device=self.device) * 2 - 1).view(len(uv1_px), 1, 1, 2)
                uv2_norm = (uv2_px / torch.tensor([W - 1, H - 1], device=self.device) * 2 - 1).view(len(uv2_px), 1, 1, 2)

                match_conf1 = F.grid_sample(conf1[batch_indices].unsqueeze(1), uv1_norm, align_corners=True, mode='bilinear').view(-1)
                match_conf2 = F.grid_sample(conf2[batch_indices].unsqueeze(1), uv2_norm, align_corners=True, mode='bilinear').view(-1)

                confidence = (match_conf1 * match_conf2).clamp_min(0)

                conf_min = float(self.config.loss.get('match_conf_min', 0.05))
                good = confidence >= conf_min
                if good.sum() == 0:
                    return None
                matches_px = matches_px[good]
                batch_indices = batch_indices[good]
                confidence = confidence[good]

                max_per_batch = int(self.config.loss.get('match_max_per_batch', 4096))
                if matches_px.shape[0] > max_per_batch:
                    idx = torch.randperm(matches_px.shape[0], device=matches_px.device)[:max_per_batch]
                    matches_px = matches_px[idx]
                    batch_indices = batch_indices[idx]
                    confidence = confidence[idx]

                matches_info = {
                    'matches_px': matches_px,
                    'confidence': confidence,
                    'batch_indices': batch_indices
                }
        return matches_info

    def _is_batch_corrupted_on_any_rank(self, batch) -> bool:
        is_corrupted_local = 1.0 if isinstance(batch, dict) and batch.get('is_corrupted_batch') else 0.0

        status_tensor = torch.tensor(is_corrupted_local, device=self.device)

        torch.distributed.all_reduce(status_tensor, op=torch.distributed.ReduceOp.SUM)

        return status_tensor.item() > 0

    def training_step(self, batch, batch_idx):
        if self._is_batch_corrupted_on_any_rank(batch):
            if self.global_rank == 0:
                logging.warning(
                    f"Skipping training batch at batch_idx {batch_idx} across all ranks due to data corruption."
                )
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if batch is None:
            logging.warning(f"Skipping an entirely corrupted training batch at batch_idx {batch_idx}.")
            return None

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        pred1, pred2 = self.forward(view1, view2)

        if self.global_step % self.config.get('debug_log_freq', 200) == 0:
            self.debug_gaussian_params(pred1, "train/view1")
            self.debug_gaussian_params(pred2, "train/view2")

        matches_info = self._get_matches_info(pred1, pred2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))
        mask = loss_mask.calculate_loss_mask(batch)

        loss, mse, lpips_v, consistency_loss = self.calculate_loss(
            batch, view1, view2, pred1, pred2, matches_info, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask
        )

        self._histogram_summary(pred1['scales'], "train/view1/scale_dist")
        self._histogram_summary(pred2['scales'], "train/view2/scale_dist")
        self._track_scale_drift(pred1, pred2, "train")

        self._check_nan_inf(loss, mse, lpips_v, consistency_loss)
        self.log_metrics('train', loss, mse, lpips_v, consistency_loss=consistency_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self._is_batch_corrupted_on_any_rank(batch):
            if self.global_rank == 0:
                logging.warning(
                    f"Skipping validation batch at batch_idx {batch_idx} across all ranks due to data corruption."
                )
            return None

        if batch is None:
            logging.warning(f"Skipping an entirely corrupted validation batch at batch_idx {batch_idx}.")
            return None

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        pred1, pred2 = self.forward(view1, view2)

        if batch_idx % self.config.get('debug_log_freq_val', 20) == 0:
            self.debug_gaussian_params(pred1, "val/view1")
            self.debug_gaussian_params(pred2, "val/view2")

        matches_info = self._get_matches_info(pred1, pred2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))
        mask = loss_mask.calculate_loss_mask(batch)

        loss, mse, lpips_v, consistency_loss = self.calculate_loss(
            batch, view1, view2, pred1, pred2, matches_info, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask
        )
        self._check_nan_inf(loss, mse, lpips_v, consistency_loss)
        self.log_metrics('val', loss, mse, lpips_v, consistency_loss=consistency_loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self._is_batch_corrupted_on_any_rank(batch):
            if self.global_rank == 0:
                logging.warning(
                    f"Skipping test batch at batch_idx {batch_idx} for dataloader {dataloader_idx} across all ranks due to data corruption."
                )
            return None

        if batch is None:
            logging.warning(f"Skipping an entirely corrupted test batch at batch_idx {batch_idx}.")
            return None

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)

        matches_info = self._get_matches_info(pred1, pred2)

        with self.benchmarker.time("decoder", num_calls=len(batch['target'])):
            color, _ = self.decoder(batch, pred1, pred2, (h, w))
        mask = loss_mask.calculate_loss_mask(batch)

        loss, mse, lpips_v, consistency_loss, ssim = self.calculate_loss(
            batch, view1, view2, pred1, pred2, matches_info, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True
        )
        self._check_nan_inf(loss, mse, lpips_v, consistency_loss, ssim)

        test_dataset_key = list(self.test_datasets.keys())[dataloader_idx]
        log_prefix = f'test_{test_dataset_key}'
        self.log_metrics(log_prefix, loss, mse, lpips_v, ssim=ssim, consistency_loss=consistency_loss)
        return loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None, consistency_loss=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }
        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim
        if consistency_loss is not None and consistency_loss.item() > 0:
            values[f'{prefix}/consistency_loss'] = consistency_loss

        self.log_dict(values, prog_bar=(prefix != 'val'), sync_dist=(prefix != 'train'), batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(self, *args, **kwargs):
        if len(args) == 0:
            return
        optimizer = args[0]
        optimizer_idx = args[1] if len(args) >= 2 else kwargs.get("optimizer_idx", 0)

        cfg_clip_val = float(getattr(self.config.opt, "gradient_clip_val", 0.0))
        clip_val = float(kwargs.get("gradient_clip_val", cfg_clip_val))
        algo = kwargs.get("gradient_clip_algorithm", "norm")

        if clip_val <= 0:
            return

        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm=algo)

    def on_after_backward(self):
        self._check_grad_norms(self.named_parameters(), "train/grad")

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(benchmark_file_path)

def collate_fn_skip_corrupted(batch):
    original_size = len(batch)
    batch = list(filter(lambda x: x is not None, batch))

    if not batch:
        return {'is_corrupted_batch': True, 'original_batch_size': original_size}

    return torch.utils.data.dataloader.default_collate(batch)

def run_experiment(config):
    L.seed_everything(config.seed, workers=True)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
    is_rank0 = (rank == 0)

    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []

    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=config.name
        )
        loggers.append(csv_logger)

    if getattr(config.loggers, "wandb_offline", False):
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    wandb_logger_ref = None
    if config.loggers.use_wandb and is_rank0:
        try:
            import wandb as _wandb
            settings = _wandb.Settings(start_method="thread", _service_wait=300)
            wandb_logger = L.pytorch.loggers.WandbLogger(
                project='splatt3r',
                name=config.name,
                save_dir=config.save_dir,
                config=omegaconf.OmegaConf.to_container(config),
                settings=settings,
                log_model=False,
            )
            if _wandb.run is not None and getattr(config.loggers, "log_code", False):
                _wandb.run.log_code(".")
            loggers.append(wandb_logger)
            wandb_logger_ref = wandb_logger
        except Exception as e:
            print(f"[W&B] init failed on rank0: {e}. Falling back to CSV only.")

    profiler = None
    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_stack=True
        )

    print('Loading Model')
    model = MAST3RGaussians(config)
    if config.use_pretrained:
        if not os.path.exists(config.pretrained_mast3r_path):
            raise FileNotFoundError(f"Pretrained model not found at: {config.pretrained_mast3r_path}")
        ckpt = torch.load(config.pretrained_mast3r_path, map_location='cpu')
        new_state_dict = {}
        prefix = 'encoder.'
        for k, v in ckpt['state_dict'].items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
        _ = model.encoder.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded pretrained encoder weights.")
        del ckpt

    try:
        print('Starting Training...')
        trainer = L.Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=[
                L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
                export.SaveBatchData(save_dir=config.save_dir),
            ],
            check_val_every_n_epoch=1,
            default_root_dir=config.save_dir,
            devices=config.devices,
            gradient_clip_val=config.opt.gradient_clip_val,
            log_every_n_steps=10,
            logger=loggers,
            max_epochs=config.opt.epochs,
            profiler=profiler,
            strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
        )
        trainer.fit(model)

        print("Starting Testing...")
        results = trainer.test(model)

        if is_rank0:
            final_results = {}
            test_dataset_keys = list(model.test_datasets.keys())
            for i, res_dict in enumerate(results):
                test_name = test_dataset_keys[i]
                final_results[test_name] = res_dict

            save_path = os.path.join(config.save_dir, 'results.json')
            with open(save_path, 'w') as f:
                json.dump(final_results, f, indent=4)
            print("All experiments finished. Results saved.")

    finally:
        try:
            if is_rank0 and wandb_logger_ref is not None and wandb.run is not None:
                wandb.finish()
        except Exception as e:
            print(f"[W&B] finish error ignored: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_config_yaml> [optional_overrides]")
        sys.exit(1)

    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    run_experiment(config)
