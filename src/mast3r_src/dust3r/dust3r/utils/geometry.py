import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch.nn.functional as F
from src.mast3r_src.dust3r.dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from src.mast3r_src.dust3r.dust3r.utils.device import to_numpy


def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    if device is None:
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def colmap_to_opencv_intrinsics(K):
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def normalize_pointcloud(pts1, pts2, norm_mode='avg_dis', valid1=None, valid2=None, ret_factor=False):
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1:].view(-1, H2, W2, 1)
            all_dis = log_dis
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        res = res + (norm_factor,)
    return res


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1) if z2 is not None else None
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True):
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3) if pts2 is not None else None
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values
    if z_only:
        _center[..., :2] = 0

    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))

def find_mutual_best_matches(desc1, desc2, desc_conf1=None, desc_conf2=None, subsample=None):
    B, H, W, D = desc1.shape
    device = desc1.device

    desc1_flat = desc1.view(B, -1, D)
    desc2_flat = desc2.view(B, -1, D)

    if subsample is not None and subsample < H * W:
        p1_indices = torch.stack([torch.randperm(H * W, device=device)[:subsample] for _ in range(B)])
        p2_indices = torch.stack([torch.randperm(H * W, device=device)[:subsample] for _ in range(B)])

        desc1_sub = torch.gather(desc1_flat, 1, p1_indices.unsqueeze(-1).expand(-1, -1, D))
        desc2_sub = torch.gather(desc2_flat, 1, p2_indices.unsqueeze(-1).expand(-1, -1, D))
    else:
        p1_indices = torch.arange(H * W, device=device).unsqueeze(0).expand(B, -1)
        p2_indices = p1_indices
        desc1_sub = desc1_flat
        desc2_sub = desc2_flat

    dist_matrix = torch.cdist(desc1_sub, desc2_sub)

    _, best_idx_1_to_2 = torch.min(dist_matrix, dim=2)
    _, best_idx_2_to_1 = torch.min(dist_matrix, dim=1)

    arange_sub1 = torch.arange(desc1_sub.shape[1], device=device).unsqueeze(0)
    mutual_mask = arange_sub1 == torch.gather(best_idx_2_to_1, 1, best_idx_1_to_2)

    batch_idx_match, p1_sub_indices_match = torch.where(mutual_mask)
    p2_sub_indices_match = best_idx_1_to_2[batch_idx_match, p1_sub_indices_match]

    p1_orig_indices = torch.gather(p1_indices, 0, batch_idx_match.unsqueeze(-1).expand(-1, p1_indices.shape[1]))[torch.arange(len(batch_idx_match)), p1_sub_indices_match]
    p2_orig_indices = torch.gather(p2_indices, 0, batch_idx_match.unsqueeze(-1).expand(-1, p2_indices.shape[1]))[torch.arange(len(batch_idx_match)), p2_sub_indices_match]

    p1_u, p1_v = p1_orig_indices % W, p1_orig_indices // W
    p2_u, p2_v = p2_orig_indices % W, p2_orig_indices // W

    matches_px = torch.stack([p1_u, p1_v, p2_u, p2_v], dim=1).float()

    confidence = torch.ones_like(batch_idx_match, dtype=torch.float32)
    if desc_conf1 is not None and desc_conf2 is not None:
        conf1_flat = desc_conf1.view(B, -1)
        conf2_flat = desc_conf2.view(B, -1)

        match_conf1 = conf1_flat[batch_idx_match, p1_orig_indices]
        match_conf2 = conf2_flat[batch_idx_match, p2_orig_indices]
        confidence = match_conf1 * match_conf2

    return matches_px, confidence, batch_idx_match
