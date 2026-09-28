import einops
import torch

from utils.geometry import unproject_depth, world_space_to_camera_space, camera_space_to_pixel_space


@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):

    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)

    camera_points = world_space_to_camera_space(points_3d, c2w_2)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)

    rendered_depth = camera_points[..., 2]


    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )
    in_frustum_mask = in_frustum_mask.any(dim=-3)

    non_zero_depth = depth_1 > 1e-6

    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
    for b in range(depth_1.shape[0]):
        for i in range(v1):
            for j in range(v2):
                depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j], sampled_depths, atol=1e-1)

    matching_depth = matching_depth.any(dim=-3)

    mask = in_frustum_mask & non_zero_depth & matching_depth
    return mask


@torch.no_grad()
def calculate_loss_mask(batch):

    target_depth = torch.stack([target_view['depthmap'] for target_view in batch['target']], dim=1)
    target_intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
    target_c2w = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
    context_depth = torch.stack([context_view['depthmap'] for context_view in batch['context']], dim=1)
    context_intrinsics = torch.stack([context_view['camera_intrinsics'] for context_view in batch['context']], dim=1)
    context_c2w = torch.stack([context_view['camera_pose'] for context_view in batch['context']], dim=1)

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    return mask
