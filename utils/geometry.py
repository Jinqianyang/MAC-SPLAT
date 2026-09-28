import einops
import torch


def normalize_intrinsics(intrinsics, image_shape):
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= image_shape[1]
    intrinsics[..., 1, :] /= image_shape[0]
    return intrinsics


def unnormalize_intrinsics(intrinsics, image_shape):
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] *= image_shape[1]
    intrinsics[..., 1, :] *= image_shape[0]
    return intrinsics


def quaternion_to_matrix(quaternions, eps: float = 1e-8):

    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(scale, rotation_xyzw):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ einops.rearrange(scale, "... i j -> ... j i")
        @ einops.rearrange(rotation, "... i j -> ... j i")
    )


def homogenize_points(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):

    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )

    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)

    world_points = camera_space_to_world_space(camera_points, c2w)

    return world_points
