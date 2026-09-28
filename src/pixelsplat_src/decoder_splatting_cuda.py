import torch
from torch import nn
from einops import rearrange, repeat
from math import isqrt, sqrt

from gsplat.rendering import rasterization
from utils.geometry import normalize_intrinsics


class DecoderSplattingCUDA(torch.nn.Module):
    def __init__(self, background_color, scale_booster: float = 1.0):
        super().__init__()
        self.scale_booster = scale_booster
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, batch, pred1, pred2, image_shape):

        base_pose = batch['context'][0]['camera_pose']
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
        intrinsics_normalized = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
        intrinsics_normalized = normalize_intrinsics(intrinsics_normalized, image_shape)[..., :3, :3]

        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics

        means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)
        opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

        B, V, _, _ = extrinsics.shape
        H, W = image_shape
        b, v, _, _ = extrinsics.shape

        all_rendered_imgs = []
        for i in range(B):

            means_i = rearrange(means[i], "v h w xyz -> (v h w) xyz")
            covariances_i = rearrange(covariances[i], "v h w i j -> (v h w) i j")
            harmonics_i = rearrange(harmonics[i], "v h w c d_sh -> (v h w) c d_sh")
            opacities_i = rearrange(opacities[i], "v h w 1 -> (v h w)")

            color = harmonics_i.permute(0, 2, 1).contiguous()


            ssh_degree = (int(sqrt(color.shape[-2])) - 1)

            intrinsics_i = intrinsics_normalized[i].clone()
            intrinsics_i[:, 0, 0] *= W
            intrinsics_i[:, 1, 1] *= H
            intrinsics_i[:, 0, 2] *= W
            intrinsics_i[:, 1, 2] *= H
            backgrounds_batched = self.background_color.repeat(V, 1)

            rendering, _ , _ = rasterization(
                means=means_i,
                covars=covariances_i,
                colors=color,
                opacities=opacities_i,
                viewmats=torch.inverse(extrinsics[i]),
                Ks=intrinsics_i,
                width=W,
                height=H,
                sh_degree=ssh_degree,
                backgrounds=backgrounds_batched,
                near_plane=0.1,
                far_plane=1000.0,
                packed=False,
                quats=None,
                scales=None,
            )

            rendered_img_i = rearrange(rendering, "v h w c -> v c h w")
            all_rendered_imgs.append(rendered_img_i)

        final_color = torch.cat(all_rendered_imgs, dim=0)
        final_color = rearrange(final_color, "(b v) c h w -> b v c h w", b=B, v=V)

        return final_color, None
