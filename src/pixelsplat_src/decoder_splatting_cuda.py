import torch
from torch import nn
from einops import rearrange, repeat
from math import isqrt, sqrt

# 直接导入 gsplat 的渲染器
from gsplat.rendering import rasterization
from utils.geometry import normalize_intrinsics


class DecoderSplattingCUDA(torch.nn.Module):
    """
    一个高斯溅射解码器，其 forward 方法遵循您的基线逻辑，
    但使用一个修正过的、直接的 gsplat 调用来替换 render_cuda 辅助函数。
    """
    def __init__(self, background_color, scale_booster: float = 1.0):
        """
        Args:
            background_color (list[float]): 背景颜色 [r, g, b].
            scale_booster (float): 一个乘数，用于放大高斯球的尺寸以确保可见性。
                                   如果渲染结果依然是黑点，请增大此值 (例如 10.0, 100.0)。
        """
        super().__init__()
        self.scale_booster = scale_booster
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
    
    def forward(self, batch, pred1, pred2, image_shape):
        
        base_pose = batch['context'][0]['camera_pose'] # [B, 4, 4]
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
        # 我们对批次 B 进行循环，因为每个元素可能代表一个不同的高斯场景
        for i in range(B):
            # --- 为当前场景 i 准备数据 ---
            
            # 2a. 扁平化空间维度 (H, W) 和视角维度 (V) 以创建高斯列表
            # 原始: [V, H, W, Dims], 目标: [G, Dims] 其中 G = V * H * W
            means_i = rearrange(means[i], "v h w xyz -> (v h w) xyz")
            covariances_i = rearrange(covariances[i], "v h w i j -> (v h w) i j")
            harmonics_i = rearrange(harmonics[i], "v h w c d_sh -> (v h w) c d_sh")
            opacities_i = rearrange(opacities[i], "v h w 1 -> (v h w)")

            color = harmonics_i.permute(0, 2, 1).contiguous()

            

            ssh_degree = (int(sqrt(color.shape[-2])) - 1)  # 禁用gsplat内部的SH求值
            # 修正C: Opacity激活
            #opacities_i = torch.sigmoid(opacities_i)
            
            # 修正D: 相机内参反归一化 (使用标准方法)
            intrinsics_i = intrinsics_normalized[i].clone()
            intrinsics_i[:, 0, 0] *= W  # fx
            intrinsics_i[:, 1, 1] *= H  # fy
            intrinsics_i[:, 0, 2] *= W  # cx
            intrinsics_i[:, 1, 2] *= H  # cy
            backgrounds_batched = self.background_color.repeat(V, 1) # Shape: [V, 3]

            # --- 2c. 调用 gsplat.rendering.rasterization ---
            # gsplat可以一次性处理同一场景的多个视角 (V)
            rendering, _ , _ = rasterization(
                means=means_i,
                covars=covariances_i,
                colors=color,
                opacities=opacities_i,
                viewmats=torch.inverse(extrinsics[i]), # 传入当前场景的V个视角
                Ks=intrinsics_i,                     # 传入当前场景的V个内参
                width=W,
                height=H,
                sh_degree=ssh_degree,
                backgrounds=backgrounds_batched,
                near_plane=0.1, # 使用固定的、合理的near/far值
                far_plane=1000.0,
                packed=False,
                quats=None,   # 明确不使用quats/scales
                scales=None,
            )

            # 结果的形状是 [V, H, W, 3], 转换为 PyTorch 标准格式 [V, C, H, W]
            rendered_img_i = rearrange(rendering, "v h w c -> v c h w")
            all_rendered_imgs.append(rendered_img_i)

        # 将每个场景的结果重新组合成一个完整的批次
        final_color = torch.cat(all_rendered_imgs, dim=0)
        final_color = rearrange(final_color, "(b v) c h w -> b v c h w", b=B, v=V)

        # 您的基线返回 (color, None)，我们保持一致
        return final_color, None