import torch
import numpy as np
import einops

# 假设 fast_reciprocal_NNs 位于您提供的路径
from src.mast3r_src.mast3r.fast_nn import fast_reciprocal_NNs

@torch.no_grad()
def find_matches_fast_reciprocal(desc1, desc2, subsample_stride=8):
    """
    一个高效的封装函数，用于在两个描述子张量之间寻找可靠的相互匹配。

    Args:
        desc1 (torch.Tensor): 视图1的描述子张量，形状为 [B, H, W, D]。
        desc2 (torch.Tensor): 视图2的描述子张量，形状为 [B, H, W, D]。
        subsample_stride (int): 在进行匹配时使用的下采样步长，以提高效率。

    Returns:
        torch.Tensor: 匹配的像素坐标 [N, 4]，格式为 (u1, v1, u2, v2)。
        torch.Tensor: 每个匹配对应的批次索引 [N]。
    """
    B, H, W, D = desc1.shape
    device = desc1.device

    # 存储所有批次的匹配结果
    all_matches_px = []
    all_batch_indices = []

    for b_idx in range(B):
        # 提取当前批次的描述子
        d1 = desc1[b_idx]  # Shape: [H, W, D]
        d2 = desc2[b_idx]  # Shape: [H, W, D]

        # 调用 MASt3R 的快速匹配算法
        # ret_xy=False 返回的是扁平化的一维索引
        p1_indices, p2_indices = fast_reciprocal_NNs(
            d1, d2,
            subsample_or_initxy1=subsample_stride,
            ret_xy=False,
            device=device,
            dist='dot'  # 对于归一化的描述子，点积更高效
        )

        if len(p1_indices) == 0:
            continue

        # 将一维索引转换回二维像素坐标 (u, v)
        p1_v, p1_u = np.unravel_index(p1_indices, (H, W))
        p2_v, p2_u = np.unravel_index(p2_indices, (H, W))

        # 组合成 (u1, v1, u2, v2) 的格式
        matches_px = np.stack([p1_u, p1_v, p2_u, p2_v], axis=-1)
        
        all_matches_px.append(torch.from_numpy(matches_px).to(device))
        all_batch_indices.append(torch.full((len(matches_px),), b_idx, device=device, dtype=torch.long))

    if not all_matches_px:
        return torch.empty(0, 4, device=device), torch.empty(0, device=device, dtype=torch.long)

    # 将所有批次的结果连接起来
    final_matches_px = torch.cat(all_matches_px, dim=0)
    final_batch_indices = torch.cat(all_batch_indices, dim=0)

    return final_matches_px, final_batch_indices