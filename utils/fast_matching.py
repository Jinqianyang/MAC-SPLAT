import torch
import numpy as np
import einops

from src.mast3r_src.mast3r.fast_nn import fast_reciprocal_NNs

@torch.no_grad()
def find_matches_fast_reciprocal(desc1, desc2, subsample_stride=8):
    B, H, W, D = desc1.shape
    device = desc1.device

    all_matches_px = []
    all_batch_indices = []

    for b_idx in range(B):
        d1 = desc1[b_idx]
        d2 = desc2[b_idx]

        p1_indices, p2_indices = fast_reciprocal_NNs(
            d1, d2,
            subsample_or_initxy1=subsample_stride,
            ret_xy=False,
            device=device,
            dist='dot'
        )

        if len(p1_indices) == 0:
            continue

        p1_v, p1_u = np.unravel_index(p1_indices, (H, W))
        p2_v, p2_u = np.unravel_index(p2_indices, (H, W))

        matches_px = np.stack([p1_u, p1_v, p2_u, p2_v], axis=-1)

        all_matches_px.append(torch.from_numpy(matches_px).to(device))
        all_batch_indices.append(torch.full((len(matches_px),), b_idx, device=device, dtype=torch.long))

    if not all_matches_px:
        return torch.empty(0, 4, device=device), torch.empty(0, device=device, dtype=torch.long)

    final_matches_px = torch.cat(all_matches_px, dim=0)
    final_batch_indices = torch.cat(all_batch_indices, dim=0)

    return final_matches_px, final_batch_indices
