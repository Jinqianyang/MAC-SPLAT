import torch


def fill_default_args(kwargs, func):
    import inspect
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1['instance']
    y = gt2['instance']
    if len(x) == len(y) and len(x) == 1:
        return False
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):

    if not activate:
        def wrapper_no(decout, true_shape, *extra, **kwextra):
            H, W = true_shape[0].cpu().tolist()
            return head(decout, (H, W), *extra, **kwextra)
        return wrapper_no

    def wrapper_yes(decout, true_shape, *extra, **kwextra):
        B = len(true_shape)
        height, width = true_shape.T
        is_land = width >= height
        is_port = ~is_land
        H, W = int(true_shape.min()), int(true_shape.max())

        if is_land.all():
            return head(decout, (H, W), *extra, **kwextra)

        if is_port.all():
            return transposed(head(decout, (W, H), *extra, **kwextra))

        def pick(mask):
            return [d[mask] for d in decout]

        out_l = head(pick(is_land), (H, W), *extra, **kwextra)
        out_p = transposed(head(pick(is_port), (W, H), *extra, **kwextra))

        out = {}
        for k in out_l | out_p:
            buf = out_l[k].new(B, *out_l[k].shape[1:])
            buf[is_land] = out_l[k]
            buf[is_port] = out_p[k]
            out[k] = buf
        return out

    return wrapper_yes


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz
