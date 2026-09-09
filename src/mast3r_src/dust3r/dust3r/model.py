
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
import inspect # For fill_default_args if it needs it

# Your relative imports
from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory # This will import the modified head_factory from your heads file
from dust3r.patch_embed import get_patch_embed # Assuming this path is correct

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa: Make sure this is the correct CroCoNet class

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def DBG(tag, *msgs): # Make DBG globally available or import if it's in misc
    if (not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0):
        print(f"[{tag}]", *msgs)

def load_model(model_path, device, verbose=True):
    if verbose:
        DBG("Load", f"... loading model from {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    if 'args' not in ckpt or not hasattr(ckpt['args'], 'model'):
        raise ValueError("Checkpoint 'args' or 'args.model' not found. Check checkpoint structure.")
    
    args_str = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args_str:
        if args_str.endswith(')'):
            args_str = args_str[:-1] + ', landscape_only=False)'
        else:
            DBG("Load Warning", f"Cannot easily append landscape_only to args_str: {args_str}")
    else:
        args_str = args_str.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    
    if "landscape_only=False" not in args_str:
            DBG("Load Warning", f"landscape_only=False not in args_str after manipulation: {args_str}")

    if verbose:
        DBG("Load", f"instantiating with args_str: {args_str}")
    
    try:
        # Ensure AsymmetricCroCo3DStereo is in the global scope for eval if not already
        # This is usually the case if load_model is in the same file as the class definition.
        # global AsymmetricCroCo3DStereo # Generally not needed if defined in same module
        net = eval(args_str)
    except Exception as e:
        DBG("Load Error", f"Failed to eval args_str: {args_str}. Error: {e}")
        raise e
        
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        DBG("Load", f"State dict load status: {s}")
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet, 
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r", 
    tags=["image-to-3d"], 
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='gaussian_head', 
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True, 
                 patch_embed_cls='PatchEmbedDust3R',
                 **croco_kwargs): # This should contain enc_embed_dim, img_size, patch_size etc. for CroCoNet

        self.patch_embed_cls = patch_embed_cls 

        self.croco_args = fill_default_args(croco_kwargs, CroCoNet.__init__)
        
        super().__init__(**self.croco_args) # Calls CroCoNet.__init__

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        head_specific_kwargs = {}
        head_param_keys = [
            "use_offsets", "sh_degree", "head_use_dino", "head_use_dino_reducer",
            "head_dino_reducer_dim_out", "head_dino_model_name", "head_dino_repo_path",
            "head_dino_input_target_size", "head_dino_freeze_weights"
        ]
        default_head_params = {
            "use_offsets": False, "sh_degree": 1, "head_use_dino": True,
            "head_use_dino_reducer": False, "head_dino_reducer_dim_out": 128,
            "head_dino_model_name": 'dinov2_vits14',
            "head_dino_repo_path": os.environ.get('DINOV2_REPO', 'third_party/dinov2'),
            "head_dino_input_target_size": (518,518),
            "head_dino_freeze_weights": True
        }

        for key in head_param_keys:
            head_specific_kwargs[key] = self.croco_args.get(key, default_head_params.get(key))
        
        for k, v in self.croco_args.items():
            if k not in head_specific_kwargs and k not in [
                'output_mode', 'head_type', 'landscape_only', 'depth_mode', 'conf_mode',
                'patch_size', 'img_size', 'enc_embed_dim', 'dec_embed_dim', 'patch_embed_cls', 'freeze' 
            ]:
                head_specific_kwargs[k] = v

        self.set_downstream_head(
            output_mode=output_mode,
            head_type=head_type,
            landscape_only=landscape_only, 
            depth_mode=depth_mode,
            conf_mode=conf_mode,
            patch_size=self.patch_size, 
            img_size=self.img_size,   
            **head_specific_kwargs 
        )
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        device = kw.pop('device', 'cpu') 
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device=device, **kw) 
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size, patch_size, enc_embed_dim): 
        dim_to_use_for_patch_embed = getattr(self, 'enc_embed_dim', None)
        if dim_to_use_for_patch_embed is None:
            DBG("_set_patch_embed Warning", f"self.enc_embed_dim not set by CroCoNet prior to this call. Using enc_embed_dim argument: {enc_embed_dim}. Check CroCoNet __init__ and arg propagation.")
            dim_to_use_for_patch_embed = enc_embed_dim 
        
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, dim_to_use_for_patch_embed)
        
        if not hasattr(self, 'img_size') or self.img_size != img_size :
             DBG("_set_patch_embed Sync", f"Updating self.img_size from {getattr(self, 'img_size', 'N/A')} to {img_size}")
             self.img_size = img_size
        if not hasattr(self, 'patch_size') or self.patch_size != patch_size:
             DBG("_set_patch_embed Sync", f"Updating self.patch_size from {getattr(self, 'patch_size', 'N/A')} to {patch_size}")
             self.patch_size = patch_size


    def load_state_dict(self, ckpt, **kw):
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2.') for k in new_ckpt): 
            DBG("Load SD", "Duplicating decoder weights for dec_blocks2.")
            keys_to_process = list(ckpt.keys())
            for key in keys_to_process:
                if key.startswith('dec_blocks.'):
                    new_key = key.replace('dec_blocks.', 'dec_blocks2.', 1)
                    new_ckpt[new_key] = ckpt[key] 
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):
        self.freeze = freeze
        modules_to_freeze = []
        if freeze == 'mask':
            if hasattr(self, 'mask_token') and self.mask_token is not None:
                modules_to_freeze.append(self.mask_token)
        elif freeze == 'encoder':
            if hasattr(self, 'mask_token') and self.mask_token is not None:
                modules_to_freeze.append(self.mask_token)
            if hasattr(self, 'patch_embed') and self.patch_embed is not None:
                modules_to_freeze.append(self.patch_embed)
            if hasattr(self, 'enc_blocks') and self.enc_blocks is not None:
                modules_to_freeze.append(self.enc_blocks) 

        if modules_to_freeze:
            DBG("Freeze", f"Freezing parameters for: {freeze}")
            freeze_all_params(modules_to_freeze)
        elif freeze != 'none':
            DBG("Freeze", f"No parameters specified or modules not found for freeze='{freeze}'.")


    def _set_prediction_head(self, *args, **kwargs):
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only,
                            depth_mode, conf_mode, patch_size, img_size,
                            **kw): 
        if not isinstance(img_size, (list, tuple)):
            img_size = (img_size, img_size)
        if not (isinstance(img_size[0], int) and isinstance(img_size[1], int)):
            try:
                img_size = (int(img_size[0]), int(img_size[1]))
            except:
                 raise ValueError(f"img_size must be a tuple of int or convertible, got {img_size} of type {type(img_size)}")

        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'img_size {img_size} must be multiple of patch_size {patch_size}'

        self.output_mode = output_mode
        self.head_type = head_type 
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        kw.pop('img_size', None)
        kw.pop('patch_size', None)

        head_factory_kwargs = {
            "has_conf": bool(conf_mode), 
            "use_offsets": kw.get("use_offsets", False), 
            "sh_degree": kw.get("sh_degree", 1),
            "use_dino": kw.get("head_use_dino", True),
            "dino_model_name": kw.get("head_dino_model_name", 'dinov2_vits14'),
            "dino_repo_path": kw.get("head_dino_repo_path", os.environ.get('DINOV2_REPO', 'third_party/dinov2')),
            "dino_input_target_size": kw.get("head_dino_input_target_size", (518, 518)),
            "dino_freeze_weights": kw.get("head_dino_freeze_weights", True),
            "use_dino_reducer": kw.get("head_use_dino_reducer", False),
            "dino_reducer_dim_out": kw.get("head_dino_reducer_dim_out", 128),
        }
        
        for k, v in kw.items():
            if k not in head_factory_kwargs and k not in ["output_mode", "head_type", "landscape_only", "depth_mode", "conf_mode"]: 
                head_factory_kwargs[k] = v

        self.downstream_head1 = head_factory(head_type, output_mode, self, **head_factory_kwargs)
        self.downstream_head2 = head_factory(head_type, output_mode, self, **head_factory_kwargs)

        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)



    def _encode_image(self, image, true_shape):
        if isinstance(true_shape, torch.Tensor):
            true_shape = true_shape.to(image.device)
        if not hasattr(self, 'patch_embed') or self.patch_embed is None:
            raise RuntimeError("self.patch_embed is not initialized before _encode_image. Check _set_patch_embed call order.")
        x, pos = self.patch_embed(image, true_shape=true_shape)

        assert self.enc_pos_embed is None

        for blk in self.enc_blocks:
            x = blk(x, pos) # pos is passed here, its dtype matters for RoPE

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if isinstance(true_shape1, torch.Tensor): true_shape1 = true_shape1.to(img1.device)
        if isinstance(true_shape2, torch.Tensor): true_shape2 = true_shape2.to(img2.device)

        def ensure_batch_dim(shape_tensor, ref_tensor):
            if ref_tensor.ndim == 0: # Should not happen for image tensors
                return shape_tensor
            if shape_tensor.ndim == 1 and ref_tensor.ndim > 1 and ref_tensor.shape[0] > 0 : 
                shape_tensor = shape_tensor.unsqueeze(0).repeat(ref_tensor.shape[0], 1)
            elif shape_tensor.ndim == 2 and ref_tensor.ndim > 1 and ref_tensor.shape[0] > 0 and shape_tensor.shape[0] != ref_tensor.shape[0]:
                 shape_tensor = shape_tensor[0:1].repeat(ref_tensor.shape[0],1) 
            return shape_tensor

        true_shape1 = ensure_batch_dim(true_shape1, img1)
        true_shape2 = ensure_batch_dim(true_shape2, img2)

        if img1.shape[-2:] == img2.shape[-2:]:
            combined_images = torch.cat((img1, img2), dim=0)
            combined_true_shapes = torch.cat((true_shape1, true_shape2), dim=0)
            out_combined, pos_combined, _ = self._encode_image(combined_images, combined_true_shapes)
            out1, out2 = out_combined.chunk(2, dim=0)
            pos1, pos2 = pos_combined.chunk(2, dim=0)
        else:
            out1, pos1, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out1, out2, pos1, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        def get_valid_true_shape(view_img_tensor, view_dict_item_name, view_dict):
            batch_size_current = view_img_tensor.shape[0]
            default_shape_tensor = torch.tensor(view_img_tensor.shape[-2:], device=view_img_tensor.device, dtype=torch.long)
            default_shape_batch = default_shape_tensor.unsqueeze(0).repeat(batch_size_current, 1) if batch_size_current > 0 else torch.empty((0,2), device=view_img_tensor.device, dtype=torch.long)

            true_shape_val = view_dict.get(view_dict_item_name, default_shape_batch)

            if not isinstance(true_shape_val, torch.Tensor):
                try:
                    true_shape_val = torch.tensor(true_shape_val, device=view_img_tensor.device, dtype=torch.long)
                except Exception as e:
                    DBG("_encode_symmetrized Warning", f"Failed to convert true_shape {true_shape_val} to tensor. Using default. Error: {e}")
                    return default_shape_batch

            if true_shape_val.ndim == 1 and batch_size_current > 0 :
                true_shape_val = true_shape_val.unsqueeze(0).repeat(batch_size_current,1)
            elif true_shape_val.ndim == 2 and batch_size_current > 0 and true_shape_val.shape[0] != batch_size_current :
                if true_shape_val.shape[0] == 1: 
                     true_shape_val = true_shape_val.repeat(batch_size_current,1)
                else: 
                    DBG("_encode_symmetrized Warning", f"true_shape batch size mismatch ({true_shape_val.shape[0]} vs {batch_size_current}). Using default.")
                    return default_shape_batch
            elif true_shape_val.ndim != 2 or (true_shape_val.ndim == 2 and true_shape_val.shape[1] != 2) :
                if batch_size_current == 0 and true_shape_val.numel() == 0 and true_shape_val.shape[1] == 2: 
                    pass
                else:
                    DBG("_encode_symmetrized Warning", f"true_shape has invalid dimensions {true_shape_val.shape}. Using default.")
                    return default_shape_batch
            return true_shape_val.to(view_img_tensor.device)

        shape1 = get_valid_true_shape(img1, 'true_shape', view1)
        shape2 = get_valid_true_shape(img2, 'true_shape', view2)

        if B > 0 and is_symmetrized(view1, view2):
            if B % 2 != 0:
                DBG("_encode_symmetrized Warning", "Odd batch size for symmetrized input, processing as non-symmetrized.")
                feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)
            else:
                feat1_half, feat2_half, pos1_half, pos2_half = self._encode_image_pairs(
                    img1[::2], img2[::2], shape1[::2], shape2[::2]
                )
                feat1, feat2 = interleave(feat1_half, feat2_half)
                pos1, pos2 = interleave(pos1_half, pos2_half)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        # Features (f1, f2) might be in half precision from encoder, convert to float for decoder.
        f1_fl = f1.float()
        f2_fl = f2.float()
        
        # **MODIFICATION**: Do NOT convert pos1, pos2 to float here.
        # Let them retain the dtype from patch_embed, assuming it's compatible with RoPE kernel (often Long or int).
        # pos1_fl, pos2_fl = pos1.float(), pos2.float() # REMOVED .float() for positions

        # Use original pos1, pos2 (or ensure they are on the correct device if not already)
        # It's good practice to ensure all inputs to a module part are on the same device.
        # f1_fl, f2_fl are on some device. pos1, pos2 should match.
        # Assuming pos1, pos2 are already on the correct device from _encode_image_pairs.

        final_output_list = [(f1_fl, f2_fl)] 

        f1_dec_emb = self.decoder_embed(f1_fl)
        f2_dec_emb = self.decoder_embed(f2_fl)
        final_output_list.append((f1_dec_emb, f2_dec_emb)) 

        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            f1_curr, f2_curr = final_output_list[-1]
            # Pass the original (non-float converted) pos1, pos2 to the decoder blocks
            f1_next, _ = blk1(f1_curr, f2_curr, pos1, pos2) 
            f2_next, _ = blk2(f2_curr, f1_curr, pos2, pos1)
            final_output_list.append((f1_next, f2_next))
        
        if len(final_output_list) > 1:
            del final_output_list[1] 

        if final_output_list: 
            f1_last_to_norm, f2_last_to_norm = final_output_list[-1]
            final_output_list[-1] = (self.dec_norm(f1_last_to_norm), self.dec_norm(f2_last_to_norm))
        else: 
            
            return [],[] 

        return list(zip(*final_output_list)) 

    def _downstream_head(self, head_num, decout_for_view, true_shape_for_view, img_rgb_for_view=None):

        head_module_wrapper = getattr(self, f'head{head_num}')
        if img_rgb_for_view is not None:
            return head_module_wrapper(decout_for_view, true_shape_for_view, img_rgb_for_view)
        else: 
            return head_module_wrapper(decout_for_view, true_shape_for_view)


    def forward(self, view1, view2):
        B = view1['img'].shape[0] 

        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # dec_outputs1_tuple and dec_outputs2_tuple are iterables from zip
        # Each element of the tuple is a list of features for that view across decoder stages
        # e.g., dec_outputs1_tuple = (enc_raw_f1_list, dec_l1_out_f1_list, ...)
        # No, zip(*final_output_list) makes it:
        # dec_outputs1_tuple = (f1_for_stage0, f1_for_stage1, ...)
        # dec_outputs2_tuple = (f2_for_stage0, f2_for_stage1, ...)
        # So dec_outputs1_tuple is the decout_for_view for view1.
        
        # _decoder returns: list(zip(*final_output_list))
        # if final_output_list = [(s0_f1, s0_f2), (s1_f1, s1_f2), (s2_f1, s2_f2)]
        # zip(...) = ((s0_f1,s1_f1,s2_f1), (s0_f2,s1_f2,s2_f2))
        # list(zip(...)) = [(s0_f1,s1_f1,s2_f1), (s0_f2,s1_f2,s2_f2)]
        # So dec_outputs_list[0] is the decout for view1, dec_outputs_list[1] is for view2.
        
        decoder_results = self._decoder(feat1, pos1, feat2, pos2)
        if not decoder_results or len(decoder_results) < 2:
            DBG("FWD Error", "Decoder did not return expected two output lists.")
            # Handle error appropriately, e.g., return None or raise exception
            return None, None 
            
        dec_outputs1_sequence, dec_outputs2_sequence = decoder_results[0], decoder_results[1]

        decout1_for_head = [tok.float() for tok in dec_outputs1_sequence] 
        decout2_for_head = [tok.float() for tok in dec_outputs2_sequence] 




        device = view1['img'].device 
        # ---------- before calling _downstream_head ----------
        img1_rgb = view1.get('original_img', view1['img']).to(device)
        img2_rgb = view2.get('original_img', view2['img']).to(device)

        res1 = self._downstream_head(1, decout1_for_head, shape1, img1_rgb)
        res2 = self._downstream_head(2, decout2_for_head, shape2, img2_rgb)


        res2['pts3d_in_other_view'] = res2.pop('pts3d')

            

        return res1, res2