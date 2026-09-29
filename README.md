# MAC-Splat

[Paper](https://arxiv.org/abs/2607.10792)

Install PyTorch and torchvision for your CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Compile the optional CroCo RoPE extension:

```bash
cd src/mast3r_src/dust3r/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../../..
```

Use a Lightning checkpoint containing an `encoder.`-prefixed `state_dict`. Set `DINOV3_LOCAL_DIR` to the local DINO model directory and prepare ScanNet++ using the [dataset layout](data/scannetpp/scannetpp.py).

```bash
python main.py configs/main.yaml \
  pretrained_mast3r_path=/path/to/model.ckpt \
  data.root=/path/to/scannetpp
```

Based on [Splatt3R](https://github.com/btsmart/splatt3r), [MASt3R](https://github.com/naver/mast3r), [DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco), [pixelSplat](https://github.com/dcharatan/pixelsplat), and [PyTorch3D](https://github.com/facebookresearch/pytorch3d).

[License](License) · [Third-party notices](licenses/SOURCE-NOTICES.txt) · [License texts](licenses/)
