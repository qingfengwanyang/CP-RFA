# Pretrained models

Place third-party weights as follows. The mapper checkpoint is already in this folder.

## Mapper / HSIO (included)

```text
checkpoints/mapper_best.pth.tar
```

Contains the trained password-conditioned mapper and the frozen HSIO `seed_proj` (`fpe_init_seed = 0`).

## HyperSwap generator (download)

```text
checkpoints/hyperswap/hyperswap_1a_256.onnx
```

```bash
mkdir -p checkpoints/hyperswap
wget -O checkpoints/hyperswap/hyperswap_1a_256.onnx \
  https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx
```

HyperSwap is a third-party model. Follow its license on Hugging Face.

## InsightFace buffalo_l (auto-download)

On first run, InsightFace writes detection and ArcFace weights to:

```text
checkpoints/insightface/models/buffalo_l/
```

You can also copy an existing `buffalo_l` folder there. Requires `insightface>=0.7.3`.
