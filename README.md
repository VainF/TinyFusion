# TinyDiT

## Preparation

#### Extract ImageNet Features to enable fast training
```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path data/imagenet/train --features-path data/imagenet_encoded
```

All scripts end with `_fast` require the pre-extracted features.

#### Download Pre-trained DiT-XL/2

```bash
mkdir -p pretrained
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

## Layer Pruning

#### Pruning by Score
```bash
python prune_by_score.py --model DiT-XL/2 --ckpt pretrained/DiT-XL-2-256x256.pt --save-model outputs/pruned/DiT-D14-by-Score.pt --n-pruned 14
```

#### Pruning with Oracle Scheme
```bash
python prune_by_index.py --model DiT-XL/2 --ckpt pretrained/DiT-XL-2-256x256.pt --kept-indices "[0, 2, 4, 6, 8, 10, 12, 14, 16,
 18, 20, 22, 24, 26]" --save-model outputs/pruned/DiT-D14-Uniform.pt
```

#### Pruning by Indices
```bash
python prune_by_index.py --model DiT-XL/2 --ckpt pretrained/DiT-XL-2-256x256.pt --save-model outputs/pruned/DiT-D14-by-Score.pt --kept-indices "[0,2,4,6,8,10]"
```

#### Learnable Pruning (Ours)

TODO


## Training

```bash
# Finetune
torchrun --nnodes=1 --nproc_per_node=8 train_fast.py --model DiT-D14/2 --load-weight outputs/pruned/DiT-XL-D14-Learned.pt --data-path data/imagenet_encoded --epochs 100 --prefix D14-Learned-Finetuning 

# KD
torchrun --nnodes=1 --nproc_per_node=8 kd_fast.py --model DiT-D14/2 --load-weight outputs/pruned/DiT-XL-D14-Learned.pt --data-path data/imagenet_encoded --epochs 100 --prefix D14-Learned-KD --teacher DiT-XL/2 --load-teacher pretrained/DiT-XL-2-256x256.pt

# RepKD
torchrun --nnodes=1 --nproc_per_node=8 kd_rep_fast.py --model DiT-D14/2 --load-weight outputs/pruned/DiT-XL-D14-Learned.pt --data-path data/imagenet_encoded --epochs 100 --prefix D14-Learned-RepKD --teacher DiT-XL/2 --load-teacher pretrained/DiT-XL-2-256x256.pt
```
