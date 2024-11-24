import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models, DiTBlock
import argparse
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Union

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--data-path", type=str, default="data/imagenet/train")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--n-pruned", type=int, required=True)
    args = parser.parse_args()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model.to(device)

    # Layer Expansion
    @torch.no_grad()
    def compute_BI_score(module, inputs, outputs):
        # N, L, D 
        out_state = outputs
        in_state = inputs[0]
        cosine_sim = torch.nn.functional.cosine_similarity(out_state, in_state, dim=-1).mean().cpu()
        if hasattr(module, 'BI_score'):
            module.BI_score.append(cosine_sim)
        else:
            module.BI_score = [cosine_sim]

    hooks = []
    for layer in model.blocks:
        hooks.append(layer.register_forward_hook(compute_BI_score))
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    nbatches = args.nsamples // args.batch_size
    for i, (x, y) in enumerate(loader):
        if i == nbatches: break
        print(f"Batch {i+1}/{nbatches}")
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            inputs_dict = {
                "x": x,
                "t": t,
                "y": y,
            }
            _ = model(**inputs_dict)

    for hook in hooks:
        hook.remove()

    layer_scores = []
    for layer in model.blocks:
        layer_scores.append( sum(layer.BI_score) / len(layer.BI_score) )
    layer_scores = torch.tensor(layer_scores)

    n_pruned = args.n_pruned
    topk = torch.topk(torch.tensor(layer_scores)[:-1], n_pruned, largest=True).indices # topk layers will be removed
    print(layer_scores, topk)
    new_blocks = []
    kept_indices = []
    for i in range(len(model.blocks)):
        if i not in topk:
            new_blocks.append(model.blocks[i])
            kept_indices.append(i)
    model.blocks = torch.nn.ModuleList(new_blocks)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of layers: {len(model.blocks)}")
    print(f"Remaining layers: {kept_indices}")
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    # Check sparsiy
    if args.save_model is not None:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)

    