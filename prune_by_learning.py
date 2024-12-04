# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
import os, random

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
#import wandb

import torchtune
from torchtune.modules.peft import get_adapter_params 
import torch.nn as nn 

def convert_to_lora_model(model, rank, alpha=1.0, exclude=[]):
    """ replace all linear layers with lora layers """
    for name, module in model.named_children():
        if module in exclude:
            continue
        if isinstance(module, nn.Linear):
            lora_linear = torchtune.modules.peft.LoRALinear(in_dim=module.in_features, out_dim=module.out_features, rank=rank, alpha=alpha, use_bias=module.bias is not None)
            lora_linear.weight.data = module.weight.data
            if module.bias is not None:
                lora_linear.bias.data = module.bias.data
            lora_linear.to(module.weight.device)
            setattr(model, name, lora_linear)   
        else:
            convert_to_lora_model(module, rank, alpha, exclude)     
    return model

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, labels_dir, flip=0):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.flip = flip

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))

        if self.flip>0:
            if random.random() < self.flip:
                features = features[1:]
            else:
                features = features[:1]
        return torch.from_numpy(features), torch.from_numpy(labels)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    #if rank == 0:
    #   wandb.init(project='RescaleDiT', config=args, name=args.prefix)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{args.prefix}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        print(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    if args.load_weight is not None:
        initial_ckpt = torch.load(args.load_weight, map_location='cpu')
        if 'ema' in initial_ckpt:
            model.load_state_dict(initial_ckpt['ema'], strict=False)
            print(f"Loaded initial EMA weights from {args.load_weight}")
        elif 'model' in initial_ckpt:
            model.load_state_dict(initial_ckpt['model'], strict=False)
            print(f"Loaded initial weights from {args.load_weight}")
        else:
            model.load_state_dict(initial_ckpt, strict=False)
            print(f"Loaded plain weights from {args.load_weight}")

    features_dir = f"{args.data_path}/imagenet256_features"
    labels_dir = f"{args.data_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir, flip=0.5)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if rank==0:
        print(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    lr = 1e-4
    gate_params = {}
    trainable_params = {}

    for name, p in model.named_parameters():
        if 'gumbel_gates' in name:
            gate_params[name] = p

    if args.delta_w:
        if args.lora:
            model = convert_to_lora_model(model, rank=args.lora_rank, alpha=args.lora_rank*2, exclude=[])
            trainable_params = get_adapter_params(model)
        else:
            for name, p in model.named_parameters():
                if 'gumbel_gates' not in name:
                    trainable_params[name] = p 
    
    for name, p in model.named_parameters():
        if name not in trainable_params and name not in gate_params:
            p.requires_grad = False
    
    if rank==0:
        print("Trainable Parameters: \n", list(trainable_params.keys())) 
        print("Trainable Coefficients: \n", list(gate_params.keys()))
    opt = torch.optim.AdamW([
        {"params": list(trainable_params.values()), "lr": lr}, 
        {"params": list(gate_params.values()), "lr": 10*lr}], 
    lr=lr, weight_decay=0.0)
    #sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=lr*0.5)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if rank==0:
        print(model)
        print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Fetch all params from the model that are associated with LoRA.
    #sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=lr*0.5)
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{device}')
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        #sched.load_state_dict(checkpoint["sched"])
        print(f"Resume training from {args.resume}")
        train_steps = checkpoint["train_steps"]
        start_epoch = train_steps // len(loader) 
        del checkpoint
    else:
        start_epoch = 0
        train_steps = 0
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    scaler = torch.cuda.amp.GradScaler()
    if rank==0:
        print(f"Training for {args.epochs} epochs ({args.epochs * len(loader)} steps)...")
    
        print(f"Init Probs and Selection:")
        layer_offset = 0
        decisions = []
        conf = []
        for gate, option in zip(ema.gumbel_gates, ema.options):
            selected = gate.max(1)[1].item()
            mask = option[selected]
            selected_layers = (mask.nonzero()+layer_offset).squeeze().tolist()
            if isinstance(selected_layers, int):
                selected_layers = [selected_layers]
            #print(torch.softmax(gate*1e1, dim=1).detach().cpu().tolist(), selected_layers)
            conf.append( max( torch.softmax(gate*model.module.scaling, dim=1).detach().cpu().tolist()[0] ) )
            decisions.extend(selected_layers)
            layer_offset += option.size(1)
        print(f"Decision: {decisions}")
        print(f"Confidence: {conf}")


    total_steps = args.epochs * len(loader)
    model_params = [p for name, p in model.named_parameters() if 'gumbel_gates' not in name]
    gate_params = [p for name, p in model.named_parameters() if 'gumbel_gates' in name]
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        if rank==0:
            print(f"Beginning epoch {epoch}...")

        for x, y in loader:

            tau = args.tau_range[0] + (args.tau_range[1] - args.tau_range[0]) * train_steps / max(total_steps - 1, 1)
            scaling = args.scaling_range[0] + (args.scaling_range[1] - args.scaling_range[0]) * train_steps / max(total_steps - 1, 1)

            model.module.tau = tau
            model.module.scaling = scaling

            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            opt.zero_grad()
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
            loss.backward()
            if (loss.isnan().sum() == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            
            update_ema(ema, model.module, decay=args.ema_decay)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                if rank==0:
                    print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {opt.param_groups[0]['lr']:.6f}")
                    print(f"Probs and Selection:")
                    layer_offset = 0
                    decisions = []
                    conf = []
                    for gate, option in zip(ema.gumbel_gates, ema.options):
                        selected = gate.max(1)[1].item()
                        mask = option[selected]
                        selected_layers = (mask.nonzero()+layer_offset).squeeze().tolist()
                        if isinstance(selected_layers, int):
                            selected_layers = [selected_layers]
                        #print(torch.softmax(gate*1e1, dim=1).detach().cpu().tolist(), selected_layers)
                        conf.append( max( torch.softmax(gate*model.module.scaling, dim=1).detach().cpu().tolist()[0] ) )
                        decisions.extend(selected_layers)
                        layer_offset += option.size(1)
                    print(f"Decision: {decisions}")
                    print(f"Confidence: {conf}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                #if rank == 0:
                #    try:
                #        wandb.log({"train_loss": avg_loss, "misc/steps_per_sec": steps_per_sec, "misc/lr": opt.param_groups[0]['lr']}, step=train_steps)
                #    except:
                #        pass

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        #"sched": sched.state_dict(),
                        "train_steps": train_steps,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    # Final Decision
    if rank==0 and args.save_model is not None:
        layer_offset = 0
        decisions = []
        conf = []
        for gate, option in zip(ema.gumbel_gates, ema.options):
            selected = gate.max(1)[1].item()
            mask = option[selected]
            selected_layers = (mask.nonzero()+layer_offset).squeeze().tolist()
            if isinstance(selected_layers, int):
                selected_layers = [selected_layers]
            #print(torch.softmax(gate*1e1, dim=1).detach().cpu().tolist(), selected_layers)
            conf.append( max( torch.softmax(gate*model.module.scaling, dim=1).detach().cpu().tolist()[0] ) )
            decisions.extend(selected_layers)
            layer_offset += option.size(1)
        print(f"[!] Final Decision: {decisions}")
        print(f"[!] Final Confidence: {conf}")

        # Reload original checkpoint
        model_without_ddp = model.module

        initial_ckpt = torch.load(args.load_weight, map_location='cpu')
        if 'ema' in initial_ckpt:
            model_without_ddp.load_state_dict(initial_ckpt['ema'], strict=False)
            print(f"Loaded initial EMA weights from {args.load_weight}")
        elif 'model' in initial_ckpt:
            model_without_ddp.load_state_dict(initial_ckpt['model'], strict=False)
            print(f"Loaded initial weights from {args.load_weight}")
        else:
            model_without_ddp.load_state_dict(initial_ckpt, strict=False)
            print(f"Loaded plain weights from {args.load_weight}")

        # Save pruned checkpoint: 
        new_blocks = []
        for i in range(len(model_without_ddp.blocks)):
            if i in decisions:
                new_blocks.append(model_without_ddp.blocks[i])
        model_without_ddp.blocks = torch.nn.ModuleList(new_blocks)
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        state_dict = model_without_ddp.state_dict()
        clean_state_dict = {}
        for k, v in state_dict.items():
            if 'gumbel_gates' in k or 'lora' in k:
                continue
            clean_state_dict[k] = v
        torch.save(clean_state_dict, args.save_model)
        print(f"Saved pruned model to {checkpoint_dir}/pruned.pt")
        print(f"Remaining parameters: {sum(p.numel() for p in model_without_ddp.parameters())}")
        print(f"Number of Remaining layers: {len(model_without_ddp.blocks)}")
        print(f"Remaining Layers: {decisions}")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/imagenet/train")
    parser.add_argument("--results-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--load-weight", type=str, default=None)
    parser.add_argument("--prefix", type=str, default='learn_mask')
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora", action='store_true', default=False)
    parser.add_argument("--delta-w", action='store_true', default=False, help="enable efficient weight update during structure learning")
    parser.add_argument('--scaling-range', nargs='+', type=float, default=[1e2, 1e2]) # this controls the one-hotness of the learnable probability, see https://arxiv.org/abs/2409.17481
    parser.add_argument('--tau-range', nargs='+', type=float, default=[4, 0.1]) # this controls the temperature of the gumbel softmax.
    parser.add_argument("--save-model", type=str, default=None)
    args = parser.parse_args()
    main(args)
