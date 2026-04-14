import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import importlib
from .datasets import FingerprintDataset, DiscriminatorDataset

def create_dataloader(paths, opt):
    
    if opt.dataset_mode.startswith("normal"):
        dataset = FingerprintDataset(paths, opt)
    elif opt.dataset_mode.startswith("discriminator"):
        dataset = DiscriminatorDataset(paths, opt)
    elif opt.dataset_mode.startswith("triple"):
        dataset = DiscriminatorDataset(paths, opt)
    
    else:
        raise ValueError("Invalid dataset mode:", opt.dataset_mode)
   
    
    shuffle = True
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads),
        pin_memory=True
    )
    
    return data_loader
