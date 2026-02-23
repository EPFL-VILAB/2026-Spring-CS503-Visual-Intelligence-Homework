# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from einops import rearrange
import math

from ..utils import infinite_iterator


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1).long()

def tokenize_MNIST(imgs, patch_size=2, shift_vocab_for_labels=False):
    """
    Tokenizes a batch of MNIST images into discrete tokens.
    Each image is first thresholded (binarized) and then split into non-overlapping patches.

    Parameters:
        imgs (Tensor): Batch of MNIST images with shape [B, 1, H, W].
        patch_size (int): Size of each patch (both height and width).

    Returns:
        Tensor: Tokenized representation with shape [B, num_patches], where each token is an integer.
    """
    # Binarize the image using a threshold of 0.5
    imgs = (imgs[:, 0] > 0.5).int()
    bits = rearrange(
        imgs, 
        'b (nh ph) (nw pw) -> b (nh nw) (ph pw)',
        ph=patch_size, pw=patch_size
    )
    tokens = bin2dec(bits, patch_size ** 2)
    if shift_vocab_for_labels:
        tokens += 10
    return tokens

def detokenize_MNIST(imgs_tokenized, patch_size=2, account_for_labels=False):
    """
    Reconstructs MNIST images from tokenized representations.

    Parameters:
        imgs_tokenized (Tensor): Tokenized MNIST images with shape [B, num_patches].
        patch_size (int): The patch size used during tokenization.

    Returns:
        Tensor: Reconstructed images with shape [B, H, W].
    """
    imgs_tokenized = imgs_tokenized.clone()
    if account_for_labels:
        imgs_tokenized = imgs_tokenized[:, 1:]
        imgs_tokenized -= 10
    bits = dec2bin(imgs_tokenized, patch_size ** 2)
    N = int(math.sqrt(imgs_tokenized.shape[-1]))
    return rearrange(
        bits, 
        'b (nh nw) (ph pw) -> b (nh ph) (nw pw)',
        nh=N, nw=N, ph=patch_size, pw=patch_size
    )

def create_tokenized_mnist_dataloader(
    train: bool = True,
    image_size: int = 14,
    patch_size: int = 2,
    add_sos_token: bool = False,
    add_label_token: bool = False,
    batch_size: int = 64,
    infinite: bool = False,
    num_workers: int = 10,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
):
    """
    Creates a dataloader for the MNIST dataset that tokenizes images in a very simple patch-wise manner.

    Parameters:
        train (bool): Whether to use the training split (True) or test split (False).
        image_size (int): The size to which MNIST images are resized (image_size x image_size).
        patch_size (int): The patch size for tokenization.
        add_sos_token (bool): Whether to add a start-of-sequence token to the beginning of each sequence.
        add_label_token (bool): Whether to add a label token to the beginning of each sequence.
        batch_size (int): Batch size.
        infinite (bool): If True, returns an infinite iterator that reshuffles data every epoch.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        distributed (bool): Whether to use a distributed sampler.

    Returns:
        DataLoader or generator: A PyTorch DataLoader (or infinite iterator) that yields batches of tokenized MNIST images.
    """
    if train:
        # A tiny amount of data augmentation goes a long way
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    dataset = MNIST(root='/tmp/mnist_data', train=train, download=True, transform=transform)

    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None

    def collate_fn(batch):
        # Each batch element is a tuple (image, label); we only need the image.
        images = torch.stack([item[0] for item in batch], dim=0)
        tokens = tokenize_MNIST(images, patch_size=patch_size, shift_vocab_for_labels=add_label_token)
        labels = torch.tensor([item[1] for item in batch])
        if add_sos_token:
            # Add a start-of-sequence token to the beginning of each sequence, for unconditional generation.
            vocab_size = 2**(patch_size*patch_size)
            start_of_seq_token = vocab_size
            tokens = torch.nn.functional.pad(tokens, (1,0,0,0), value=start_of_seq_token)
        elif add_label_token:
            # Concatenate the label to the beginning of the sequence.
            tokens = torch.cat([labels.unsqueeze(-1), tokens], dim=-1)
        return {'input_ids': tokens, 'labels': labels}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    if infinite:
        return infinite_iterator(dataloader, distributed, sampler)

    return dataloader