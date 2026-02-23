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

from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .simple_multimodal_dataset import SimpleMultimodalDataset
from .masking import SimpleMultimodalMasking
from ..utils import infinite_iterator


def create_multimodal_masked_dataloader(
    root_dir: str,
    split: str,
    modalities: List[str],
    vocab_sizes: List[int],
    max_seq_lens: List[int],
    input_alphas: List[str],
    target_alphas: List[str],
    input_tokens_range: Union[int, Tuple[int, int]],
    target_tokens_range: Optional[Union[int, Tuple[int, int]]] = None,
    overlap_vocab: bool = True,
    overlap_posembs: bool = True,
    sample_from_k_augmentations: int = 10,
    text_tokenizer_path: str = 'gpt2',
    text_max_length: int = 256,
    batch_size: int = 64,
    infinite: bool = False,
    num_workers: int = 10,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
):
    """
    Creates a dataloader for a multimodal masked dataset.

    Args:
        root_dir: Root directory of the dataset.
        split: Split of the dataset (e.g. train, val, test).
        modalities: List of modalities.
        vocab_sizes: List of vocabulary sizes for each modality.
        max_seq_lens: List of maximum sequence lengths for each modality.
        input_alphas: Dirichlet alphas for the input modalities.
        target_alphas: Dirichlet alphas for the target modalities.
        input_tokens_range: Range of input tokens to use.
        target_tokens_range: Range of target tokens to use.
        overlap_vocab: Whether to use a unified vocabulary across modalities.
        overlap_posembs: Whether to reuse position indices/embeddings across modalities.
        sample_from_k_augmentations: Number of augmentations to sample from.
        text_tokenizer_path: Path or HuggingFace Hub ID of the text tokenizer, e.g. gpt2.
        text_max_length: Maximum length of the text.
        batch_size: Batch size for the dataloader.
        infinite: Whether to create an infinite dataloader.
        num_workers: Number of workers for the dataloader.
        pin_memory: Whether to pin memory for the dataloader.
        shuffle: Whether to shuffle the dataloader.
        drop_last: Whether to drop the last batch if it's smaller than the batch size.
        distributed: Whether to use a distributed sampler.
    """
    masking_transforms = SimpleMultimodalMasking(
        modalities=modalities,
        vocab_sizes=vocab_sizes,
        max_seq_lens=max_seq_lens,
        input_alphas=input_alphas,
        target_alphas=target_alphas,
        input_tokens_range=input_tokens_range,
        target_tokens_range=target_tokens_range,
        overlap_vocab=overlap_vocab,
        overlap_posembs=overlap_posembs,
    )

    dataset = SimpleMultimodalDataset(
        root_dir=root_dir,
        split=split,
        modalities=modalities,
        transforms=masking_transforms,
        sample_from_k_augmentations=sample_from_k_augmentations,
        text_tokenizer_path=text_tokenizer_path,
        text_max_length=text_max_length,
    )

    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if infinite:
        return infinite_iterator(dataloader, distributed, sampler)

    return dataloader
