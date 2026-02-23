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

from typing import Optional
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

from ..utils import infinite_iterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_hf_dataloader(
    dataset_id: str,
    split: str,
    tokenizer_path: str,
    add_padding_tokens: bool = False,
    add_sos_eos_tokens: bool = False,
    replace_newline: Optional[str] = None,
    batch_size: int = 32,
    max_seq_len: int = 256,
    increase_seq_len_by_one: bool = True,
    infinite: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
):
    """
    Creates a dataloader from a Hugging Face Hub dataset.

    Parameters:
        dataset_id (str): The Hugging Face Hub dataset id.
        split (str): Which split to use (e.g. "train", "validation").
        tokenizer_path (str): The Hugging Face Hub path to the tokenizer.
        add_padding_tokens (bool): Whether to add special padding tokens to the tokenizer.
        add_sos_eos_tokens (bool): Whether to add special SOS and EOS tokens to the tokenizer.
            For example, usful for adding these tokens to GPT-2 tokenizer.
        replace_newline (str): If not None, replace newline characters with this string.
        batch_size (int): Batch size.
        max_seq_len (int): Maximum sequence length.
        increase_seq_len_by_one (bool): Whether to increase the sequence length by one. Set to
            True when performing autoregressive modeling, where the target sequence is shifted by one.
        infinite (bool): If True, returns an infinite iterator that reshuffles data every epoch.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        distributed (bool): Whether to use a distributed sampler.
        
    Returns:
        DataLoader or generator: A PyTorch DataLoader (or infinite iterator) that yields batches.
    """
    dataset = load_dataset(dataset_id, split=split)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if add_padding_tokens:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if add_sos_eos_tokens:
        tokenizer.add_special_tokens({
            'bos_token': '[SOS]',
            'eos_token': '[EOS]',
        })
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[('[EOS]', tokenizer.eos_token_id), ('[SOS]', tokenizer.bos_token_id)],
        )
    
    loader_seq_len = max_seq_len + 1 if increase_seq_len_by_one else max_seq_len
    def collate_fn(batch):
        texts = [sample['text'].replace('\n', replace_newline) if replace_newline else sample['text'] for sample in batch]
        tokenized = tokenizer(texts, max_length=loader_seq_len, padding='max_length', truncation=True, return_tensors='pt')
        return tokenized
    
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    
    dataloader = DataLoader(
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
