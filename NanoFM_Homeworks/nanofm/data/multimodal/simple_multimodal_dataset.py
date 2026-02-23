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

from typing import List, Callable, Optional
import os
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleMultimodalDataset(Dataset):
    def __init__(
            self, 
            root_dir: str,
            split: str,
            modalities: List[str],
            transforms: Optional[Callable] = None,
            sample_from_k_augmentations: int = 10,
            text_tokenizer_path: str = 'gpt2',
            text_max_length: int = 256,
        ):
        """
        Simple multimodal dataset.
        
        Assumptions:
        - Each modality contains the same filenames in {root_dir}/{split}/{modality}/{file_name}.{ext}
        - Tokenized data is saved as .npy files and captions are saved as .json files
        - Each sample contains K random augmentations. The k-th augmentations are aligned across modalities.
        
        Args:
            root_dir: Root directory of the dataset.
            split: Split of the dataset (train, val, test).
            modalities: List of modalities.
            transforms: Transformations to apply to the data_dict before returning.
            sample_from_k_augmentations: Number of augmentations to sample from. If K=1, no augmentations are sampled,
                and only the "center crop" version is used. K is the total number of versions for each sample.
            text_tokenizer_path: Path or HuggingFace Hub ID of the text tokenizer, e.g. gpt2.
            text_max_length: Maximum length of the text.
        """
        self.root_dir = root_dir
        self.split = split
        self.modalities = modalities
        self.transforms = transforms
        self.sample_from_k_augmentations = sample_from_k_augmentations
        
        self.file_names = self._get_file_names()
        self.modality_extensions = self._get_modality_extensions()

        self.text_tokenizer_path = text_tokenizer_path
        self.text_max_length = text_max_length
        self.text_tokenizer = self._get_text_tokenizer()
        
    def __len__(self):
        return len(self.file_names)

    def _get_file_names(self):
        # Get all filenames. We assume the dataset is fully aligned across all modalities.
        file_names = [
            Path(file_path).stem
            for file_path in os.listdir(os.path.join(self.root_dir, self.split, self.modalities[0]))
        ]
        return sorted(file_names)
    
    def _get_modality_extensions(self):
        # Get file extension for all modalities
        extensions = {}
        for modality in self.modalities:
            modality_dir = Path(os.path.join(self.root_dir, self.split, modality))
            first_file = next(modality_dir.glob('*'))
            extensions[modality] = first_file.suffix
        return extensions

    def _get_text_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({
            'bos_token': '[SOS]',
            'eos_token': '[EOS]',
        })
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[('[EOS]', tokenizer.eos_token_id), ('[SOS]', tokenizer.bos_token_id)],
        )
        return tokenizer
        
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        data_dict = {}

        augmentation_idx = np.random.randint(0, self.sample_from_k_augmentations)
        
        for modality in self.modalities:
            ext = self.modality_extensions[modality]
            file_path = os.path.join(self.root_dir, self.split, modality, f"{file_name}{ext}")

            if 'tok' in modality:
                tokens = np.load(file_path)[augmentation_idx]
                tokens = torch.from_numpy(tokens).long()
            elif 'scene_desc' in modality:
                with open(file_path, 'r') as f:
                    captions = json.load(f)
                caption = captions[augmentation_idx]
                tokenized = self.text_tokenizer(
                    caption, max_length=self.text_max_length, padding='max_length', 
                    truncation=True, return_tensors='pt'
                )
                tokens = tokenized['input_ids'][0]
            else:
                raise ValueError(f"Unknown modality: {modality}")

            data_dict[modality] = tokens

        if self.transforms:
            data_dict = self.transforms(data_dict)
        
        return data_dict