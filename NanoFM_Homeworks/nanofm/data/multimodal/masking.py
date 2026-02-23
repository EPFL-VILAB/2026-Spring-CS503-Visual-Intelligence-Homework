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

from typing import List, Tuple, Dict, Any, Union, Optional
import random
from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from .utils import to_unified_multimodal_vocab


class SimpleMultimodalMasking(object):
    def __init__(
            self,
            modalities: List[str],
            vocab_sizes: List[int],
            max_seq_lens: List[int],
            input_alphas: List[str],
            target_alphas: List[str],
            input_tokens_range: Union[int, Tuple[int, int]],
            target_tokens_range: Union[int, Tuple[int, int]],
            overlap_vocab: bool = True,
            overlap_posembs: bool = True,
            include_unmasked_data_dict: bool = False,
        ):
        """
        Simple multimodal masking class for sampling input and target masks for each modality.
        Operates on a dictionary of modalities, where each entry is a dictionary with 
        a 'tokens' key containing the token tensor.

        Args:
            modalities: List of modality names
            vocab_sizes: Vocabulary size of each modality. Used to create a unified vocabolary.
            max_seq_lens: Maximum sequence length for each modality
            input_alphas: List of Dirichlet alphas for the input modalities
            target_alphas: List of Dirichlet alphas for the target modalities
            input_tokens_range: Range of number of input tokens to sample from
            target_tokens_range: Range of number of target tokens to sample from
            overlap_vocab: Whether to use a unified vocabulary across modalities.
            overlap_posembs: Whether to reuse position indices/embeddings across modalities.
            include_unmasked_data_dict: If True, adds the unmasked data dictionary to the output
                using the key 'unmasked_data_dict'.
        """
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.vocab_sizes = vocab_sizes
        self.max_seq_lens = max_seq_lens
        self.input_alphas = torch.tensor(input_alphas)
        self.target_alphas = torch.tensor(target_alphas)
        self.input_tokens_range = to_2tuple(input_tokens_range)
        self.target_tokens_range = to_2tuple(target_tokens_range)
        self.overlap_vocab = overlap_vocab
        self.overlap_posembs = overlap_posembs
        self.include_unmasked_data_dict = include_unmasked_data_dict

        self.max_seq_len_shifts = torch.tensor(max_seq_lens).cumsum(0) - max_seq_lens[0]

        # Dirichlet sampling
        eps = 1e-9
        self.input_dirichlet = Dirichlet(torch.clamp(self.input_alphas, min=eps))
        self.target_dirichlet = Dirichlet(torch.clamp(self.target_alphas, min=eps))
        
    def input_token_budget(self, num_input_tokens: int, max_tokens: torch.Tensor) -> List[int]:
        """Sample the number of input tokens for each modality, i.e. the
        per-modality token budget.

        Args:
            num_input_tokens: Number of tokens in the input
            max_tokens: Maximum number of tokens per modality

        Returns:
            Token budget for the input
        """
        # Get the number of tokens for each modality
        input_token_budget = (self.input_dirichlet.sample() * num_input_tokens).floor().int()
        diff = num_input_tokens - input_token_budget.sum()
        # Adds the remaining tokens by sampling from the Dirichlet and taking the argmax
        # This avoids adding tokens to modalities that shouldn't be sampled (i.e. with alphas ~=0)
        input_token_budget += torch.bincount(self.input_dirichlet.sample((diff,)).argmax(dim=-1), minlength=len(input_token_budget))

        # If token budget is over max tokens for a given modality, set it to max
        input_token_budget = torch.clamp(input_token_budget, max=max_tokens)

        return input_token_budget.tolist()

    def target_token_budget(
            self, 
            input_token_budget: List[int], 
            num_target_tokens: int,
            max_tokens: torch.Tensor,
        ) -> List[int]:
        """Sample the number of target tokens for each modality, i.e. the
        per-modality token budget.

        Args:
            input_token_budget: Token budget for the input modalities
            num_target_tokens: Number of tokens in the target
            max_tokens: Maximum number of tokens per modality

        Returns:
            Token budget for the target
        """
        max_tokens_remaining = max_tokens - torch.tensor(input_token_budget)

        target_token_budget = (self.target_dirichlet.sample() * num_target_tokens).floor().int()
        diff = num_target_tokens - target_token_budget.sum()
        # Adds the remaining tokens by sampling from the Dirichlet and taking the argmax
        # This avoids adding tokens to modalities that shouldn't be sampled (i.e. with alphas ~=0)
        target_token_budget += torch.bincount(self.target_dirichlet.sample((diff,)).argmax(dim=-1), minlength=len(target_token_budget))

        # If token budget is over max tokens for a given modality, set it to max
        target_token_budget = torch.clamp(target_token_budget, max=max_tokens_remaining)

        return target_token_budget.tolist()

    def perform_random_masking(
            self, 
            data_dict: Dict[str, Any],
            input_token_budget: List[int],
            target_token_budget: List[int],
        ) -> Dict[str, Any]:
        """
        Applies input and target masking to a dictionary of modalities.

        Args:
            data_dict: Dictionary of modalities and the corresponding tokens
            input_token_budget: Token budget for the input modalities
            target_token_budget: Token budget for the target modalities
        Returns:
            Dictionary containing the masked modality information
        """
        enc_tokens, enc_positions, enc_modalities = [], [], []
        dec_tokens, dec_positions, dec_modalities = [], [], []

        for mod_idx, mod in enumerate(self.modalities):
            num_tokens = data_dict[mod].shape[0]
            n_input_tokens = input_token_budget[mod_idx]
            n_target_tokens = target_token_budget[mod_idx]
            
            # Sample input and target positions
            noise = torch.rand(num_tokens)
            ids_shuffle = torch.argsort(noise, dim=0)
            input_pos = ids_shuffle[:n_input_tokens].sort()[0]
            target_pos = ids_shuffle[n_input_tokens:n_input_tokens+n_target_tokens].sort()[0]
            # Optionally shift the position indices such that each modality learns unique position embeddings
            pos_idx_shift = 0 if self.overlap_posembs else self.max_seq_len_shifts[mod_idx]
            enc_positions.append(input_pos + pos_idx_shift)
            dec_positions.append(target_pos + pos_idx_shift)

            # Get the corresponding input and target tokens
            input_tokens, target_tokens = data_dict[mod][input_pos], data_dict[mod][target_pos]
            enc_tokens.append(input_tokens)
            dec_tokens.append(target_tokens)

            # In case n_input_tokens+n_target_tokens was larger than num_tokens, let's recompute 
            # the actual number of input and target tokens
            n_input_tokens, n_target_tokens = input_pos.shape[0], target_pos.shape[0]
            
            # To decide which token to predict in the encoder and decoder, we pass modality indices 
            # that are transformed into a modality embedding
            enc_modalities.append(mod_idx * torch.ones(n_input_tokens, dtype=torch.long))
            dec_modalities.append(mod_idx * torch.ones(n_target_tokens, dtype=torch.long))
                        
        # Concatenate all lists into tensors
        enc_tokens, dec_tokens = torch.cat(enc_tokens), torch.cat(dec_tokens)
        enc_positions, dec_positions = torch.cat(enc_positions), torch.cat(dec_positions)
        enc_modalities, dec_modalities = torch.cat(enc_modalities), torch.cat(dec_modalities)

        # For batching, all sequences need the same length.
        max_input_tokens, max_target_tokens = self.input_tokens_range[1], self.target_tokens_range[1]
        enc_pad_length = max_input_tokens - enc_tokens.shape[0]
        dec_pad_length = max_target_tokens - dec_tokens.shape[0]
        enc_tokens = F.pad(enc_tokens, (0, enc_pad_length), mode='constant', value=0)
        enc_positions = F.pad(enc_positions, (0, enc_pad_length), mode='constant', value=0)
        enc_modalities = F.pad(enc_modalities, (0, enc_pad_length), mode='constant', value=0)
        dec_positions = F.pad(dec_positions, (0, dec_pad_length), mode='constant', value=0)
        dec_tokens = F.pad(dec_tokens, (0, dec_pad_length), mode='constant', value=-100)
        dec_modalities = F.pad(dec_modalities, (0, dec_pad_length), mode='constant', value=0)

        # Create attention masks for encoder and decoder
        enc_pad_mask = torch.ones(max_input_tokens, dtype=torch.bool)
        if enc_pad_length > 0:
            enc_pad_mask[-enc_pad_length:] = False
        dec_pad_mask = torch.ones(max_target_tokens, dtype=torch.bool)
        if dec_pad_length > 0:
            dec_pad_mask[-dec_pad_length:] = False

        masked_data_dict = {
            'enc_tokens': enc_tokens,
            'enc_positions': enc_positions,
            'enc_modalities': enc_modalities,
            'enc_pad_mask': enc_pad_mask,
            'dec_tokens': dec_tokens,
            'dec_positions': dec_positions,
            'dec_modalities': dec_modalities,
            'dec_pad_mask': dec_pad_mask,
        }

        return masked_data_dict

    def __call__(self, data_dict):
        """Applies input and target masking to a dictionary of modalities

        Args:
            data_dict: Dictionary of modalities

        Returns:
            Dictionary containing the masked modalities
        """
        if not self.overlap_vocab:
            # Unify the vocabulary for all modalities, making sure the indices for each modality 
            # are non-overlapping with other modalities.
            data_dict = to_unified_multimodal_vocab(data_dict, self.modalities, self.vocab_sizes)

        # Get maximum number of tokens for each modality
        max_tokens = torch.tensor(self.max_seq_lens)
        
        # Sample number of input and target tokens
        num_input_tokens = random.randint(*self.input_tokens_range)
        num_target_tokens = random.randint(*self.target_tokens_range)
        
        # Get input and target per-modality token budgets
        input_token_budget = self.input_token_budget(num_input_tokens, max_tokens)
        target_token_budget = self.target_token_budget(input_token_budget, num_target_tokens, max_tokens)
            
        # Apply input and target masking
        masked_data_dict = self.perform_random_masking(data_dict, input_token_budget, target_token_budget)

        if self.include_unmasked_data_dict:
            masked_data_dict['unmasked_data_dict'] = data_dict
            
        return masked_data_dict
