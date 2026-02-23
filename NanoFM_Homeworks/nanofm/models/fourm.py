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

from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from nanofm.modeling.transformer_layers import TransformerTrunk, TransformerDecoderTrunk, LayerNorm
from nanofm.utils.sampling import sample_tokens


def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d.
    Returns positional embedding of shape (N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32) # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega]) # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1) # Shape (N, D)
    return pos_emb


class FourM(nn.Module):
    """Simplified 4M definition, in which all modalities are handled in a single unified vocabulary.

    Args:
        enc_tokens_read_key: Key for reading encoder input tokens from the data dictionary
        dec_tokens_read_key: Key for reading decoder target tokens from the data dictionary.
            Only used for loss computation.
        enc_modalities_read_key: Key for reading encoder input modality ids from the data dictionary
        dec_modalities_read_key: Key for reading decoder target modality ids from the data dictionary
        enc_positions_read_key: Key for reading encoder input positions from the data dictionary
        dec_positions_read_key: Key for reading decoder target positions from the data dictionary
        enc_pad_mask_read_key: Key for reading encoder input padding mask from the data dictionary
        dec_pad_mask_read_key: Key for reading decoder target padding mask from the data dictionary
        modalities: List of modality names
        vocab_sizes: List of vocabulary sizes for each modality
        max_seq_lens: List of maximum sequence lengths for each modality
        dim: Transformer dimension
        enc_depth: Number of Transformer encoder layers
        dec_depth: Number of Transformer decoder layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        padding_idx: Padding index for the target sequences
        init_std: Standard deviation for weight initialization
        per_modality_loss_avg: If True, compute the loss for each modality separately and average them.
            Otherwise, compute the loss over all target tokens together.
    """
    def __init__(
        self,
        enc_tokens_read_key: str,
        dec_tokens_read_key: str,
        enc_modalities_read_key: str,
        dec_modalities_read_key: str,
        enc_positions_read_key: str,
        dec_positions_read_key: str,
        enc_pad_mask_read_key: str,
        dec_pad_mask_read_key: str,
        modalities: List[str],
        vocab_sizes: List[int],
        max_seq_lens: List[int],
        dim: int = 512,
        enc_depth: int = 8,
        dec_depth: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        use_bias: bool = False,
        padding_idx: int = -100,
        init_std: float = 0.02,
        per_modality_loss_avg: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.enc_tokens_read_key = enc_tokens_read_key
        self.dec_tokens_read_key = dec_tokens_read_key
        self.enc_modalities_read_key = enc_modalities_read_key
        self.dec_modalities_read_key = dec_modalities_read_key
        self.enc_positions_read_key = enc_positions_read_key
        self.dec_positions_read_key = dec_positions_read_key
        self.enc_pad_mask_read_key = enc_pad_mask_read_key
        self.dec_pad_mask_read_key = dec_pad_mask_read_key

        self.modalities = modalities
        self.vocab_sizes = vocab_sizes
        self.vocab_size = max(vocab_sizes)
        self.max_seq_lens = max_seq_lens
        self.max_posemb_len = max(max_seq_lens)
        self.num_modalities = len(modalities)
        self.padding_idx = padding_idx
        self.per_modality_loss_avg = per_modality_loss_avg

        # Initialize encoder token embedding
        self.enc_tok_emb = ??? # TODO: Define the input embedding layer using self.vocab_size

        # Initialize positional embeddings of predefined maximum length that we re-use for different modalities
        pos_emb = build_1d_sincos_posemb(self.max_posemb_len, dim)
        self.register_buffer("pos_emb", pos_emb)

        # Initialize modality embeddings
        self.enc_mod_emb = ??? # TODO: Define the encoder modality embedding layer using self.num_modalities
        self.dec_mod_emb = ??? # TODO: Define the decoder modality embedding layer using self.num_modalities
                
        # Initialize Transformer encoder and decoder trunks
        self.encoder = ??? # TODO: Define the Transformer encoder trunk
        self.decoder = ??? # TODO: Define the Transformer decoder trunk

        # Initialize encoder -> decoder context projection
        self.dec_context_proj = ??? # TODO: Define the encoder -> decoder projection. This is simply a linear layer mapping dim -> dim.

        # Initialize decoder output projection
        self.to_logits = ??? # TODO: Define the output projection layer using self.vocab_size

        # Initialize norm layers
        self.enc_norm = ??? # TODO: Define the encoder output layer normalization. Use the LayerNorm class defined in modeling/transformer_layers.py
        self.dec_norm = ??? # TODO: Define the decoder output layer normalization. Use the LayerNorm class defined in modeling/transformer_layers.py

        # Weight initialization
        self.init_std = init_std
        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        """Initialize the weights of the model.""" 
        self.apply(self._init_weights) # Initialize nn.Linear and nn.Embedding
        nn.init.constant_(self.to_logits.weight, 0) # Zero-init the output projection

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: For non-embedding count (default), the input and output embeddings get subtracted.
        Returns:
            The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.enc_tok_emb.weight.numel()
            n_params -= self.enc_mod_emb.weight.numel()
            n_params -= self.dec_mod_emb.weight.numel()
            n_params -= self.to_logits.weight.numel()
        return n_params

    def forward_encoder(
        self,
        enc_input_tokens: torch.LongTensor,
        enc_input_modalities: torch.LongTensor,
        enc_input_positions: torch.LongTensor,
        enc_pad_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            enc_input_tokens: LongTensor of shape (B, N) with encoder input token IDs.
            enc_input_modalities: LongTensor of shape (B, N) with IDs specifying which modality
                is input at each position.
            enc_input_positions: LongTensor of shape (B, N) with encoder input positions, 
                used to get the corresponding positional embeddings.
            enc_pad_mask: Boolean tensor of shape (B, N) where True indicates a valid token,
                and False indicates a padded token.
        Returns:
            Encoded tokens tensor of shape (B, N, D) and corresponding positional embeddings 
            tensor of shape (B, N, D).
        """
        B, N = enc_input_tokens.shape

        # TODO: Embed the input tokens `enc_input_tokens` using the input embedding layer `enc_tok_emb`. Shape: [B, N, D]
        x = ???

        # TODO: Embed the input modality IDs `enc_input_modalities` using the input embedding layer `enc_mod_emb`. Shape: [B, N, D]
        # Sum the modality embeddings to the token embeddings.
        x = x + ???

        # TODO: Get the positional embeddings for the input positions `enc_input_positions` and add them to the input tokens. Shape: [B, N, D]
        # Sum the positional embeddings to the token embeddings.
        enc_posembs = ???
        x = x + ???

        # Construct (B, N, N) attention mask for padding. True = used, False = masked out.
        enc_pad_attn_mask = repeat(enc_pad_mask, 'b n -> b m n', m=N) if enc_pad_mask is not None else None

        # TODO: Forward pass through the Transformer encoder. Shape [B, N, D]
        # Hint: Don't forget to pass the encoder attention mask `enc_pad_attn_mask`.
        x = ???

        # TODO: Pass to the encoder output normalization layer
        x = ???

        return x, enc_posembs

    def forward_decoder(
        self,
        dec_input_modalities: torch.LongTensor,
        dec_input_positions: torch.LongTensor,
        enc_context: torch.Tensor,
        enc_posembs: torch.Tensor,
        enc_pad_mask: Optional[torch.BoolTensor] = None,
        dec_pad_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            dec_input_modalities: LongTensor of shape (B, M) with IDs specifying which modality
                to predict at each position.
            dec_input_positions: LongTensor of shape (B, N) with decoder input positions,
                used to get the corresponding positional embeddings and specify which token
                to predict for each modality.
            enc_context: Tensor of shape (B, N, D) with encoder context tokens.
            enc_posembs: Tensor of shape (B, N, D) with encoder positional embeddings.
            enc_pad_mask: Boolean tensor of shape (B, N) where True indicates a valid token,
                and False indicates a padded token.
            dec_pad_mask: Boolean tensor of shape (B, M) where True indicates a valid token,
                and False indicates a padded token.
        Returns:
            Decoded tokens tensor of shape (B, M, D).
        """
        B, M = dec_input_modalities.shape
        _, N, _ = enc_context.shape

        # TODO: Embed the target modality IDs `dec_input_modalities` using the embedding layer `dec_mod_emb`. Shape: [B, M, D]
        x = ???

        # TODO: Get the positional embeddings for the target positions `dec_input_positions` and add them to the tokens. Shape: [B, M, D]
        # Sum the positional embeddings to the token embeddings.
        x = x + ???

        # Construct attention masks for padding. True = used, False = masked out.
        # [B, M, M] self-attention mask and [B, M, N] cross-attention mask
        dec_pad_sa_mask = repeat(dec_pad_mask, 'b m -> b n m', n=M) if dec_pad_mask is not None else None
        dec_pad_xa_mask = repeat(enc_pad_mask, 'b n -> b m n', m=M) if enc_pad_mask is not None else None

        # TODO: Project context `enc_context` to the decoder dimension using `dec_context_proj`. Shape: [B, N, D] 
        context = ???

        # Add the encoder positional embeddings `enc_posembs`. Shape: [B, N, D]
        context = context + enc_posembs

        # TODO: Pass through the Transformer decoder. Shape [B, M, D]
        # Hint: Don't forget to pass the decoder self-attention mask `dec_pad_sa_mask` and the cross-attention mask `dec_pad_xa_mask`.
        x = ???

        # TODO: Pass to the decoder output normalization layer
        x = ???

        return x

    def forward_model(
            self, 
            enc_input_tokens: torch.LongTensor,
            enc_input_modalities: torch.LongTensor,
            enc_input_positions: torch.LongTensor,
            dec_input_modalities: torch.LongTensor,
            dec_input_positions: torch.LongTensor,
            enc_pad_mask: Optional[torch.BoolTensor] = None,
            dec_pad_mask: Optional[torch.BoolTensor] = None,
        ) -> torch.Tensor:

        # Encoder forward pass
        enc_x, enc_posembs = self.forward_encoder(enc_input_tokens, enc_input_modalities, enc_input_positions, enc_pad_mask)

        # Decoder forward pass
        dec_x = self.forward_decoder(dec_input_modalities, dec_input_positions, enc_x, enc_posembs, enc_pad_mask, dec_pad_mask)

        # TODO: Pass `dec_x` through linear output head `to_logits` to compute the logits. Shape: [B, M, vocab_size]
        logits = ???

        return logits

    def compute_ce_loss(
            self, 
            logits: torch.Tensor, 
            target_seq: torch.LongTensor, 
            padding_idx: int = -100,
            per_modality_loss_avg: bool = False,
            modality_indices: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the cross-entropy loss given logits and target labels, ignoring padding tokens.

        Args:
             logits: Tensor of shape (B, L, vocab_size)
             target_seq: Tensor of shape (B, L) containing the target token indices.
             padding_idx: The index of the [PAD] token that should be ignored in the loss computation.
             per_modality_loss_avg: If True, compute the loss for each modality separately and average them.
             modality_indices: Tensor of shape (B, L) with IDs specifying which modality
        Returns:
             A scalar loss value and a dictionary of per-modality losses.
        """
        B, L, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        target_seq = target_seq.reshape(-1)

        if not per_modality_loss_avg:
            loss = F.cross_entropy(logits, target_seq, ignore_index=padding_idx)
            per_modality_losses = {}
        else:
            modality_indices = modality_indices.reshape(-1)
            valid_mask = target_seq != padding_idx

            losses = F.cross_entropy(logits, target_seq, reduction='none')

            per_modality_losses = {}
            for mod_idx, modality in enumerate(self.modalities):
                mod_mask = (modality_indices == mod_idx) & valid_mask
                if mod_mask.any():
                    per_modality_losses[modality] = losses[mod_mask].mean()
                else:
                    per_modality_losses[modality] = torch.tensor(0.0, device=losses.device)

            loss = sum(per_modality_losses.values()) / len(per_modality_losses)

        return loss, per_modality_losses


    def forward(self, data_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the model.

        Args:
            data_dict: A dictionary containing the input and target sequences.
        Returns:
            The loss and a dictionary containing the perplexity metric.
        """
        # Encoder, decoder, and output head forward pass
        logits = self.forward_model(
            enc_input_tokens=data_dict[self.enc_tokens_read_key],
            enc_input_modalities=data_dict[self.enc_modalities_read_key],
            enc_input_positions=data_dict[self.enc_positions_read_key],
            dec_input_modalities=data_dict[self.dec_modalities_read_key],
            dec_input_positions=data_dict[self.dec_positions_read_key],
            enc_pad_mask=data_dict.get(self.enc_pad_mask_read_key, None),
            dec_pad_mask=data_dict.get(self.dec_pad_mask_read_key, None),
        )
        
        # Compute loss
        loss, modality_metrics =  self.compute_ce_loss(
            logits=logits,
            target_seq=data_dict[self.dec_tokens_read_key],
            padding_idx=self.padding_idx,
            per_modality_loss_avg=self.per_modality_loss_avg,
            modality_indices=data_dict[self.dec_modalities_read_key],
        )

        return loss, modality_metrics

    def get_unmasking_schedule(self, total_tokens: int, num_steps: int = 8) -> List[int]:
        """
        Generates a schedule for unmasking tokens at inference time. We only added a 
        constant schedule for now, but feel free to add more schedules, e.g. a cosine schedule!
        This can be used for both "ROAR" and MaskGIT-style decoding.

        Args:
            total_tokens: Number of tokens to predict in the target modality.
            num_steps: Number of steps to unmask tokens.
        Returns:
            A list of integers representing the number of tokens to unmask at each step.
        """
        assert total_tokens > 0, "No tokens to unmask in the input sequence."
        assert num_steps > 0, "Number of steps should be greater than zero."
        assert num_steps <= total_tokens, "Number of steps should be less than or equal to the total number of tokens to unmask."
        
        tokens_per_step = total_tokens // num_steps
        remainder = total_tokens % num_steps
        schedule = [tokens_per_step] * num_steps
        schedule[-1] += remainder

        assert len(schedule) == num_steps, "Schedule length should match the number of steps."
        assert sum(schedule) == total_tokens, "Total number of tokens to unmask should match the sum of the schedule."

        return schedule

    def generate_one_modality_roar(
            self,
            enc_input_tokens: torch.LongTensor,
            enc_input_positions: torch.LongTensor,
            enc_input_modalities: torch.LongTensor,
            target_mod: str,
            num_steps: int = 8, 
            temp: float = 1.0, 
            top_p: float = 0.0, 
            top_k: float = 0.0,
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generate one modality through iterative unmasking using the Random Order Auto Regressive 
        (ROAR) decoding scheme introduced in 4M.

        Args:
            enc_input_tokens: LongTensor of shape (1, N_input) with encoder input token IDs.
            enc_input_positions: LongTensor of shape (1, N_input) with encoder input positions.
            enc_input_modalities: LongTensor of shape (1, N_input) with IDs specifying which modality
                is input at each position.
            target_mod: The target modality to generate.
            num_steps: Number of unmasking steps to perform.
            temp: Temperature for sampling.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            pred_tokens: LongTensor of shape (1, N_target) with the predicted tokens for the target modality.
            enc_input_tokens: LongTensor of shape (1, N_input+N_target) with the updated encoder input token IDs.
            enc_input_positions: LongTensor of shape (1, N_input+N_target) with the updated encoder input positions.
            enc_input_modalities: LongTensor of shape (1, N_input+N_target) with the updated IDs specifying which modality
        """
        B, N = enc_input_tokens.shape
        assert B == 1
        device = enc_input_tokens.device
        target_mod_index = self.modalities.index(target_mod)
        n_tokens_target = self.max_seq_lens[target_mod_index]

        # Get schedule for unmasking tokens
        schedule = self.get_unmasking_schedule(n_tokens_target, num_steps)

        # TODO: There are n_tokens_target positions to predict, and we will unmask them in `num_steps` steps.
        # The order in which we unmask the tokens is arbitrary, but here we will use a random order.
        # That means, we will randomly shuffle the positions from 0 to n_tokens_target - 1, and then
        # split them into `num_steps` steps. 
        ???
        # dec_input_positions_list is a list of position indices of shape (1, k) for each step. 
        # Together, they should contain all the positions from 0 to n_tokens_target - 1 exactly once.
        dec_input_positions_list = ???
        
        for step, k in enumerate(schedule):
            # Select the k positions to predict for this step
            dec_input_positions = dec_input_positions_list[step]
            # Create a tensor of k IDs specifying the target modality
            dec_input_modalities = target_mod_index * torch.ones(1, k, device=device, dtype=torch.long)

            # TODO: Forward pass through the model to get the next tokens' logits. 
            # Select the 0-th element to get shape: [k, vocab_size]
            predicted_logits = ???

            # TODO: Sample new tokens for the predicted_logits
            # Hint: Use the sample_tokens function from utils/sampling.py
            # Make sure to pass the `temp`, `top_k` and `top_p` arguments
            samples, _ = ???

            # TODO: Concatenate the new tokens to the encoder input tokens for the next step
            # Specifically, concatenate the k samples to enc_input_tokens, the k dec_input_positions
            # to enc_input_positions, and the k dec_input_modalities to enc_input_modalities.
            # The resulting shapes for each tensor should be [1, N_prev + k].
            enc_input_tokens = ???
            enc_input_positions = ???
            enc_input_modalities = ???
                   
        # Select the predicted tokens for the target modality and unshuffle them
        pred_tokens = enc_input_tokens[enc_input_modalities == target_mod_index]
        indices = enc_input_positions[enc_input_modalities == target_mod_index]
        pred_tokens = pred_tokens[indices.argsort()].unsqueeze(0)
        
        return pred_tokens, enc_input_tokens, enc_input_positions, enc_input_modalities
