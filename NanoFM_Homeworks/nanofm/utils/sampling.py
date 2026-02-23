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
# --------------------------------------------------------
# Based on:
# https://github.com/apple/ml-4m/
# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np


def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0):
    # Compatible with batching
    # From https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    if top_k > 0.0:
        if isinstance(top_k, int):
            k = min(top_k, logits.shape[-1])
        elif isinstance(top_k, float):
            k = min(int(top_k * logits.shape[-1]), logits.shape[-1])
        else:
            raise ValueError(f"Invalid value for top_k: {top_k}")

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        restore_indices = torch.argsort(sorted_indices, dim=-1)
        indices_to_remove = torch.gather(sorted_indices_to_remove, dim=-1, index=restore_indices)
        logits[indices_to_remove] = float("-inf")

    return logits

def sample_tokens(logits, temperature=1.0, top_k=0.0, top_p=0.0):
    if np.isclose(temperature, 0, atol=1e-10):
        samples = torch.argmax(logits, dim=-1)
        # Since argmax is used, all sampled_probs will be 1 as we're selecting the max probability
        sampled_probs = torch.ones_like(samples, dtype=torch.float32)
    else:
        filtered_logits = top_k_top_p_filtering(logits, top_k, top_p)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        samples = torch.multinomial(probs, 1)[:, 0]
        sampled_probs = probs[torch.arange(len(samples)), samples]
    return samples, sampled_probs