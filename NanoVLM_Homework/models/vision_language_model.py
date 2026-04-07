import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model


MODEL_CARD_TEMPLATE = """
---
language: en
license: mit
library_name: nanovlm
tags:
- vision-language
- multimodal
- smollm2
- siglip
---

# nanoVLM - {repo_id}

This is a nano Vision-Language Model (nanoVLM) trained as part of the COM-304 course.

## Model Description
The model consists of three main components:
- **Vision Backbone**: Pretrained `google/siglip-base-patch16-224`
- **Language Model**: Pretrained `HuggingFaceTB/SmolLM2-135M`
- **Modality Projector**: A learnable linear layer with Pixel Shuffle reduction.

## Usage
You can load this model using the `VisionLanguageModel` class from the `nanovlm` repository.

```python
from models.vision_language_model import VisionLanguageModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionLanguageModel.from_pretrained("{repo_id}").to(device)
```
"""


class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone

    def forward(self, input_ids, image, attention_mask=None, targets=None):

        # TODO
        # Step 1: Compute image embeddings
        # Process image through vision backbone and vision modality projector
        image_embeds = ...


        # Step 2: Compute text embeddings
        # Get text embeddings using the token_embedding layer of self.decoder
        text_embeds = ...


        # Step 3: Concatenate image and text embeddings
        combined_embeds = ...


        # Step 4: Extend the attention mask
        # The current attention_mask only covers text tokens (B, T)
        # Note: image tokens should always be attended to
        if attention_mask is not None:
            image_attention = ... # define attention mask for image
            attention_mask = ...  # combined attention mask

        # Step 5: LLM forward pass
        # Pass combined embeddings and attention mask to the LLM decoder to get the final token embeddings
        output_token_embeddings = ...

        loss = None
        # Step 6, 7 & 8: Compute Loss (only if targets are provided)
        if targets is not None:
            # Step 6: Project the embeddings to vocabulary distribution via decoder head (self.decoder.head)
            logits = ...

            # Step 7: Obtain the text part of logits (ignore image tokens)
            logits = ...

            # Step 8: Compute Cross-Entropy loss on answer tokens only
            # Hint: use ignore_index to mask out non-answer tokens
            loss = ...

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        """
        VLM Autoregressive Generation
        ════════════════════════════════════════════════════════════════
        Inputs:
            input_ids      : (B, T)       — tokenized text prompt
            image          : (B, C, H, W) — raw image
            attention_mask : (B, T)       — text attention mask
            max_new_tokens : int          — number of tokens to generate
        ════════════════════════════════════════════════════════════════
        """

        # TODO
        # Step 1: Image Embeddings
        # Pass image through vision encoder and modality projector
        image_embd_encoder = ...
        image_embd = ...


        # Step 2: Text Token Embeddings
        # Embed the input token ids using the decoder's token_embedding layer (self.decoder.token_embedding)
        token_embd = ...


        # Step 3: Concatenate image and text embeddings
        combined_embed = ...
        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)

        # Step 4: Extend Attention Mask
        if attention_mask is not None:
            image_attention_mask = ...  # hint: we want all image tokens to be attended
            attention_mask = ...  # concat image_attention + text attention_mask

        # Step 5: Autoregressive Generation Loop
        # At each sub-step (till max_new_tokens):
        #   (i)   pass current embeddings to decoder
        #   (ii)  get logits for the last token only
        #   (iii) apply language model decoder head (self.decoder.head) if decoder is in embedding mode (check self.decoder.lm_use_tokens)
        #   (iv)  sample next token by first applying the softmax (torch.softmax) and then the multinomial sampling (torch.multinomial)
        #   (v)   Now prepare the input for next forward pass: embed the new token and append to output sequence
        #   (vi)  extend attention mask to accommodate new token
        #   (vii) stop the generation loop if the generated token == 2 (token id 2 is the EOS token we used during training)
        #   (viii) return the generated tokens

        outputs = combined_embed
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)

        for i in range(...):

            model_out = ....  # (i)


            last_token_logits = ... # (ii)

            if ...:  # (iii)
                last_token_logits = ...

            probs = ... # (iv)
            next_token = ...
            ... # fill in the generated next token in generated_tokens at ith position


            generated_embed = ... # (v)
            outputs = ...  # concat generated_embed to outputs along dim=1

            if attention_mask is not None:
                attention_mask = ... # (vi)


            if ...: # vii)
                break

        return ... # (viii)

    @torch.no_grad()
    def generate_with_kv_cache(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        """
        Autoregressive generation with KV caching.

        Two phases:
          PREFILL — run the full prompt (image + text) through the decoder once,
                    collecting the K and V matrices for every layer.
          DECODE  — at each later step, feed only the single new token;
                    every attention layer reuses its cached K/V instead of
                    reprocessing the entire history.

        Parameters
        ----------
        input_ids      : (B, T)       tokenised text prompt
        image          : (B, C, H, W) raw image tensor
        attention_mask : (B, T)       text attention mask  (optional)
        max_new_tokens : int          number of tokens to generate
        """
        
        # Step 1: Build combined image + text embeddings
        # hint: you've done this in the previous exercise
        image_embd = ...
        token_embd = ...
        combined_embd = ...
        batch_size = image_embd.size(0)

        # Step 2: PREFILL
        # Run the full prompt through decoder.forward_kv (past_key_values should be None at the start).
        model_out, past_key_values = ...

        # Step 3: Obtain the first generated token from the last position
        last_logits = ...
        if not self.decoder.lm_use_tokens:
            last_logits = ... # apply lm head (self.decoder.head) if decoder is in embedding mode

        # Step 4: Sample new token by first applying the softmax (torch.softmax) and then the multinomial sampling (torch.multinomial)
        probs = ...
        next_token = ...

        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        ... # fill in the generated next token in generated_tokens at 0th position

        # Step 5: DECODE LOOP
        for i in range(1, max_new_tokens):

            next_embd = ... # embed only the single last-generated token

            model_out, past_key_values = ... # decoder.forward_kv processes 1 token but attends over full history via cache

            last_logits = ... # obtain the last token logits
            if not self.decoder.lm_use_tokens:
                last_logits = ... # apply lm head (self.decoder.head) if decoder is in embedding mode

            probs = ...
            next_token = ...
            ... # fill in the generated next token in generated_tokens at ith position

            if ...: # stop the generation loop if the generated token == 2 (token id 2 is the EOS token we used during training)
                break

        return generated_tokens

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


