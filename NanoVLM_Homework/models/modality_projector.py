# Modality Projection from Vision to Language
import torch.nn as nn

class ModalityProjector(nn.Module):
    """
    Modality Projector for a Vision-Language Model.

    Bridges a Vision Transformer (ViT) encoder and a language model (LM) by:
      1. Applying pixel shuffle to spatially merge patch tokens which leads to reduction of sequence
         length by ``scale_factor**2`` and expansion of the embedding size by the same factor.
      2. Projecting the repacked embeddings into the LM's hidden dimension via a
         learned linear layer.

    Pixel shuffle exploits the implicit 2-D spatial grid structure of ViT patch tokens.
    With a scale_factor of 2, every 2×2 block of neighboring tokens is re-shuffled into
    one token — token count shrinks by 4×, embedding size grows by 4×, so no
    information is discarded, just repacked.

    Args:
        cfg: Configuration object with the following attributes:

            - **vit_hidden_dim** (*int*): Hidden dimension of the ViT encoder.
            - **lm_hidden_dim** (*int*): Hidden dimension of the language model.
            - **mp_pixel_shuffle_factor** (*int*): Spatial downscale factor for pixel
              shuffle (e.g. ``2`` merges 2×2 blocks).

    Attributes:
        input_dim (int): Projector input dimension —
            ``vit_hidden_dim * mp_pixel_shuffle_factor**2``.
        output_dim (int): Projector output dimension — ``lm_hidden_dim``.
        scale_factor (int): Pixel-shuffle spatial scale factor.
        proj (nn.Linear): Linear projection from ``input_dim`` to ``output_dim``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor ** 2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        ## TODO
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x):
        """
        Spatially merge neighbouring patch tokens via pixel shuffle.

        Treats the flat sequence of ViT patch tokens as an implicit 2-D grid and
        folds every ``scale_factor × scale_factor`` block of neighbouring tokens
        into a single token with a proportionally larger embedding.

        Steps
        -----
        1. Reshape ``(B, S, E)`` → ``(B, H, W, E)``  where ``H = W = √S``.
        2. Split spatial dims by scale factor:
           ``(B, H, W, E)`` → ``(B, h_out, sf, w_out, sf, E)``
           where ``h_out = H // sf``, ``w_out = W // sf``.
        3. Permute to bring scale-factor axes adjacent to the embedding dim:
           ``→ (B, h_out, w_out, sf, sf, E)``.
        4. Merge scale-factor axes into the embedding:
           ``→ (B, h_out * w_out, sf * sf * E)``.

        Args:
            x (torch.Tensor): Patch token sequence of shape
                ``(batch_size, seq_len, embed_dim)``, where ``seq_len`` must be a
                perfect square and ``√seq_len`` must be divisible by ``scale_factor``.

        Returns:
            torch.Tensor: Repacked token sequence of shape
                ``(batch_size, seq_len // scale_factor**2, embed_dim * scale_factor**2)``.

        Raises:
            AssertionError: If ``seq_len`` is not a perfect square.
            AssertionError: If ``√seq_len`` is not divisible by ``scale_factor``.
        """
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq
        assert seq_root % self.scale_factor == 0

        ## TODO
        height = width = seq_root # set height and width
        x = x.view(bsz, height, width, embed_dim)  # reshape x by expanding sequence dimension to height and width dimensions
        h_out = height // self.scale_factor # new height downsampled by scale factor
        w_out = width // self.scale_factor # new width downsampled by scale factor

        x = x.view(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim) # reshape
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # permute
        x = x.view(bsz, h_out * w_out, embed_dim * self.scale_factor * self.scale_factor) # merge

        return x # expected shape → (B, h_out * w_out, sf * sf * E)

    def forward(self, x):
        """
        Project ViT patch tokens into the language model's embedding space.

        Args:
            x (torch.Tensor): ViT patch token sequence of shape
                ``(batch_size, seq_len, vit_hidden_dim)``, where ``seq_len`` must be a
                perfect square, and ``√seq_len`` must be divisible by ``scale_factor``.

        Returns:
            torch.Tensor: Projected token sequence of shape
                ``(batch_size, seq_len // scale_factor**2, lm_hidden_dim)``.
        """
        ## TODO
        x = self.pixel_shuffle(x) # perform pixel shuffle operation
        x = self.proj(x) # perform linear projection

        return x

    