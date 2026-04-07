import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image


def visualize_patch_embeddings(patch_embeddings, image):
    """
    Project SigLIP patch embeddings to 3D with PCA and display them alongside
    the original image. Semantically similar regions map to similar colours,
    illustrating that the encoder has learned meaningful structure.

    Args:
        patch_embeddings: torch.Tensor of shape (1, num_patches, vision_dim)
        image: PIL.Image — the original input image
    """
    feats = patch_embeddings[0].cpu().numpy()            # (num_patches, vision_dim)
    grid  = int(feats.shape[0] ** 0.5)                   # e.g. 14 for 196 patches

    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(feats)                 # (num_patches, 3)
    print(f'Variance explained by top-3 PCs: {pca.explained_variance_ratio_.sum()*100:.1f}%')

    lo, hi = pca_feats.min(0), pca_feats.max(0)
    pca_feats = (pca_feats - lo) / (hi - lo + 1e-8)
    pca_img = pca_feats.reshape(grid, grid, 3)

    # Upscale to the original image's aspect ratio so both panels look comparable.
    # NEAREST keeps the blocky patch boundaries visible — a useful visual cue.
    pca_pil = Image.fromarray((pca_img * 255).astype(np.uint8)).resize(
        (image.width, image.height), Image.NEAREST
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=13)
    axes[0].axis('off')
    axes[1].imshow(pca_pil)
    axes[1].set_title('SigLIP Feature Map  (top-3 PCA → RGB)', fontsize=13)
    axes[1].axis('off')
    plt.suptitle('Semantically similar regions share similar colours in the feature map',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


def launch_chatbot(model, image_path, image_processor, tokenizer, device, max_new_tokens=100):
    """
    Launch an interactive ipywidgets chatbot for a single image.
    The user types a question, clicks Ask, and the model generates an answer.
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = image_processor(img).unsqueeze(0).to(device)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Ask me about this image!', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    question_box = widgets.Text(
        placeholder='Type your question here...',
        layout=widgets.Layout(width='75%'),
    )
    ask_button = widgets.Button(
        description='Ask',
        button_style='primary',
        layout=widgets.Layout(width='10%'),
    )
    output_area = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #d0d0d0',
            padding='8px',
            margin='6px 0',
            min_height='40px',
        )
    )

    def on_ask_clicked(b):
        with output_area:
            clear_output()
            question = question_box.value.strip()
            if not question:
                print("Please enter a question.")
                return
            template  = f"Question: {question}\nAnswer:"
            input_ids = tokenizer.encode(template, return_tensors='pt').to(device)
            with torch.no_grad():
                gen = model.generate(input_ids, img_tensor, max_new_tokens=max_new_tokens)
            answer = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
            print(f"Answer: {answer}")

    ask_button.on_click(on_ask_clicked)
    display(widgets.VBox([widgets.HBox([question_box, ask_button]), output_area]))


def visualize_sample(sample, max_turns=None):
    """
    Display a training sample as a side-by-side figure:
      left  — the raw image
      right — the QA turns (up to max_turns; None = show all)
    """
    texts = sample['texts']
    turns_to_show = texts if max_turns is None else texts[:max_turns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(sample['images'][0])
    axes[0].set_title('Input Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].axis('off')
    display_text = ''
    for i, turn in enumerate(turns_to_show):
        display_text += f'Turn {i + 1}:\n\n{turn["user"]}\n---\n{turn["assistant"]}\n\n'
    if max_turns is not None and len(texts) > max_turns:
        display_text += f'... ({len(texts) - max_turns} more turns)'
    axes[1].text(0.05, 0.95, display_text, transform=axes[1].transAxes,
                 fontsize=10, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))
    axes[1].set_title('Question-Answer Turns', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()
    print(f'Number of turns in this sample: {len(texts)}')


def check_projector_output(patch_embeddings, image_tokens, cfg):
    """
    Print a shape summary for the modality projector and assert the output is correct.
    Raises AssertionError with a helpful message if the shape is wrong.
    """
    print('After projection:')
    print(f'  Input  (patch embeddings): {patch_embeddings.shape}'
          f'  [{patch_embeddings.shape[1]} tokens × {patch_embeddings.shape[2]} dims]')
    print(f'  Output (image tokens):     {image_tokens.shape}'
          f'  [{image_tokens.shape[1]} tokens × {image_tokens.shape[2]} dims]')
    print()
    print(f'  PixelShuffle reduced {patch_embeddings.shape[1]} → {image_tokens.shape[1]} tokens')
    print(f'  Linear projected dim {patch_embeddings.shape[2]} → {image_tokens.shape[2]}')
    print()
    print('These image tokens are now in the SAME embedding space as text tokens!')

    expected_tokens = (cfg.vit_img_size // 16) ** 2 // cfg.mp_pixel_shuffle_factor ** 2
    expected_dim    = cfg.lm_hidden_dim
    assert image_tokens.shape == (1, expected_tokens, expected_dim), (
        f"Shape mismatch! "
        f"Expected (1, {expected_tokens}, {expected_dim}), "
        f"got {tuple(image_tokens.shape)}. "
        f"Check your pixel_shuffle() and forward() implementations."
    )
    print(f'Shape check passed: {tuple(image_tokens.shape)}'
          f' = (batch=1, tokens={expected_tokens}, dim={expected_dim})')


def visualize_text_tokenization(conversation, tokenizer, n_tokens=15):
    """
    Display a side-by-side figure:
      left  — the raw conversation string
      right — the first n_tokens token id → decoded string pairs
    """
    token_ids = tokenizer.encode(conversation)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].axis('off')
    axes[0].text(0.05, 0.95, conversation, transform=axes[0].transAxes,
                 fontsize=10, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))
    axes[0].set_title('Conversation String', fontsize=13, fontweight='bold', loc='left', x=0.05)

    axes[1].axis('off')
    token_text = f'Total tokens: {len(token_ids)}\n\n'
    token_text += f'First {n_tokens} tokens (id → string):\n\n'
    for tid in token_ids[:n_tokens]:
        token_text += f'  {tid:6d}  →  {repr(tokenizer.decode([tid]))}\n'
    axes[1].text(0.05, 0.95, token_text, transform=axes[1].transAxes,
                 fontsize=10, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f5fff0', alpha=0.8))
    axes[1].set_title('Token Breakdown', fontsize=13, fontweight='bold', loc='left', x=0.05)

    plt.tight_layout()
    plt.show()


def format_conversation(turns, eos_token='<|endoftext|>'):
    """
    nanoVLM conversation template.
    Each turn becomes: '<user_text>\n<assistant_text>'
    All turns are joined with a newline, and a single EOS token is appended at the end.
    """
    formatted = '\n'.join(f'{turn["user"]}\n{turn["assistant"]}' for turn in turns)
    return formatted + eos_token
