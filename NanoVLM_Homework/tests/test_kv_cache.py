"""
Helpers used by the Exercise 5b cells in the lecture notebook.
Import these into the notebook rather than defining them inline.
"""

import time
import torch


def benchmark_generate(model, input_ids, img_tensor, max_new_tokens, n_runs, device, use_kv=False):
    """
    Time generate() or generate_with_kv_cache() over n_runs and return
    the best (minimum) wall-clock time in milliseconds.
    """
    times = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_kv:
                model.generate_with_kv_cache(input_ids, img_tensor, max_new_tokens=max_new_tokens)
            else:
                model.generate(input_ids, img_tensor, max_new_tokens=max_new_tokens)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000 * min(times)


def print_speedup_summary(naive_ms, kv_ms, max_new_tokens, input_ids, img_tokens=49):
    """Print a comparison table and theoretical speedup analysis."""
    speedup = naive_ms / kv_ms

    print(f"{'Method':<30} {'Total (ms)':>12} {'ms / token':>12}")
    print("-" * 56)
    print(f"{'Naive generate()':30} {naive_ms:12.0f} {naive_ms / max_new_tokens:12.1f}")
    print(f"{'KV-cache generate()':30} {kv_ms:12.0f} {kv_ms / max_new_tokens:12.1f}")
    print("-" * 56)
    print(f"\nObserved speedup : {speedup:.1f}x")

    print(f"(Actual speedup includes prefill overhead, so empirical number is lower)")


@torch.no_grad()
def greedy_generate(mdl, ids, img_t, n_tok, use_kv=False):
    """Generate n_tok tokens greedily (argmax), optionally with a KV cache."""
    img_embd  = mdl.MP(mdl.vision_encoder(img_t))      # (B, img_len, D)
    tok_embd  = mdl.decoder.token_embedding(ids)        # (B, T,       D)
    combined_vision_language_token_embeddings  = torch.cat([img_embd, tok_embd], dim=1)  # (B, img_len+T, D)
    B, img_len = img_embd.size(0), img_embd.size(1)

    generated = []
    if use_kv:
        out, past_kv = mdl.decoder.forward_kv(combined_vision_language_token_embeddings)
    else:
        out     = mdl.decoder(combined_vision_language_token_embeddings)
        past_kv = None

    _logits  = mdl.decoder.head(out[:, -1, :]) if not mdl.decoder.lm_use_tokens else out[:, -1, :]
    next_tok = _logits.argmax(dim=-1, keepdim=True)
    generated.append(next_tok)

    for _ in range(1, n_tok):
        if use_kv:
            embd = mdl.decoder.token_embedding(next_tok)       # (B, 1, D)
            out, past_kv = mdl.decoder.forward_kv(
                embd,
                past_key_values=past_kv,
            )
        else:
            all_ids  = torch.cat([ids] + generated, dim=1)
            all_embd = mdl.decoder.token_embedding(all_ids)
            all_embd = torch.cat([img_embd, all_embd], dim=1)
            full_mask = torch.ones(B, all_embd.size(1), device=ids.device)
            out = mdl.decoder(all_embd, full_mask)

        _logits  = mdl.decoder.head(out[:, -1, :]) if not mdl.decoder.lm_use_tokens else out[:, -1, :]
        next_tok = _logits.argmax(dim=-1, keepdim=True)
        generated.append(next_tok)

    return torch.cat(generated, dim=1)  # (B, n_tok)
