"""
Unit tests for Exercise 2: VQACollator

Run this file directly to verify your implementation:
    %run tests/test_collator.py       (from Jupyter)
    python tests/test_collator.py     (from terminal)
"""

import torch
from transformers import AutoTokenizer
from data.collators import VQACollator

MODEL_ID  = "HuggingFaceTB/cosmo2-tokenizer"
MAX_LEN   = 20

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
collator  = VQACollator(tokenizer, MAX_LEN)


def make_sample(question, answer):
    return {
        'image':     torch.zeros(3, 8, 8),  # dummy image
        'text_data': question,
        'answer':    answer,
    }


# Test 1: output tensor shapes
batch = [make_sample('What colour is the sky?', ' Blue.'),
         make_sample('How many dogs?',           ' Two.')]
out   = collator(batch)

assert out['image'].shape          == (2, 3, 8, 8), 'image shape wrong'
assert out['input_ids'].shape      == (2, MAX_LEN), 'input_ids shape wrong'
assert out['attention_mask'].shape == (2, MAX_LEN), 'attention_mask shape wrong'
assert out['labels'].shape         == (2, MAX_LEN), 'labels shape wrong'
print('Test 1 passed: shapes are correct')

# Test 2: labels at padding positions are -100
for i in range(2):
    pad_positions = (out['attention_mask'][i] == 0).nonzero(as_tuple=True)[0]
    if len(pad_positions):
        assert (out['labels'][i, pad_positions] == -100).all(), \
               f'Sample {i}: padding labels should be -100'
print('Test 2 passed: padding positions masked correctly')

# Test 3: last column is always -100
assert (out['labels'][:, -1] == -100).all(), 'last column must be -100'
print('Test 3 passed: last column is -100')

# Test 4: at least one non-(-100) label exists per sample
for i in range(2):
    assert (out['labels'][i] != -100).any(), \
           f'Sample {i}: no trainable label found — answer may be fully masked'
print('Test 4 passed: each sample has at least one trainable label')

# Test 5: truncated samples are fully masked
long_answer  = ' ' + 'yes ' * 50  # forces truncation
batch_trunc  = [make_sample('Short question?', long_answer)]
out_trunc    = collator(batch_trunc)
assert (out_trunc['labels'][0] == -100).all(), \
       'Truncated sample should have all labels = -100'
print('Test 5 passed: truncated sample fully masked')

print('\nAll tests passed!')
