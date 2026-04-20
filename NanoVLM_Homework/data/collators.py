import torch

class VQACollator(object):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]
        # TODO
        # Step 1 — Stack images
        # torch.stack the list of image tensors into a single batched tensor.
        images = torch.stack(images)

        # Step 2 — Build input sequences
        # The model expects a single continuous sequence per sample: f"{text}{answer}"
        # No separator is needed — plain string concatenation is enough.
        # Produce a list `input_sequences` of merged strings.
        input_sequences = []
        for i in range(len(texts)):
            input_sequences.append(texts[i] + answers[i])

        # Step 3 — Batch tokenization
        # Tokenize `input_sequences` with `self.tokenizer.batch_encode_plus`.
        # Use left-padding and right-truncation up to `self.max_length`.
        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            padding_side="left",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Retrieve `input_ids` (Long tensor) and `attention_mask` from the result.
        input_ids = encoded_full_sequences['input_ids']
        attention_mask = encoded_full_sequences['attention_mask']


        # Step 4 — Create causal labels
        labels = input_ids.clone()       # clone the input_ids
        labels[:, :-1] = input_ids[:, 1:].clone() # in a causal LM, the target at position t is the token at position t+1
        labels[:, -1] = -100 # make sure, no target (label = -100) for the very last position

        # Step 5 — Per-sample label masking
        # Even though we created labels for the input_ids, we need to mask the loss calculation for tokens that belongs to pad positions, and for the examples that were truncated

        # The tokenizer has different behavior for padding and truncation:
        # 1. If the full text (answer + question) is shorter than the max length, it gets padded on the left
        # 2. If the full text is longer than the max length, it gets truncated on the right
        # Therefore, I need to handle multiple cases, this is the different scenarios:
        # If the full text is longer than the max length, we need to set the labels to -100 for the whole sample (we want to ignore the whole sample)
        # If the full text is shorter than the max length, we need to set the labels to -100 only for the question part, and create causal language modeling labels for the answer part, taking into account the padding

        # First, compute the *untruncated* `original_lengths` token length for every input sequence
        original_lengths = [
            len(self.tokenizer.encode(seq, add_special_tokens=False))
            for seq in input_sequences
        ]
        answer_lengths = [
            len(self.tokenizer.encode(ans, add_special_tokens=False))
            for ans in answers
        ]
        # Then iterate over each sample i of the batch and handle two cases:
        for i in range(len(batch)):
            # Get the length of the question for this sample
            question_length = original_lengths[i] - answer_lengths[i]

            if original_lengths[i] > self.max_length:
                # case A: sequence was truncated — ignore the whole sample
                labels[i, :] = -100
            else:
                # Case B: sequence fits within max_length (left-padded).
                # The tokenizer left-pads short sequences, so find where non-padding tokens begin (using attention_mask)
                first_token_pos = (attention_mask[i] != 0).nonzero(as_tuple=True)[0][0] # the first 1 (non-zero value) in the attention mask for a given batch sample marks the first non-padding token
                # using first_token_pos and question_length to find the position where the question ends. Also, because labels are already shifted left by 1 relative to input_ids,
                # take into account the left shift by subtracting 1)
                question_end = first_token_pos + question_length - 1
                labels[i, :question_end] = -100 # mask padding + question tokens, keep answer tokens trainable

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class MMStarCollator(object):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        questions = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Stack images
        images = torch.stack(images)

        encoded_question_sequences = self.tokenizer.batch_encode_plus(
            questions,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        encoded_answer_sequences = self.tokenizer.batch_encode_plus(
            answers,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        return {
            "images": images,
            "input_ids": encoded_question_sequences['input_ids'],
            "attention_mask": encoded_question_sequences['attention_mask'],
            "labels": encoded_answer_sequences['input_ids'],
        }