# Training Data Format

This document specifies the definitive format for the training data (`train.txt`) used in the miniature-memory project. All data preparation and model training must adhere to this specification to ensure consistency and reproducibility.

## 1. Tokenization

The project uses **character-level tokenization**.

### Rationale:

-   **Minimal Vocabulary:** The set of unique characters is very small (~100-200), resulting in a tiny embedding table and minimal memory usage.
-   **Robustness:** Character-level models are naturally resilient to typos, slang, and stylistic variations present in web-scraped data.
-   **No Dependencies:** It requires no external tokenizer library or pre-trained vocabulary file.
-   **Constraint-First:** While it results in longer sequence lengths, this is an acceptable trade-off given the project's hard `block_size` limit.

## 2. File Encoding

All text files, from `raw` to `processed`, must be encoded in **UTF-8**.

## 3. Control Tokens

To provide structural and narrative context to the model, the following control tokens are defined. These tokens will be inserted into the text during the (not-yet-implemented) narrative structuring phase.

The format for all control tokens is `<|token_name|>`.

### Defined Tokens:

-   `<|scene_break|>`: Indicates a significant break between scenes, such as a change in time, location, or focus.
-   `<|summary|>`: Prefixes a condensed summary of previous text, to be used for the rolling summary memory mechanism.
-   `<|end_of_text|>`: A special token to explicitly mark the end of a distinct story or document. This helps the model learn boundaries.

### Example Usage in `train.txt`:

```
This is the first part of the story. It ends here.
<|scene_break|>
This is the second scene, taking place hours later.
...
and that's the end of the story.
<|end_of_text|>
This is a completely new story starting now.
...
```

### Rationale for Format:

The `<|...|>` format is chosen because the sequence `<|` is extremely unlikely to appear naturally in narrative English or Indonesian text, preventing accidental tokenization. It is also a convention used in other large language models (e.g., GPT-2), making it a familiar pattern.
