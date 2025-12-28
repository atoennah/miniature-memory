# miniature-memory

This project is dedicated to building, studying, and deploying language models for adult entertainment. For more information on the project's purpose, scope, and rules, please see `CONTRIBUTING.md`.
NanoGPT-Lite: Constraint-First Storytelling Model

A from-scratch NanoGPT implementation optimized for ultra-low RAM environments (≈2 GB), focused on long-form narrative coherence under tight constraints.

This project explores a constraint-first design philosophy: instead of scaling parameters, we co-design architecture, data format, and inference strategy to preserve storytelling quality on minimal hardware.


---

Goals

✅ Run inference on 2 GB RAM, CPU-only

✅ Maintain coherent multi-paragraph narrative flow

✅ Use from-scratch GPT architecture (no pretrained weights)

✅ Be readable, hackable, and modifiable

❌ Not designed for large-scale training

❌ Not optimized for GPUs or large contexts



---

What This Is (and Isn’t)

This is:

Decoder-only Transformer (GPT-style)

Character-level or small-BPE tokenizer

Trained from scratch

Optimized for inference under tight memory


This is NOT:

GPT-4 architecture

A general-purpose LLM

A high-throughput serving system

A large-context model (2k–8k tokens)



---

Key Design Ideas

1. Constraint-First Architecture

Instead of asking “how big can we go?”, we ask:

> What is the smallest model that still tells a coherent story?



Memory pressure comes mostly from:

attention matrices (T × T)

KV cache growth

embeddings


So we cap context aggressively and compensate elsewhere.


---

2. Hard Context + Soft Memory

We use:

Small fixed context (e.g. 256 tokens)

Rolling summary tokens prepended at inference


This simulates long-term memory without quadratic attention growth.


---

3. Character-Level Tokenization (By Design)

Char-level tokenization is chosen because:

tiny vocabulary (~100 tokens)

minimal embedding memory

robust to slang, typos, and stylistic text

zero tokenizer dependencies


Longer sequences are acceptable because context is capped anyway.


---

4. Narrow-but-Deep Tradeoff

We favor:

smaller embeddings + more layers

over:

wider embeddings + shallow depth

This improves narrative flow while keeping memory low.


---

Recommended Model Config (2 GB Safe)

n_embd     = 256
n_head     = 4
n_layer    = 7
block_size = 256
tokenizer  = character-level
precision  = FP16 (inference)

Approximate size:

18–25M parameters

~300–500 MB total RAM usage (Python + PyTorch included)



---

Project Structure

.
├── data/
│   └── train.txt          # narrative training corpus
├── model.py               # NanoGPT model definition
├── tokenizer.py           # char-level tokenizer
├── train.py               # training loop (offline)
├── generate.py            # inference / sampling
├── summary.py             # rolling-summary logic
├── utils.py
└── README.md


---

Training Notes

⚠️ Training on a 2 GB machine is not recommended.

Typical workflow:

1. Train on a larger machine (PC / cloud)


2. Save weights in FP16


3. Deploy inference on low-RAM system



Suggested optimizer:

AdamW(lr=1e-4, betas=(0.9, 0.95))

Training tips:

use long contiguous text, not shuffled samples

avoid over-cleaning the dataset

train past loss plateau for stylistic coherence



---

Inference Tips (Critical)

Always:

batch_size = 1

torch.no_grad()

FP16 weights

capped context


Recommended sampling:

temperature = 0.7–0.9
top_p       = 0.9
repetition_penalty ≈ 1.1

Avoid:

greedy decoding

very long generations without summarization

large batch inference



---

Memory-Saving Techniques Used

FP16 weights

small vocab embedding table

capped context length

rolling summary tokens

optional KV-cache pruning


These keep memory usage flat over time, even for long sessions.


---

Why This Is Interesting

Most “small GPT” projects just shrink parameters.

This project explores:

data-level compression

context management as a first-class concern

story quality per megabyte, not per parameter


That makes it useful for:

low-end servers

embedded / edge deployments

experimental narrative systems

research on constrained generation



---

Limitations

No long-range attention

No factual reliability

No safety layers included

Single-style bias (depends heavily on dataset)


This is a specialized engine, not a general LLM.


---

License & Use

This code is provided for research and experimentation.

You are responsible for:

dataset legality

deployment context

downstream use



---

Next Directions

Possible extensions:

asymmetric transformer blocks

LoRA fine-tuning

smarter summary compression

lightweight evaluation metrics for narrative flow

hybrid char/BPE tokenization



---

If you want, I can also:

tailor this README for open-source release

add benchmark tables

write a technical design doc

or help you name the project properly


Just say the word.
