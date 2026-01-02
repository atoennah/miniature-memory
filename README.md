# miniature-memory

A minimal, memory-aware dataset and training pipeline for small GPT-style models.

This project is an experiment in doing more with less. We are building, growing, and training a small-but-mighty text-based language model from scratch, optimized for tight resource constraints (like the free tier of Google Colab).

Our philosophy is "constraint-first." Instead of scaling up, we're scaling smart, focusing on a clean, ever-growing dataset and a fully-automated pipeline.

## What This Project Does

- Scrapes text content from the web.
- Extracts readable text using a browser-style “read mode.”
- Stores raw text in a carefully structured, append-only dataset.
- Cleans, prepares, and tokenizes data for training.
- Incrementally trains a small GPT-style model.
- Saves checkpoints so training can resume anytime.

This project is not a large-model framework. It is intentionally small, explicit, and designed for learning and experimentation.

## Getting Started

Welcome! Whether you're a new developer or a seasoned machine learning engineer, we have a place for you.

- **To understand the project's vision and future,** start with our **[ROADMAP.md](ROADMAP.md)**.
- **For a full technical breakdown, setup instructions, and contribution guidelines,** please see our **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**.

## Our Philosophy

We believe that great things can be built with limited resources. This project is a testament to that idea. By focusing on a strong data foundation and a reproducible, automated workflow, we aim to push a NanoGPT-style model as far as it can go.

No black boxes, no magic, no assumptions. Just clean code and a clear process.

Join us!

