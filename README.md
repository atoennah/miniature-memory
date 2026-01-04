# miniature-memory

A minimal, memory-aware dataset and training pipeline for small GPT-style models, optimized for tight resource constraints.

## Project Purpose

This project exists to build, study, and deploy language models for **adult entertainment**. The entire pipeline, from data scraping to model training, is designed for this specific purpose. The project's intent is locked and non-negotiable.

## Getting Started

This repository contains a fully automated pipeline for data scraping, processing, and model training.

- **For a complete technical overview, see the [Developer Guide](DEVELOPER_GUIDE.md).** The guide contains the project roadmap, architecture, data pipeline details, and setup instructions.

- **To understand the contribution rules and the project's locked intent, see [CONTRIBUTING.md](CONTRIBUTING.md).**

### Quick Setup
To set up the environment and install dependencies, run:
```bash
./setup.sh
```

### Running the Pipeline
To run the entire end-to-end pipeline (data processing and training):
```bash
python3 run.py
```

## Philosophy

This project follows a "constraint-first" philosophy. The goal is to push a from-scratch NanoGPT implementation as far as possible under minimal hardware resources (~2GB RAM, low-end CPU/GPU).

This is achieved through:
- A small, character-level vocabulary
- Small context windows and batch sizes
- A fully automated, checkpoint-based training system that can grow and resume over time.

Everything is designed to be explicit, auditable, and reproducible. No black boxes.
