Below is a developer-grade roadmap focused on automation, reproducibility, and collaboration, while staying implementation-realistic for Linux-based workflows.

This is written as a systems roadmap, not a tutorial to bypass safeguards. You’ll still need to ensure compliance with site terms and applicable laws.


Roadmap — Auto-Scraping & Training Pipeline

Indonesian Lewd Story Text → Growing Dataset → Automated Training

High-Level Goal

Build an end-to-end, reproducible pipeline that:

Discovers Indonesian story URLs via search Extracts readable story text Stores raw + cleaned datasets Continuously grows the dataset Automatically prepares training data Triggers training runs Keeps collaborators in sync via Git


Phase 0 — Repository & Workflow Foundations

Repo Structure (from day one)

.
├── scraper/
│   ├── search/
│   ├── fetch/
│   ├── extract/
│   └── dedupe/
├── dataset/
│   ├── raw/
│   ├── cleaned/
│   ├── rejected/
│   └── metadata/
├── processing/
│   ├── normalize.py
│   ├── segment.py
│   └── quality_filter.py
├── training/
│   ├── prepare_data.py
│   ├── train.py
│   └── configs/
├── automation/
│   ├── cron/
│   └── pipelines/
├── scripts/
│   └── cli.py
└── README.md


Ground Rules

Dataset is append-only No manual edits in dataset/raw All transformations are scripted All contributors regenerate the same outputs from the same commits


Phase 1 — URL Discovery (Search Layer)

Goal: Maintain a growing list of candidate URLs.

Approach

Use Google Search as a discovery layer, not a crawler Search queries in Indonesian:  genre keywords slang variations site-specific queries

Output

dataset/metadata/urls.jsonl

Each entry:

{
  "url": "...",
  "query": "...",
  "discovered_at": "...",
  "status": "new|fetched|rejected"
}


Key Design Choices

Never scrape directly from search results pages Discovery ≠ fetching URLs are immutable once logged


Phase 2 — Fetching HTML (Rendering Layer)

Goal: Get the human-readable version of the page.

Strategy

Python CLI wrapper Headless browser (for JS-heavy pages) Optional reader-mode rendering (browser readability layer)

Output

Raw HTML snapshot per URL

dataset/raw/html/<hash>.html


Metadata Stored

HTTP status render mode used fetch timestamp failure reason (if any)


Phase 3 — Text Extraction (Read-Mode Leverage)

Goal: Extract story text, not UI noise.

Extraction Pipeline

Reader-mode / readability algorithm Remove:  navigation comments ads   Preserve:  paragraph breaks dialogue formatting

Output

dataset/raw/text/<hash>.txt


Rejection Criteria

Too short Non-story content Wrong language Duplicates

Rejected samples go to:

dataset/rejected/



Phase 4 — Deduplication & Canonicalization

Goal: Prevent dataset rot as it grows.

Techniques

URL hash dedupe Text similarity fingerprinting Near-duplicate paragraph detection

Metadata

{
  "hash": "...",
  "duplicate_of": "...",
  "method": "exact|near"
}



Phase 5 — Dataset Cleanup & Normalization

Goal: Convert raw text into model-friendly narrative text.

Processing Steps

Normalize whitespace Standardize quotes & punctuation Remove site watermarks Normalize paragraph spacing Fix obvious encoding issues

Output

dataset/cleaned/<hash>.txt


Raw data is never modified.


Phase 6 — Narrative Structuring

Goal: Teach the model story rhythm.

Transformations

Scene segmentation Optional control tokens:
<SCENE>
<POV>
<TONE>

Paragraph length normalization

Output

Training-ready text files Manifest file listing inclusion order


Phase 7 — Training Data Configuration

Goal: Deterministic training inputs.

Build Step

Concatenate cleaned text in a stable order Generate:  train.txt val.txt

Guarantees

Same git commit → same training corpus Changes are diffable Dataset growth is transparent


Phase 8 — Automated Training (Linux)

Goal: Training runs automatically as data grows.

Automation

Triggered by:  dataset change scheduled cron job   Logs:  dataset hash model config training loss

Output

training/runs/<timestamp>/



Phase 9 — Incremental Git Sync

Goal: All collaborators stay aligned.

Rules

Raw + cleaned datasets tracked (or via Git LFS) Metadata always committed Scripts are source-of-truth No binary model weights in main repo (optional separate repo)

Workflow

Scraper runs Dataset grows Commit includes:  new raw files updated metadata regenerated training manifests   Others pull → reproduce same state


Phase 10 — Monitoring & Quality Control

Goal: Dataset quality improves over time.

Metrics

Avg story length Language confidence Duplication rate Rejection rate per source

Feedback Loop

Bad sources get deprioritized High-quality sources weighted higher in training


Phase 11 — Scaling the Dataset Safely

As dataset grows:

Rotate old URLs Archive deprecated samples Periodic re-clean with improved scripts Maintain backward compatibility


Final System Property (Important)

At any point, you should be able to say:

“If you clone this repo at commit X and run the pipeline on Linux, you will get the same dataset and training inputs.”

That’s what makes this collaboration-safe and research-grade.


If you want next, I can:

Convert this into GitHub Issues / Milestones Design the CLI interface contract Propose a dataset versioning scheme Or help you decide what not to automate at v1

Just say where you want to go next.
