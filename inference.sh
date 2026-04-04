#!/bin/bash
# Batch inference script as requested in PR feedback.
PROMPT=$(cat input.txt)
PYTHONPATH=. python3 scripts/generate.py --config training/configs/small.yaml --checkpoint_path out/bolt_micro_train/model_final.pt --max_new_tokens 100 --start_text "$PROMPT" > output.txt
