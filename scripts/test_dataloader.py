import os
import sys
import time
import torch

# Ensure the repository root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.data_loader import DataManager

DATA_PATH = 'dataset/processed/train.txt'
BLOCK_SIZE = 64
BATCH_SIZE = 32
DEVICE = 'cpu'

def test_data_loader():
    # Remove existing cache if any to test first-time tokenization
    bin_path = DATA_PATH + ".bin"
    meta_path = DATA_PATH + "_meta.pkl"
    if os.path.exists(bin_path): os.remove(bin_path)
    if os.path.exists(meta_path): os.remove(meta_path)

    print("--- Test 1: First-time Initialization ---")
    start = time.time()
    dm1 = DataManager(DATA_PATH, BLOCK_SIZE, BATCH_SIZE, DEVICE)
    duration1 = time.time() - start
    print(f"First-time init took {duration1:.4f}s")
    assert os.path.exists(bin_path)
    assert os.path.exists(meta_path)

    print("\n--- Test 2: Cached Initialization ---")
    start = time.time()
    dm2 = DataManager(DATA_PATH, BLOCK_SIZE, BATCH_SIZE, DEVICE)
    duration2 = time.time() - start
    print(f"Cached init took {duration2:.4f}s")
    assert duration2 < duration1
    assert dm1.vocab_size == dm2.vocab_size

    print("\n--- Test 3: Batch Extraction ---")
    x, y = dm2.get_batch()
    print(f"Batch shapes: x={x.shape}, y={y.shape}")
    assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y.shape == (BATCH_SIZE, BLOCK_SIZE)

    # Check if y is shifted x
    # Note: since they are extracted from the same stream, y[i] should be data[ix[i]+1:ix[i]+block_size+1]
    # and x[i] is data[ix[i]:ix[i]+block_size]. So y[i, :-1] == x[i, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1])
    print("Batch integrity verified.")

if __name__ == "__main__":
    test_data_loader()
