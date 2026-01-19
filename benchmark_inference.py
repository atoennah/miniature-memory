
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_inference_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use a dummy vocab_size for benchmark purposes
    # In a real scenario, this would come from the tokenizer
    vocab_size = 512

    model_config = config_data['model']
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    )

    model = GPT(config)
    model.eval() # Set to eval mode

    # --- Benchmark Settings ---
    num_runs = 5
    max_new_tokens = 100
    prompt = torch.zeros((1, 1), dtype=torch.long) # A single <start> token

    throughputs = []
    print("--- Starting Inference Benchmark ---")
    print(f"Configuration: {config_path}")
    print(f"Number of runs: {num_runs}")
    print(f"Tokens to generate per run: {max_new_tokens}")
    print("------------------------------------")


    for i in range(num_runs):
        start_time = time.time()
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
        end_time = time.time()

        duration = end_time - start_time
        throughput = max_new_tokens / duration
        throughputs.append(throughput)
        print(f"Run {i+1}/{num_runs}: {throughput:.2f} tokens/sec")

    # --- Calculate and Report Average ---
    avg_throughput = sum(throughputs) / len(throughputs)

    print("\n--- Benchmark Results ---")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"-----------------------")
    return avg_throughput

if __name__ == "__main__":
    run_inference_benchmark()
