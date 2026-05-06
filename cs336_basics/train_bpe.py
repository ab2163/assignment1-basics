from bpe import train_bpe
import cProfile
import json
from memory_profiler import memory_usage
import time

def run():
    cProfile.run("train_bpe('../data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])", 'output.prof')

if __name__ == '__main__':
    start = time.time()
    mem = memory_usage(run)
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")
    print(f"Peak memory: {max(mem) / 1024:.2f} GB")

    with open("vocab.json") as f:
        vocab = {int(k): bytes(v) for k, v in json.load(f).items()}
    longest_id, longest_bytes = max(vocab.items(), key=lambda x: len(x[1]))
    print(f"Token ID: {longest_id}")
    print(f"Token: {longest_bytes}")
    print(f"Length: {len(longest_bytes)} bytes")
    print(f"As string: {longest_bytes.decode('utf-8', errors='replace')}")