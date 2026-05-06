import os
import time
import random
import numpy as np
import cProfile
from bpe import Tokenizer
from bpe import find_chunk_boundaries

def load_documents(filepath, n=None):
    """Load documents lazily, stopping once we have n documents."""
    docs = []
    current_doc = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "<|endoftext|>" in line:
                parts = line.split("<|endoftext|>")
                current_doc.append(parts[0])
                doc = "".join(current_doc).strip()
                if doc:
                    docs.append(doc)
                current_doc = [parts[-1]]
                if n is not None and len(docs) >= n:
                    break
            else:
                current_doc.append(line)
    
    return docs

def compression_ratio(tokenizer, docs):
    """Compute bytes/token compression ratio."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    return total_bytes / total_tokens

def experiment_a(ts_tokenizer, owt_tokenizer, n=10):
    """Compression ratio for each tokenizer on its own dataset."""
    print("=== Experiment A ===")
    ts_docs = load_documents("../data/TinyStoriesV2-GPT4-train.txt", n)
    owt_docs = load_documents("../data/owt_train.txt", n)
    
    ts_ratio = compression_ratio(ts_tokenizer, ts_docs)
    owt_ratio = compression_ratio(owt_tokenizer, owt_docs)
    
    print(f"TinyStories tokenizer (10K vocab) on TinyStories: {ts_ratio:.2f} bytes/token")
    print(f"OpenWebText tokenizer (32K vocab) on OpenWebText: {owt_ratio:.2f} bytes/token")
    return ts_ratio, owt_ratio

def experiment_b(ts_tokenizer, owt_tokenizer, n=10):
    """What happens when you tokenize OWT with TinyStories tokenizer?"""
    print("=== Experiment B ===")
    owt_docs = load_documents("../data/owt_train.txt", n)
    
    ts_ratio = compression_ratio(ts_tokenizer, owt_docs)
    owt_ratio = compression_ratio(owt_tokenizer, owt_docs)
    
    print(f"TinyStories tokenizer on OpenWebText: {ts_ratio:.2f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.2f} bytes/token")
    
    # qualitative example
    example = owt_docs[0][:200]
    ts_tokens = [ts_tokenizer.decode([id]) for id in ts_tokenizer.encode(example)]
    owt_tokens = [owt_tokenizer.decode([id]) for id in owt_tokenizer.encode(example)]
    print(f"\nExample text:\n{example}")
    print(f"\nTinyStories tokenizer:\n{ts_tokens}")
    print(f"\nOpenWebText tokenizer:\n{owt_tokens}")
    return ts_ratio, owt_ratio

def experiment_c(tokenizer, filepath, n=10):
    """Estimate tokenizer throughput."""
    print("=== Experiment C ===")
    docs = load_documents(filepath, n)
    text = " ".join(docs)
    total_bytes = len(text.encode("utf-8"))
    
    start = time.time()
    tokenizer.encode(text)
    elapsed = time.time() - start
    
    throughput = total_bytes / elapsed
    pile_size = 825 * 1024 ** 3  # 825GB in bytes
    estimated_time = pile_size / throughput
    
    print(f"Throughput: {throughput / 1024**2:.2f} MB/s ({throughput:.0f} bytes/s)")
    print(f"Estimated time for Pile (825GB): {estimated_time / 3600:.1f} hours")
    return throughput

def document_generator(filepath):
    """Lazily yield documents from a txt file, adding back the special token."""
    with open(filepath, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            buffer += line
            if "<|endoftext|>" in buffer:
                parts = buffer.split("<|endoftext|>")
                for part in parts[:-1]:
                    if part.strip():
                        yield part.strip() + "<|endoftext|>"
                buffer = parts[-1]
        if buffer.strip():
            yield buffer.strip()

def document_batch_generator(filepath, batch_size=500):
    """Lazily yield batches of documents joined together as a single string."""
    batch = []
    for doc in document_generator(filepath):
        batch.append(doc)
        if len(batch) >= batch_size:
            yield "<|endoftext|>".join(batch)
            batch = []
    if batch:  # flush remaining documents
        yield "<|endoftext|>".join(batch)

def experiment_d(tokenizer, filepath, output_path, num_chunks=1000):
    print(f"=== Experiment D: encoding {filepath} ===")
    
    split_token = "<|endoftext|>".encode("utf-8")
    
    # find chunk boundaries
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)
    
    print(f"Processing {len(boundaries)-1} chunks...")
    
    total_tokens = 0
    start = time.time()
    
    with open(filepath, "rb") as text_f:
        with open(output_path, "wb") as out_f:
            for i, (start_pos, end_pos) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                # read and decode chunk
                text_f.seek(start_pos)
                chunk_text = text_f.read(end_pos - start_pos).decode("utf-8", errors="ignore")
                
                # encode chunk
                ids = tokenizer.encode(chunk_text)
                
                # write to file
                out_f.write(np.array(ids, dtype=np.uint16).tobytes())
                total_tokens += len(ids)
                
                elapsed = time.time() - start
                print(f"Chunk {i+1}/{len(boundaries)-1} | Tokens: {total_tokens:,} | Speed: {total_tokens/elapsed:.0f} tokens/s | Elapsed: {elapsed:.0f}s", end="\r")
    
    elapsed = time.time() - start
    print(f"\nDone! Saved {total_tokens:,} tokens to {output_path} in {elapsed:.0f}s ({total_tokens/elapsed:.0f} tokens/s)")
    return total_tokens

# --- run all experiments ---
if __name__ == "__main__":
    ts_tokenizer = Tokenizer.from_files("vocab_ts.json", "merges_ts.json", ["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_files("vocab_owt.json", "merges_owt.json", ["<|endoftext|>"])
    #experiment_a(ts_tokenizer, owt_tokenizer)
    #experiment_b(ts_tokenizer, owt_tokenizer)
    #experiment_c(ts_tokenizer, "../data/TinyStoriesV2-GPT4-train.txt")
    #experiment_d(ts_tokenizer, "../data/TinyStoriesV2-GPT4-train.txt", "ts_train.npy")
    experiment_d(ts_tokenizer, "../data/TinyStoriesV2-GPT4-valid.txt", "ts_valid.npy")