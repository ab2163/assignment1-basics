import regex
import re
import os
import heapq
import json
import io
from multiprocessing import Pool
from collections import Counter
from functools import total_ordering
from typing import BinaryIO
import torch
import torch.nn.functional as F
from cs336_basics.lang_model import TransformerLM, softmax

NUM_BYTE_TOKENS = 256

@total_ordering
class RevBytes:
    def __init__(self, b): self.b = b
    def __eq__(self, other): return self.b == other.b
    def __lt__(self, other): return self.b > other.b

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(args):
    input_source, start, end, special_tokens, is_string = args
    
    if is_string:
        chunk_str = input_source[start:end].decode("utf-8", errors="ignore")
    else:
        with open(input_source, "rb") as f:
            f.seek(start)
            chunk_str = f.read(end - start).decode("utf-8", errors="ignore")

    if not special_tokens:
        parts = [chunk_str] # fixes bug where empty special_tokens causes splitting on every character
    else:
        special_pat = "|".join(re.escape(tok) for tok in special_tokens) # pattern for special tokens
        if is_string:
            parts = re.split(f"({special_pat})", chunk_str)  # keeps special tokens
        else:
            parts = re.split(special_pat, chunk_str)  # discards special tokens

    tiktoken_pat = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" # provided by assignment

    if is_string:
        pretok_list = []
        for part in parts:
            if part in special_tokens:
                pretok_list.append(part.encode("utf-8"))  # special token as single unit
            else:
                for pretok in regex.finditer(tiktoken_pat, part):
                    pretok_list.append(pretok.group().encode("utf-8"))
        return pretok_list
    else:
        pretok_freqs = {}
        for part in parts:  # parts == nonspecial_strs in this branch
            for pretok in regex.finditer(tiktoken_pat, part): # get all pretokens
                pretok_bytes = pretok.group().encode("utf-8")
                pretok_freqs[pretok_bytes] = pretok_freqs.get(pretok_bytes, 0) + 1 # add pretoken to frequency map
        return pretok_freqs # return frequency map as bytes -> int

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    num_merges = vocab_size - NUM_BYTE_TOKENS - len(special_tokens)
    special_tokens = sorted(special_tokens, key=len, reverse=True) # largest first order (for overlaps)

    vocab = {idx: bytes([idx]) for idx in range(NUM_BYTE_TOKENS)} # initialise vocab with [0,255]
    for idx, token in enumerate(special_tokens):
        vocab[NUM_BYTE_TOKENS + idx] = token.encode("utf-8")
    
    num_cores = os.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_cores, b"<|endoftext|>") # get boundaries for chunks of text
    args = [(input_path, start, end, special_tokens, False) for start, end in zip(boundaries[:-1], boundaries[1:])]
    num_cores = len(args) # update num_cores in case the chunk finding process returns fewer chunks

    with Pool(processes=num_cores) as pool:
        freq_maps = pool.map(pretokenize_chunk, args)

    # aggregate all the pretoken frequencies
    total_freqs = {}
    for chunk_freqs in freq_maps:
        for pretok, cnt in chunk_freqs.items():
            total_freqs[pretok] = total_freqs.get(pretok, 0) + cnt

    # optionally print out the most frequent pre-tokens
    '''
    for pretok, cnt in sorted(total_freqs.items(), key=lambda x: -x[1])[:20]:
        print(repr(pretok.decode("utf-8", errors="replace")), cnt)
    '''

    pair_counts = {}
    inverted_index = {}  # pair -> set of pre-tokens containing it

    # populate pair_counts and inverted_index
    for pretok, freq in total_freqs.items():
        for i in range(len(pretok) - 1):
            pair = (pretok[i], pretok[i+1]) # note "pair" is a tuple of integers
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if pair not in inverted_index:
                inverted_index[pair] = set() # use set ratther than list since lookup is O(1)
            inverted_index[pair].add(pretok)
    
    # pretok_sequences maps original pre-token -> current token ID sequence
    pretok_sequences = {
        pretok: list(pretok)  # list(b" scared") gives [32, 115, 99, 97, 114, 101, 100]
        for pretok in total_freqs
    }

    # put all (count, pair) tuples in max heap
    # negative numbers for max heap
    heap = [(-count, (RevBytes(vocab[a]), RevBytes(vocab[b]))) for (a, b), count in pair_counts.items()]
    heapq.heapify(heap)

    # to allow conversion from bytes to int
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    merges = {} # describes which numerical token values have merged

    for i in range(num_merges):
        if i % 100 == 0:
            print(f"Merge {i}/{num_merges}", flush=True)

        # use lazy deletion to pop the true maximum:
        while heap:
            neg_count, (rev_a, rev_b) = heapq.heappop(heap)
            bytes_a, bytes_b = rev_a.b, rev_b.b
            pair_int = (reverse_vocab[bytes_a], reverse_vocab[bytes_b])
            if -neg_count == pair_counts.get(pair_int, 0):  # still fresh?
                break
        
        # convert from (bytes, bytes) to (int, int)
        int_a, int_b = pair_int[0], pair_int[1]

        # delete pair entry from pair_counts
        del pair_counts[(int_a, int_b)]

        # create new merged token and add to vocab
        new_bytes = bytes_a + bytes_b # concatenate bytes
        new_id = len(vocab) # smallest unused integer
        vocab[new_id] = new_bytes
        reverse_vocab[new_bytes] = new_id

        # add to merges dict
        merges[(int_a, int_b)] = new_id

        updated_pairs = set()

        # go through all pre-tokens which contain the merged pair
        for pretok in set(inverted_index[(int_a, int_b)]):
            freq = total_freqs[pretok]
            seq = pretok_sequences[pretok]
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == int_a and seq[i+1] == int_b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            pretok_sequences[pretok] = new_seq

            # compare old and new pairs to find what changed
            old_pairs = Counter(zip(seq[:-1], seq[1:]))
            new_pairs = Counter(zip(new_seq[:-1], new_seq[1:]))

            # decrement old pairs
            for pair, count in old_pairs.items():
                pair_counts[pair] = pair_counts.get(pair, 0) - freq * count
                if pair in inverted_index:
                    inverted_index[pair].discard(pretok)
                    if not inverted_index[pair]:
                        del inverted_index[pair]
                updated_pairs.add(pair)

            # increment new pairs
            for pair, count in new_pairs.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + freq * count
                inverted_index.setdefault(pair, set()).add(pretok)
                updated_pairs.add(pair)

        # put the new frequency values in the heap
        for pair in updated_pairs:
            a, b = pair
            heapq.heappush(heap, (-pair_counts[pair], (RevBytes(vocab[a]), RevBytes(vocab[b]))))
        
        assert (int_a, int_b) not in inverted_index, f"pair {(int_a, int_b)} should have been fully removed!"

    '''
    for (int_a, int_b), new_id in merges.items():
        print(f"{vocab[int_a]} + {vocab[int_b]} -> {vocab[new_id]}")
    '''

    # serialise output for further examination
    with open("vocab.json", "w") as f:
        json.dump({k: list(v) for k, v in vocab.items()}, f)
    with open("merges.json", "w") as f:
        json.dump([(list(vocab[a]), list(vocab[b])) for a, b in merges.keys()], f)
    
    return vocab, [(vocab[a], vocab[b]) for (a, b) in merges.keys()]

from typing import Iterable, Iterator

def encode_chunk(args):
    pretokens, merges, reverse_vocab = args
    merge_table = {(a, b): (rank, a + b) for rank, (a, b) in enumerate(merges)}
    
    token_ids = []
    for pretoken in pretokens:
        if pretoken in reverse_vocab:
            token_ids.append(reverse_vocab[pretoken])
            continue
        
        # build doubly linked list
        # each node: [value, prev_idx, next_idx]
        nodes = [[bytes([b]), i-1, i+1] for i, b in enumerate(pretoken)]
        head = 0
        tail = len(nodes) - 1
        nodes[0][1] = -1   # head has no prev
        nodes[tail][2] = -1  # tail has no next
        
        # build initial heap using node indices (not positions)
        heap = []
        i = head
        while nodes[i][2] != -1:
            j = nodes[i][2]
            pair = (nodes[i][0], nodes[j][0])
            if pair in merge_table:
                rank, _ = merge_table[pair]
                heapq.heappush(heap, (rank, i, j))
            i = j
        
        # track deleted nodes
        deleted = set()
        
        while heap:
            rank, i, j = heapq.heappop(heap)
            
            # skip if either node was deleted
            if i in deleted or j in deleted:
                continue
            
            # validate pair still matches (node values may have changed)
            pair = (nodes[i][0], nodes[j][0])
            if pair not in merge_table or merge_table[pair][0] != rank:
                continue
            
            # merge j into i
            _, merged = merge_table[pair]
            nodes[i][0] = merged
            
            # remove j from linked list
            next_j = nodes[j][2]
            nodes[i][2] = next_j
            if next_j != -1:
                nodes[next_j][1] = i
            deleted.add(j)
            
            # check new pair formed on the right (i, next_j)
            if next_j != -1:
                new_pair = (nodes[i][0], nodes[next_j][0])
                if new_pair in merge_table:
                    new_rank, _ = merge_table[new_pair]
                    heapq.heappush(heap, (new_rank, i, next_j))
            
            # check new pair formed on the left (prev_i, i)
            prev_i = nodes[i][1]
            if prev_i != -1:
                new_pair = (nodes[prev_i][0], nodes[i][0])
                if new_pair in merge_table:
                    new_rank, _ = merge_table[new_pair]
                    heapq.heappush(heap, (new_rank, prev_i, i))
        
        # traverse linked list to get final tokens
        i = head
        while i != -1:
            if i not in deleted:
                token_ids.append(reverse_vocab[nodes[i][0]])
            i = nodes[i][2]
    
    return token_ids

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and optionally special tokens.
        
        vocab: dict mapping token ID (int) -> bytes
        merges: list of (bytes, bytes) pairs in order of creation during BPE training
        special_tokens: optional list of special token strings to add to vocabulary
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in vocab.items()}

        # add special tokens to vocabulary
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.reverse_vocab:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = token_bytes
                self.reverse_vocab[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Construct and return a Tokenizer from serialized vocab and merges files.
        
        vocab_filepath: path to vocabulary file
        merges_filepath: path to merges file
        special_tokens: optional list of special token strings
        """
        with open(vocab_filepath) as f:
            vocab = {int(k): bytes(v) for k, v in json.load(f).items()}
        with open(merges_filepath) as f:
            merges = [(bytes(a), bytes(b)) for a, b in json.load(f)]
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        text: input string to encode
        returns: list of integer token IDs
        """
        # fix for test case with empty string
        if not text:
            return []
    
        special_tokens = sorted(self.special_tokens, key=len, reverse=True) # largest first order (for overlaps)
        
        num_cores = os.cpu_count()
        raw_bytes = text.encode("utf-8")
        text_bytes = io.BytesIO(raw_bytes)
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>" # picks longest special token
        boundaries = find_chunk_boundaries(text_bytes, num_cores, split_token) # get boundaries for chunks of text
        args = [(raw_bytes, start, end, special_tokens, True) for start, end in zip(boundaries[:-1], boundaries[1:])]
        num_cores = len(args) # update num_cores in case the chunk finding process returns fewer chunks

        with Pool(processes=num_cores) as pool:
            pretok_lists = pool.map(pretokenize_chunk, args)
            encode_args = [(pretok_list, self.merges, self.reverse_vocab) for pretok_list in pretok_lists]
            encoded_chunks = pool.map(encode_chunk, encode_args)
        
        token_ids = [tid for chunk in encoded_chunks for tid in chunk]
        return token_ids
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, lazily yield token IDs.
        For memory-efficient tokenization of large files.
        
        iterable: iterable of strings (e.g. a file handle)
        yields: integer token IDs
        """
        for text in iterable:
            ids = self.encode(text)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text.
        
        ids: list of integer token IDs
        returns: decoded string
        """
        all_bytes = b"".join(self.vocab.get(id, "\uFFFD".encode("utf-8")) for id in ids)
        return all_bytes.decode("utf-8", errors="replace")

def decode(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token: int | None = None,
    device: str = "mps",
) -> str:
    model.eval()

    # encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, prompt_len)

    # if no end-of-text token provided then see if the <|endoftext|> exists in vocab
    if eot_token is None:
        eot_token = tokenizer.reverse_vocab.get(b"<|endoftext|>", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # truncate to context length if needed
            context = tokens[:, -model.context_length:]

            # forward pass: (1, seq_len, vocab_size)
            logits = model(context)

            # take logits for the last token: (1, vocab_size)
            next_token_logits = logits[:, -1, :]

            # apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # convert to probabilities
            probs = torch.tensor(
                softmax(next_token_logits, dim=-1).tolist(),
                device=device
            )

            # top-p (nucleus) sampling
            if top_p < 1.0:
                # sort probabilities descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # remove tokens once cumulative prob exceeds top_p
                sorted_probs[cumulative_probs - sorted_probs > top_p] = 0.0

                # renormalise
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # sample from filtered distribution
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # append to sequence
            tokens = torch.cat([tokens, next_token], dim=-1)

            # stop if we hit end of text token (if specified)
            if eot_token is not None and next_token.item() == eot_token:
                break

    # decode generated tokens (excluding prompt)
    generated_ids = tokens[0, len(input_ids):].tolist()
    return tokenizer.decode(generated_ids)