import os
import pickle
import pathlib
import regex as re
import logging
import time
from collections import defaultdict, Counter
from typing import Generator, List, Dict
import multiprocessing as mp
from tqdm import tqdm 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BPE-Trainer")

N_BYTES = 256
MULIPLICATION_FACTOR = 8
CHUNK_SIZE = 1024 * 1024

class BPE:
    def __init__(
        self, 
        input_path: str, 
        special_tokens: list[str] = None
    ):
        self.input_path = pathlib.Path(input_path)
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[tuple[bytes, bytes]] = []
        self.word_counts = Counter()
        self.pair_counts = Counter()
        self.pair_locations = defaultdict(set)
        self.vocab_size = 0
        self.save_model = True

    def train(
        self, 
        tar_vocab_size: int, 
    ):
        logger.info("="*50)
        logger.info(f"Starting BPE model training | Target vocab size: {tar_vocab_size} | Special tokens: {self.special_tokens}")
        logger.info(f"Input file path: {self.input_path.absolute()}")
        logger.info("="*50)
        
        start_time = time.time()
        
        logger.info("[Step 1/3] Initializing base byte vocabulary...")
        self._initialize_vocabulary()
        logger.info(f"Base vocabulary initialized | Initial size: {self.vocab_size} (256 base bytes + {len(self.special_tokens)} special tokens)")

        logger.info("[Step 2/3] Starting text pre-tokenization (multiprocessing)...")
        self._pre_tokenize()
        logger.info(f"Pre-tokenization completed | Total word types: {len(self.word_counts)} | Total word tokens: {sum(self.word_counts.values())}")

        logger.info(f"[Step 3/3] Starting BPE merging (Need to merge {tar_vocab_size - self.vocab_size} times)...")
        self._compute_merges(tar_vocab_size)
        
        total_time = time.time() - start_time
        logger.info("="*50)
        logger.info(f"Training completed! ✅")
        logger.info(f"Total training time: {total_time:.2f}s | Final vocab size: {self.vocab_size} | Merge iterations: {len(self.merges)}")
        logger.info(f"First 5 merge pairs: {self.merges[:5]}")
        logger.info("="*50)

        return self.vocab, self.merges

    def _initialize_vocabulary(self):
        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}
        for i, token in enumerate(self.special_tokens):
            self.vocab[N_BYTES + i] = token.encode('utf-8')
        self.vocab_size = N_BYTES + len(self.special_tokens)

    def _pre_tokenize(self):
        multi_factor = min(mp.cpu_count(), MULIPLICATION_FACTOR)
        logger.info(f"Pre-tokenization processes: {multi_factor} | Chunk size: {CHUNK_SIZE/1024:.1f}MB")
        
        # Get all chunks and show progress bar
        chunks = list(self._chunk_document(self.input_path))
        logger.info(f"Total file chunks: {len(chunks)}")
        
        with mp.Pool(multi_factor) as p:
            # Multiprocessing with progress bar
            with tqdm(total=len(chunks), desc="Pre-tokenization Progress", unit="chunk") as pbar:
                results = p.imap_unordered(self._process_chunk, chunks, chunksize=4)
                for local_counts in results:
                    self.word_counts.update(local_counts)
                    pbar.update(1)

    def _compute_merges(
        self, 
        tar_vocab_size: int
    ):
        multi_factor = min(mp.cpu_count(), MULIPLICATION_FACTOR)
        words = list(self.word_counts.keys())
        logger.info(f"Merging processes: {multi_factor} | Words to process: {len(words)}")

        with mp.Pool(multi_factor) as p:
            batched_words = [words[i::multi_factor] for i in range(multi_factor)]
            with tqdm(total=len(batched_words), desc="Byte Pair Counting Progress", unit="batch") as pbar:
                results = p.imap_unordered(self._process_words, batched_words, chunksize=4)
                for local_pair_counts, local_pair_locations in results:
                    self.pair_counts.update(local_pair_counts)
                    for pair, words_set in local_pair_locations.items():
                        if pair in self.pair_locations:
                            self.pair_locations[pair].update(words_set)
                        else:
                            self.pair_locations[pair] = words_set.copy()
                    pbar.update(1)
        
        logger.info(f"Byte pair counting completed | Initial byte pairs count: {len(self.pair_counts)}")
        
        self.word_to_tokens: Dict[str, List[bytes]] = {
            word: [bytes([b]) for b in word.encode('utf-8')] 
            for word in words
        }
        
        remaining_size = tar_vocab_size - self.vocab_size
        merged = 0
        
        with tqdm(total=remaining_size, desc="BPE Merging Progress", unit="merge") as pbar:
            while merged < remaining_size and self.pair_counts:
                success = self._merge_once()
                if success:
                    merged += 1
                    pbar.update(1)
                    if merged % 1000 == 0:
                        logger.info(f"Merged {merged}/{remaining_size} times | Current vocab size: {self.vocab_size} | Remaining byte pairs: {len(self.pair_counts)}")
                else:
                    logger.warning("No valid byte pairs to merge, terminating early")
                    break

    def _merge_once(self):
        best_pair = None
        max_priority = (-1, b'')

        for pair, count in self.pair_counts.items():
            if count <= 0:
                continue
            
            tie_breaker_key = self._get_tie_breaker_key(pair)
            current_priority = (count, tie_breaker_key)

            if current_priority > max_priority:
                max_priority = current_priority
                best_pair = pair

        if best_pair is None:
            logger.warning("No mergeable byte pairs found")
            return False

        new_token_bytes = self._decode_pair(best_pair, string=False, flattened=True)
        new_id = self.vocab_size
        
        self.vocab[new_id] = new_token_bytes
        self.vocab_size += 1

        affected_words = self.pair_locations.get(best_pair, set()).copy()
        affected_word_count = len(affected_words)

        for word in affected_words:
            word_tokens = self.word_to_tokens[word]
            word_count = self.word_counts[word]

            for i in range(len(word_tokens) - 1):
                old_pair = (word_tokens[i], word_tokens[i + 1])
                self.pair_counts[old_pair] -= word_count
                
                self.pair_locations[old_pair].discard(word)
                if self.pair_counts[old_pair] <= 0:
                    del self.pair_counts[old_pair]
                    if old_pair in self.pair_locations:
                        del self.pair_locations[old_pair]

            i = 0
            new_tokens = []
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token_bytes)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            
            self.word_to_tokens[word] = new_tokens

            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                self.pair_counts[new_pair] += word_count
                self.pair_locations[new_pair].add(word)

        self.merges.append(best_pair)
        
        if best_pair in self.pair_counts:
            del self.pair_counts[best_pair]
        if best_pair in self.pair_locations:
            del self.pair_locations[best_pair]
            
        return True 

    def _chunk_document(
        self, 
        input_path: str, 
        chunk_size: int = CHUNK_SIZE, 
        special_token: str = "<|endoftext|>"
    ) -> Generator[str, None, None]:
        leftover = ""
        token_len = len(special_token)
        with open(input_path, 'r', encoding='utf-8') as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                block = leftover + block
                leftover = ""
                last_eot_index = block.rfind(special_token)
                if last_eot_index == -1:
                    leftover = block
                else:
                    yield block[:last_eot_index + token_len]
                    leftover = block[last_eot_index + token_len:]
        if leftover:
            yield leftover

    def _process_chunk(
        self, 
        chunk: str
    ) -> Counter:
        sub_chunks = re.split(self.special_pattern, chunk)
        sub_chunks = [re.finditer(self.pattern, sub_chunk) for sub_chunk in sub_chunks]
        sub_chunks = [m.group() for sub_chunk in sub_chunks for m in sub_chunk]
        counts = Counter(sub_chunks)
        return counts
    
    def _process_words(
        self, 
        words: List[str]
    ) -> tuple[Counter, defaultdict[tuple[bytes, bytes], set[str]]]:
        pair_counts = Counter()
        pair_locations = defaultdict(set)

        for word in words:
            word_bytes = word.encode('utf-8')
            word_count = self.word_counts[word]
            for i in range(len(word_bytes) - 1):
                b1 = word_bytes[i]
                b2 = word_bytes[i+1]
                pair = (bytes([b1]), bytes([b2]))
                pair_counts[pair] += word_count
                pair_locations[pair].add(word)

        return pair_counts, pair_locations
    
    def _decode_pair(self, pair: tuple[bytes, bytes], string: bool = True, flattened: bool = False):
        token1_bytes = self.vocab.get(pair[0], pair[0]) if isinstance(pair[0], int) else pair[0]
        token2_bytes = self.vocab.get(pair[1], pair[1]) if isinstance(pair[1], int) else pair[1]

        byte_tuple = (token1_bytes, token2_bytes)

        if string:
            return b'\x00'.join(byte_tuple)

        if flattened:
            return b''.join(byte_tuple)
        
        return byte_tuple

    def _get_tie_breaker_key(self, pair: tuple[bytes, bytes]):
        decoded_string = self._decode_pair(pair, string=True)
        return decoded_string


if __name__ == "__main__":
    DATA_DIR = '/root/autodl-fs/cs336_data/assignment1'
    INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")

    TOKENIZER_DIR = '/root/autodl-tmp/cs336/assignment1'
    VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
    MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    bpe = BPE(
        input_path=INPUT_PATH,
        special_tokens=special_tokens
    )
    vocab, merges = bpe.train(vocab_size)

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)
    
    logger.info(f"Vocabulary saved to: {VOCAB_PATH}")
    logger.info(f"Merges saved to: {MERGES_PATH}")