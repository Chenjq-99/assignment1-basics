import os
import sys
import pickle
import pathlib
from tests.adapters import run_train_bpe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DATA_DIR = '/root/autodl-fs/cs336_data/assignment1'
INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")

TOKENIZER_DIR = '/root/autodl-tmp/cs336/assignment1'
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

vocab, merges = run_train_bpe(
    input_path=INPUT_PATH,
    vocab_size=vocab_size,
    special_tokens=special_tokens
)

os.makedirs(TOKENIZER_DIR, exist_ok=True)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, "wb") as f:
    pickle.dump(merges, f)

longest_token = max(vocab.values(), key=len)
print("最长token:", longest_token, "长度:", len(longest_token))
