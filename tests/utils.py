import os
from typing import BinaryIO, Generator
from collections import defaultdict
import regex as re
import multiprocessing as mp

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "split_special_token must be bytes"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def get_chunks(
    input_path: str | os.PathLike,
    num_processes: int = 4,
    split_special_token: str = "<|endoftext|>"
) -> Generator[str, None, None]:
    split_bytes = split_special_token.encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_bytes)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk

def process_chunk(
    chunk: str,
    pattern: str,
    shared_cnt: dict,
) -> None:
    local_cnt = defaultdict(int)
    to_bytes_tuple = lambda word: tuple(bytes([b]) for b in word.encode("utf-8"))

    for m in re.finditer(pattern, chunk):
        token_str = m.group(1) or m.group(2)
        if token_str:
            token_bytes = to_bytes_tuple(token_str)
            local_cnt[token_bytes] += 1

    with mp.Lock():
        for token, cnt in local_cnt.items():
            if token in shared_cnt:
                shared_cnt[token] += cnt
            else:
                shared_cnt[token] = cnt

