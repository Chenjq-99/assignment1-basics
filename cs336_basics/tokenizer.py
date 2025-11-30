import regex as re
import pickle
from typing import Generator, Dict, List, Tuple, Iterator, Iterable
from collections import defaultdict
import multiprocessing as mp

MULIPLICATION_FACTOR = 8
CHUNK_SIZE = 1024 * 1024

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.id_to_token = vocab
        self.token_to_id: Dict[bytes, int] = {token: idx for idx, token in vocab.items()}
        self.merges = merges
        self.merges_to_id: Dict[Tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.sorted_special_tokens = sorted(
            self.special_tokens, 
            key=lambda x: len(x), 
            reverse=True
        )
        self.special_tokens_pattern = "|".join(
            re.escape(token) for token in self.sorted_special_tokens
        ) if self.sorted_special_tokens else ""
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_file(
        cls, 
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        try:
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            with open(merges_path, "rb") as f:
                merges = pickle.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Model file not found: {e}")
        except pickle.UnpicklingError as e:
            raise ValueError(f"Invalid model file format: {e}")
        
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens
        )
    
    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        
        word_list: List[str] = []
        if self.special_tokens_pattern:
            chunks = re.split(f"({self.special_tokens_pattern})", text)
            chunks = [chunk for chunk in chunks if chunk]
        else:
            chunks = [text]
                
        for chunk in chunks:
            if chunk in self.special_tokens:
                word_list.append(chunk)
            else:
                word_list.extend(match.group() for match in re.finditer(self.pattern, chunk))
        
        encoded_ids: List[int] = []
        for word in word_list:
            if word in self.special_tokens:
                word_bytes = word.encode('utf-8')
                if word_bytes not in self.token_to_id:
                    raise KeyError(f"Special token '{word}' (bytes: {word_bytes}) not found in vocab")
                encoded_ids.append(self.token_to_id[word_bytes])
            else:
                tokens = self._get_tokens_from_merges(word)
                for token in tokens:
                    if token not in self.token_to_id:
                        raise KeyError(f"Token {token} (from word '{word}') not found in vocab")
                    encoded_ids.append(self.token_to_id[token])
        
        return encoded_ids
    
    def encode_iterable(
        self, 
        iterable: Iterable[str], 
        memory_efficient: bool = True
    ) -> Iterator[int]:
        if memory_efficient:
            chunks = self._chunk_iterable_stream(iterable, chunk_size=1024)
            for chunk in chunks:
                yield from self.encode(chunk)
        else:
            all_tokens = []
            multi_factor = min(mp.cpu_count(), MULIPLICATION_FACTOR)
            with mp.Pool(multi_factor) as p:
                while True:
                    batched_words = []
                    for _ in range(multi_factor):
                        try:
                            chunk = iterable.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            batched_words.append(chunk)
                        except StopIteration:
                            break
                
                    if not batched_words:
                        break
                    results = p.map(self.encode, batched_words)

                    for tokens in results:
                        all_tokens.extend(tokens)
            yield from all_tokens

    def decode(self, ids: list[int]) -> str:
        byte_list = b""
        for token_id in ids:
            if token_id in self.id_to_token:
                byte_list += self.id_to_token[token_id]
            else:
                byte_list += b"\xef\xbf\xbd"
        return byte_list.decode('utf-8', errors='replace')
    
    def _get_tokens_from_merges(self, word: str) -> list[bytes]:
        word_bytes = word.encode('utf-8')
        bytes_list: List[bytes] = [bytes([b]) for b in word_bytes]
        
        while len(bytes_list) > 1:
            candidate_id_pos: Dict[int, int] = defaultdict(int)
            for i in range(len(bytes_list) - 1):
                pair = (bytes_list[i], bytes_list[i+1])
                if pair in self.merges_to_id:
                    candidate_id_pos[self.merges_to_id[pair]] = i
            
            if not candidate_id_pos:
                break
            
            merge_id = min(candidate_id_pos.keys())
            merge_pos = candidate_id_pos[merge_id]
            
            merged_token = bytes_list[merge_pos] + bytes_list[merge_pos + 1]
            bytes_list = bytes_list[:merge_pos] + [merged_token] + bytes_list[merge_pos+2:]
        
        return bytes_list
    
    def _chunk_iterable_stream(
        self, 
        iterable: Iterable[str], 
        chunk_size: int = CHUNK_SIZE, 
        special_token: str = "<|endoftext|>"
    ) -> Generator[str, None, None]:
        leftover = ""
        token_len = len(special_token)
    
        while True:
            block = iterable.read(chunk_size)
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