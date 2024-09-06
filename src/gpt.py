from functools import wraps
from .base import Tokenizer, get_stats, merge 
from typing import Optional, List, Dict
import regex as re

GPT2_SPLIT_PATTERN = "|".join([
    r"""'(?:[sdmt]|ll|ve|re)""",    # apostrophe followed by d, s, m, t, ll, ve, re 
    r""" ?\p{L}+""",    # matches one or more letters optionally preceded by a space
    r""" ?\p{N}+""",    # matches one or more numeric chars optionally preceded by a space
    r""" ?[^\s\p{L}\p{N}]+""",  # matches one or more non-numeric or non-letter chars 
    r"""\s+(?!\S)""",   # matches one or more whitespace chars that are not followed by non-whitespace chars
    r"""\s+"""  # matches one or more whitespace anywhere
])
GPT4_SPLIT_PATTERN = "|".join([
    r"""'(?:[sdmt]|ll|ve|re)""",
    r"""[^\r\n\p{L}\p{N}]?+\p{L}+""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*""",
    r"""\s*[\r\n]""",
    r"""\s+(?!\S)""",
    r"""\s+"""
])

class GPTTokenizer(Tokenizer):
    def __init__(self, pattern: Optional[str]=None):
        super().__init__()
        if pattern is None or pattern == "gpt4":
            self.pattern = GPT4_SPLIT_PATTERN 
        elif pattern == "gpt2":
            self.pattern = GPT2_SPLIT_PATTERN 
        else:
            raise ValueError("Invalid pattern type.")
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int, verbose: Optional[bool]=None):
        """ trains the tokenizer on a given text """
        assert vocab_size >= 256 
        num_merges = vocab_size - 256

        # split the text into text chunks 
        chunks = re.findall(self.compiled_pattern, text)
        # input text processing
        ids = [list(ch.encode("utf-8")) for ch in chunks]

        # iteratively merge the most common pairs to create new tokens 
        merges = {} # (int, int) -> int 
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes 
        for i in range(num_merges):
            # count the number of times each token pair appears 
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            # pair with highest count 
            pair = max(stats, key=stats.get)
            # new token 
            idx = 256 + i
            # replace the top pair with new token 
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge 
            merges[pair] = idx 
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # for printing 
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurences.")

        # saving class variables 
        self.merges = merges 
        self.vocab = vocab 
    
    def register_special_tokens(self, special_tokens: Dict[str, int]):
        self.special_tokens = special_tokens 
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def decode(self, ids: List[int]) -> str:
        """ converts a list of integers to Python string """
        
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid Token id: {idx}")
        
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes: List[bytes]) -> List[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids) 
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids 

    def encode_ordinary(self, text: str) -> List[int]:
        """ encodes a string into list of integers ignoring the special_tokens """

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunks_ids = self._encode_chunk(chunk.encode("utf-8"))
            ids.extend(chunks_ids)
        return ids 

    def encode(self, text: str, allowed_special: Optional[str]="none_raise") -> List[int]:
        """ 
        Encodes a string into list of integers (including special_tokens) 
        allowed_special: "all" | "none" | "none_raise" | custom set of special_tokens
        if `none_raise`, then an error is raised if any special token is found in the text 
        """
        special = None 
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        # if no special_tokens, use the ordinary encoding 
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
