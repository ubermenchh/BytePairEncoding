"""
                        -- Byte-Pair Encoding -- 

Identify the most recurring pair of tokens and assign them a single new token.

- Original Paper: https://arxiv.org/pdf/1508.07909
- GPT2 Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
- minBPE repo: https://github.com/karpathy/minbpe?tab=readme-ov-file
- Wiki: https://en.wikipedia.org/wiki/Byte_pair_encoding

""" 

import unicodedata
from typing import Optional, List, Dict, Tuple


def get_stats(ids: List[int], counts: Optional[Dict]=None) -> Dict:
    """
    Given a list of integers, returns a dict of count of consecutive pairs
    
    Parameters:
    - ids: List of tokens 
    - counts[Optional]: Dict of token pair counts

    Returns:
    - Dict of token pair counts
    """

    counts = {} if counts is None else counts 
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    In the list tokens, replace all the consecutive pairs with new token idx. 

    Parameters:
    - ids: Original list of tokens 
    - pair: token-pair that needs to be replaced 
    - idx: replacement token idx
        
    Returns:
    - List of new tokens
    """

    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2                    
        else:
            new_ids.append(ids[i])
            i += 1 

    return new_ids

def replace_control_characters(s: str) -> str:
    """
    This prevents the control characters to be printed out. (Ex, \n)
    Wiki: https://en.wikipedia.org/wiki/Control_character 

    >>> replace_control_characters("\n") # next line
    \u000a
    >>> replace_control_characters("\b") # backspace 
    \u0008
    >>> replace_control_characters("\0") # null 
    \u0000
    """

    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape 
    return "".join(chars)

def render_token(t: bytes) -> str:
    """
    Pretty Printing a token, and escaping the control characters
    """
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s

class Tokenizer:
    def __init__(self):
        self.merges: Dict[int, int] = {} 
        self.pattern: str = ""
        self.special_tokens: Dict[str, int] = {}
        self.vocab: Dict[int, bytes] = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool=False): raise NotImplementedError 
    def encode(self, text: str) -> List[int]: raise NotImplementedError 
    def decode(self, ids: List[int]) -> str: raise NotImplementedError 
    
    def _build_vocab(self):
        """ vocab is derived from merges """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
