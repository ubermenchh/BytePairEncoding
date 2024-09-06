from .base import Tokenizer, get_stats, merge
from typing import List, Dict, Optional

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: Optional[bool]=False):
        """ trains the tokenizer on a new text """

        assert vocab_size >= 256 
        num_merges = vocab_size - 256 

        # input text processing 
        # text.encode("utf-8") converts the text into raw bytes
        # then they are converted into list of integers between the range of 0...255
        ids = list(text.encode("utf-8")) 

        # iteratively merge the most common pair to new token 
        merges = {} # (int, int) -> int 
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes 
        for i in range(num_merges):
            # get the counts of every token pair 
            stats = get_stats(ids) 
            # pair with the highest count 
            pair = max(stats, key=stats.get) 
            # new token 
            idx = 256 + i 
            # replace all the occurences of `pair` to `idx`(new token)
            ids = merge(ids, pair, idx) 
            # save the merge 
            merges[pair] = idx 
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # for printing 
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurences.")

        # saving the variables 
        self.merges = merges 
        self.vocab = vocab

    def decode(self, ids: List[int]) -> str:
        """ converts a list of integers to Python string """ 
        
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> List[int]:
        """ converts a string to list of integers """

        ids = list(text.encode("utf-8")) # converts the token ids to list of integers between 0..255 
        while len(ids) >= 2:
            stats = get_stats(ids)
            # find the pair with the lowest merge index
            # if there are no merges available, the key will result 
            # in an inf for every single pair, and the min will be just
            # the first pair in the list 
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # if there is nothing else to merge 
            if pair not in self.merges: break
            # otherwise merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
