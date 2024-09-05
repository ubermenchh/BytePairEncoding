"""
                        -- Byte-Pair Encoding -- 

- Identify the most recurring pair of tokens and assign them a single new token.
- Original Paper: https://arxiv.org/pdf/1508.07909
- GPT2 Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
- minBPE repo: https://github.com/karpathy/minbpe?tab=readme-ov-file
- Wiki: https://en.wikipedia.org/wiki/Byte_pair_encoding

""" 

import unicodedata
from typing import Optional, List, Dict, Tuple

with open("input.txt", "r") as f:
    raw_text = f.read()

sample = raw_text[:500]
tokens = list(map(int, sample.encode("utf-8")))

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
    while (i < len(ids)):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2                    
        else:
            new_ids.append(ids[i])
            i += 1 

    return new_ids

token_pair = get_stats(tokens)
top_pair = max(token_pair, key=token_pair.get)

new_tokens = merge(tokens, top_pair, 256)
print(new_tokens)


