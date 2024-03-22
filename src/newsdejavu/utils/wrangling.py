
import os
import pickle
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


def find_sep_token(tokenizer):

    """
    Returns sep token for given tokenizer
    """

    if 'eos_token' in tokenizer.special_tokens_map:
        sep = " " + tokenizer.special_tokens_map['eos_token'] + " " + tokenizer.special_tokens_map['sep_token'] + " "
    else:
        sep = " " + tokenizer.special_tokens_map['sep_token'] + " "

    return sep


def find_mask_token(tokenizer):
    """
    Returns mask token for given tokenizer

    """
    mask_tok = tokenizer.special_tokens_map['mask_token']
    
    return mask_tok
    
    
def featurize_text(byline, text, sep, headline=None):

    if headline == "nan":
        headline = " "
    if byline == "nan":
        byline = " "
    if text == "nan":
        text = " "

    if headline:
        new_text = headline + byline + sep + text

    else:
        new_text = byline + sep + text

    return new_text
