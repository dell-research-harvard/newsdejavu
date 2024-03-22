from typing import List, Dict, Optional, Union
from transformers import PreTrainedModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle

from newsdejavu.utils.wrangling import find_mask_token, find_sep_token


def embed(corpus: Union[List,List[Dict[str, str]]], model: str, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create embeddings from masked sentences in a given corpus using a specified model.
    
    This function processes a list of dictionaries, each representing an article with a masked sentence,
    to generate embeddings using a specified model. If the model is not directly available, it attempts
    to load it from specified repositories in a fallback manner. The function also supports saving the
    generated embeddings to a specified path.

    Args:
        corpus (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains at least
            a "masked_sentence" key representing the text to embed.
        model (str): The model identifier used for embedding. This should be a valid Hugging Face model path.
            If the model is not found, the function attempts to fetch it from 'dell-research-harvard' or
            'sentence-transformers' as fallback repositories.
        save_path (Optional[str]): The file path where the embeddings should be saved. If not provided,
            embeddings are not saved to disk. Default is None.

    Returns:
        np.ndarray: An array of embeddings, one for each masked sentence in the corpus.

    Raises:
        Exception: If the model cannot be loaded from any of the attempted paths.

    Example:
        >>> corpus = [{'masked_sentence': 'Today is a [MASK] day.'}, {'masked_sentence': 'I enjoy [MASK] coffee in the morning.'}]
        >>> model = 'all-MiniLM-L6-v2'
        >>> embeddings = embed(corpus, model)
        >>> print(embeddings.shape)
        (2, 384)  # Assuming the used model generates embeddings of size 384.

    Note:
        - The function normalizes embeddings to unit length.
        - It replaces '[MASK]' and '[SEP]' tokens in the corpus with the appropriate tokens for the specified model.
    """

    ##if no organization is provided in the model repo, try dell-research-harvard > sentence-transformers > throw error
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(f"dell-research-harvard/{model}")
        except:
            tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model}")
                    

    mask_tok = find_mask_token(tokenizer)
    sep_tok = find_sep_token(tokenizer)
    
    if isinstance(corpus[0], str):
        corpus_text_list = corpus
    else:
        corpus_text_list=[corpus_subdict["masked_sentence"] for corpus_subdict in corpus] 
    data = []
    for text in corpus_text_list:
        # Correct [MASK] token for tokenizer
        text = text.replace('[MASK]', mask_tok)
        text = text.replace('[SEP]', sep_tok)
        data.append(text)
    

    print(f'{len(data)} articles in corpus')

    print("embedding corpus ...")
    sentence_model=SentenceTransformer(model)
    corpus_embeddings = sentence_model.encode(data, show_progress_bar=True, batch_size=512)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}.pkl', 'wb') as f:
            pickle.dump(corpus_embeddings, f)
    
    return corpus_embeddings


if __name__ == '__main__':

    corpus_dict = [{"id": 1, "masked_sentence": "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian"},
                   {"id": 2, "masked_sentence": "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian"}]

    # Embed
    test_ob=embed(
        corpus_dict,
        model = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/same_story_model",
        save_path=None
    )
    
    print(test_ob.shape)