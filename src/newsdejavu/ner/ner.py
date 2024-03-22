###Use NER models on Huggingface/local path to predict entities in the text

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import List, Union

import datetime
import numpy as np
from tqdm import tqdm
import torch
from newsdejavu.utils.clean_text import clean_ocr_text



def ner(sentences, model_path: str, batch_size: int = 1,
        max_length: int = 256, torch_device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> List[dict]:
    """
    Processes a list of sentences to identify and tag named entities using a specified model.

    Args:
        sentences (List[str]): A list of sentences to process.
        model_path (str): The file path or model identifier of the pretrained model.
        batch_size (int): The number of sentences to process in a single batch. Defaults to 1.
        max_length (int): The maximum length of the sentences. Sentences longer than this will be truncated. Defaults to 256.
        torch_device (str): The torch device to use for model inference. Defaults to "cuda:0" if CUDA is available, else "cpu".

    Returns:
        List[dict]: A list of dictionaries containing the NER output for each sentence.
    """
    
    
    model=AutoModelForTokenClassification.from_pretrained(model_path)
    print("Loaded ner model")
    tokenizer=AutoTokenizer.from_pretrained(model_path, return_tensors="pt",
                                            max_length=max_length, truncation=True)
    print("Loaded tokenizer")
    
    token_classifier = pipeline(task="ner" ,
                                model=model, tokenizer=tokenizer,
                                aggregation_strategy="max", ignore_labels = [],
                                batch_size=batch_size, device=torch_device)
    
    return token_classifier(sentences)

def handle_punctuation_for_generic_mask(word):
    """If punctuation comes before the word, return it before the mask, ow return it after the mask"""
    
    if word[0] in [".",",","!","?"]:
        return word[0]+" [MASK]"
    elif word[-1] in [".",",","!","?"]:
        return "[MASK]"+word[-1]
    else:
        return "[MASK]"

def handle_punctuation_for_entity_mask(word,entity_group):
    """If punctuation comes before the word, return it before the mask, ow return it after the mask - this is for specific entity masks"""
    
    if word[0] in [".",",","!","?"]:
        return word[0]+" "+entity_group
    elif word[-1] in [".",",","!","?"]:
        return entity_group+word[-1]
    else:
        return entity_group
    
    
def replace_words_with_entity_tokens(ner_output_dict: List[dict],  
                                      desired_labels: List[str] = ['PER', 'ORG', 'LOC', 'MISC'],
                                      all_masks_same: bool = True) -> str:
    """
    Reconstructs sentences from NER output, replacing words with entity group labels or a generic mask based on the specified conditions.

    Args:
        ner_output_dict (List[dict]): NER output for a single sentence.
        desired_labels (List[str]): A list of entity labels to mask. Defaults to ['PER', 'ORG', 'LOC', 'MISC'].
        all_masks_same (bool): Flag indicating whether to use the same mask for all entities or specific entity group labels. Defaults to True.

    Returns:
        str: The reconstructed sentence with words replaced as specified.
    """
    
    if not all_masks_same:
        new_word_list=[subdict["word"] if subdict["entity_group"] not in desired_labels else handle_punctuation_for_entity_mask(subdict["word"],subdict["entity_group"]) for subdict in ner_output_dict]
    else:
        new_word_list=[subdict["word"] if subdict["entity_group"] not in desired_labels else handle_punctuation_for_generic_mask(subdict["word"]) for subdict in ner_output_dict]

    return " ".join(new_word_list)

def mask(ner_output_list: List[List[dict]], desired_labels: List[str] = ['PER', 'ORG', 'LOC', 'MISC'],
                         all_masks_same: bool = True) -> List[str]:
    """
    Processes a list of NER outputs, replacing identified entities in each sentence based on the specified labels and masking preferences.

    Args:
        ner_output_list (List[List[dict]]): A list containing the NER output for multiple sentences.
        desired_labels (List[str]): Entity labels to mask. Defaults to ['PER', 'ORG', 'LOC', 'MISC'].
        all_masks_same (bool): Whether to use a generic mask for all entities or to use specific entity labels. Defaults to True.

    Returns:
        List[str]: A list of sentences with entities replaced according to the specified criteria.
    """
    
    return [replace_words_with_entity_tokens(ner_output,desired_labels,all_masks_same) for ner_output in ner_output_list]



def ner_and_mask(sentences: List[str], model_path: str, batch_size: int = 1, max_length: int = 256,
                         torch_device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
                         labels_to_mask: List[str] = ['PER', 'ORG', 'LOC', 'MISC'], all_masks_same: bool = True,
                         preprocess_for_ocr_errors: bool =False) -> List[str]:
    """
    Obtains masked versions of input sentences by running NER and replacing identified entities based on the specified labels and masking preferences.

    Args:
        sentences (List[str]): The input sentences to process.
        model_path (str): Path or identifier for the pretrained model used for NER.
        batch_size (int): The number of sentences to process in a single batch. Helps manage memory usage and computational load.
        max_length (int): The maximum allowed length for the sentences. Longer sentences are truncated to this length.
        torch_device (str): The device on which the NER model is executed. Can be a CPU or CUDA-enabled GPU device identifier.
        labels_to_mask (List[str]): A list of entity labels (e.g., 'PER' for person, 'ORG' for organization) that should be masked. Other entities will be left unchanged.
        all_masks_same (bool): Indicates whether to use a generic mask for all entities (True) or to mask entities with their specific label (False).

    Returns:
        List[str]: A list of sentences with specified entities masked according to the provided parameters. Each sentence in the list corresponds to an input sentence, transformed based on NER results and masking preferences.

    Example:
        >>> sentences = ["John Doe works at OpenAI in San Francisco."]
        >>> model_path = 'bert-base-cased'
        >>> masked_sentences = ner_and_mask(sentences, model_path)
        >>> print(masked_sentences)
        ["[PER] works at [ORG] in [LOC]."]
    """
    if preprocess_for_ocr_errors:
        sentences=[clean_ocr_text(i,True,["#","/","*","@","~","¢","©","®","°"])[0] for i in sentences]
    ner_output_list = ner(sentences,model_path,batch_size,max_length,torch_device)
    return mask(ner_output_list,labels_to_mask,all_masks_same)


    
