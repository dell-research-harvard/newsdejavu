###Use NER models on Huggingface/local path to predict entities in the text

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

import datetime
import numpy as np


sentence = "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian"

sentences=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
           "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                      "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                      "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                      "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                      "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                      "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"

model=AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=True,return_tensors="pt",max_length=256,truncation=True)

token_classifier = pipeline(task="ner" ,
                            model=model, tokenizer=tokenizer,
                            aggregation_strategy="max",ignore_labels = [],
                            batch_size=2)

def handle_punctuation_for_generic_mask(word):
    """If punctuation comes before the word, return it before the mask, ow return it after the mask"""
    
    if word[0] in [".",",","!","?"]:
        return word[0]+" [MASK]"
    elif word[-1] in [".",",","!","?"]:
        return "[MASK]"+word[-1]
    else:
        return "[MASK]"

def handle_punctuation_for_entity_mask(word,entity_group):
    """If punctuation comes before the word, return it before the mask, ow return it after the mask"""
    
    if word[0] in [".",",","!","?"]:
        return word[0]+" "+entity_group
    elif word[-1] in [".",",","!","?"]:
        return entity_group+word[-1]
    else:
        return entity_group
    
    
def replace_words_with_entity_tokens(ner_output_dict,  
                                          desired_labels = ['PER', 'ORG', 'LOC', 'MISC'],
                                          all_masks_same=False
):
    """
     Replace words with entity tokens. Reconstruct the sentence but mask the word with it's entity group (if the entity group is desired) and return the sentence 
    """
    if not all_masks_same:
        new_word_list=[subdict["word"] if subdict["entity_group"] not in desired_labels else handle_punctuation_for_entity_mask(subdict["word"],subdict["entity_group"]) for subdict in ner_output_dict]
    else:
        new_word_list=[subdict["word"] if subdict["entity_group"] not in desired_labels else handle_punctuation_for_generic_mask(subdict["word"]) for subdict in ner_output_dict]

    return " ".join(new_word_list)




print(len(token_classifier(sentences[:2])))
print(replace_words_with_entity_tokens(token_classifier(sentence),all_masks_same=False)),print(replace_words_with_entity_tokens(token_classifier(sentence),all_masks_same=True))