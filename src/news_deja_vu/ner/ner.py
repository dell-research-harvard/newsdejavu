###Use NER models on Huggingface/local path to predict entities in the text

from transformers import pipeline
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
###Note that we want the output to be in the format (tuple) words,entity_labels


                            
                            
##Now, we want to parse the output to get the words and entity labels
def parse_output_return_labels(ner_pipeline_output):
    words=[]
    labels=[]
    for i in range(len(ner_pipeline_output)):
        words.append(ner_pipeline_output[i]['word'])
        labels.append(ner_pipeline_output[i]['entity_group'])
    return words,labels



model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"
token_classifier = pipeline(task="ner" ,
                            model=model_path,
                            aggregation_strategy="max",ignore_labels = [],
                            batch_size=2)

def handle_punctuation_for_mask(word):
    
    
def replace_words_with_entity_tokens(ner_output_dict,  
                                          desired_labels = ['PER', 'ORG', 'LOC', 'MISC'],
                                          all_masks_same=False
):
    """
     Replace words with entity tokens. Reconstruct the sentence but mask the word with it's entity group (if the entity group is desired) and return the sentence 
    """
    if not all_masks_same:
        new_word_list=[subdict["word"] if subdict["entity_group"] not in desired_labels else subdict["entity_group"] for subdict in ner_output_dict]
    else:
        new_word_list=[subdict["word"] if subdict["entity_group"] not in desired_labels else "[MASK]" for subdict in ner_output_dict]

    return " ".join(new_word_list)

print((token_classifier(sentences[:1])))
# print(replace_words_with_entity_tokens(token_classifier(sentence),all_masks_same=False)),print(replace_words_with_entity_tokens(token_classifier(sentence),all_masks_same=True))