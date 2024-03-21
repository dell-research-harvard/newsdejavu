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

def replace_words_with_entity_tokens(
      word_list,
      token_list,
      merge_consecutive_tokens=True,
      return_sentence=True,
      all_masks_same=False,
      desired_labels = ['PER', 'ORG', 'LOC', 'MISC'],
      ):
    """
    Replace words with entity tokens. Also capture all entities
    """

    new_word_list = []
    for word,token in zip(word_list,token_list):
        if token[0] == 'B':
            new_word_list.append(token[2:] + "_START")
        elif token[0] == 'I':
            ##If the previous word is not _START, then add _CONTINUE, otherwise add _START. Ignore any punctuation between 
            if len(new_word_list)>0 and (("_START" in new_word_list[-1] or "_CONTINUE" in new_word_list[-1]) or
                                          (new_word_list[-1] in [".",",","!","?",";"]  and (len(new_word_list)>1 and ("_START" in new_word_list[-2] or "_CONTINUE" in new_word_list[-2])))):
                new_word_list.append(token[2:] + "_CONTINUE")
            else:
                new_word_list.append(token[2:] + "_START")
        ##If word is a punctuation and the previous new_word_list item is a _START or _CONTINUE, then add _CONTINUE
        elif word in [".",",","!","?",";"] and len(new_word_list)>0 and ("_START" in new_word_list[-1] or "_CONTINUE" in new_word_list[-1]):
            prev_word = new_word_list[-1].split()
            new_word_list.append(prev_word[0] + "_CONTINUE")
                             
        else:
            new_word_list.append(word)


    if merge_consecutive_tokens:
      ###Replace all _CONTINUE with _START and then remove consecutive _START tokens
      new_word_list = [word for word in new_word_list if "_CONTINUE" not in word]
      new_word_list = [word for word in new_word_list if "_START" not in word or "_START" not in new_word_list[new_word_list.index(word)-1]]
      ##Remove the _START 
      new_word_list = [word.replace("_START","") for word in new_word_list]

    ##Merge consecutive tokens of the same type
    for i in range(len(new_word_list)-1):
        if i-1>0:
          if new_word_list[i-1] == new_word_list[i]:
              ##pop it
              new_word_list[i]=""
    
    new_word_list = [word for word in new_word_list if word!=""]

    if all_masks_same:
        masked_new_word_list=[]
        ##Replace by [MASK] if token contains any of the desired_labels
        for word in new_word_list:
            if any([label in word for label in desired_labels]):
                masked_new_word_list.append("[MASK]")
            else:
                masked_new_word_list.append(word)
        new_word_list = masked_new_word_list
        
    if return_sentence:
        out_sentence= " ".join(new_word_list)
        ##remove space before punctuation
        out_sentence = out_sentence.replace(" .",".").replace(" ,",",").replace(" !","!").replace(" ?","?").replace(" : ",": ").replace(" ; ","; ").replace(" '","'")
        return out_sentence
    
    else:
        return new_word_list

model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"
token_classifier = pipeline(task="ner" ,
                            model=model_path,
                            aggregation_strategy="max",ignore_labels = [],
                            batch_size=2)

print(((token_classifier(sentence))))