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
        labels.append(ner_pipeline_output[i]['entity'])
    return words,labels


model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"
token_classifier = pipeline(task="ner" ,
                            model=model_path,
                            aggregation_strategy="max",
                            batch_size=2)

print((token_classifier(sentence)))