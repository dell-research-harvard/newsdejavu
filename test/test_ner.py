"""Unit tests for named entity recognition functions."""


import os
import pytest

from newsdejavu import ner, mask, ner_and_mask


def test_ner():
    

    sentences=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

    model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"

    ner_output=ner(sentences,model,batch_size=10)
    assert len(ner_output)==len(sentences)
    
def test_mask():
 
    sentences=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

    model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"

    ner_output=ner(sentences,model,batch_size=10)
    masked_sentences=mask(ner_output)
    assert len(masked_sentences)==len(sentences)
    
def test_ner_and_mask():

    sentences=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

    model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"

    masked_sentences=ner_and_mask(sentences,model,batch_size=10)
    assert len(masked_sentences)==len(sentences)

    
def test_ner_and_mask_diff_masks():

    sentences=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

    model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"

    masked_sentences=ner_and_mask(sentences,model,batch_size=10,all_masks_same=False)
    assert len(masked_sentences)==len(sentences)

def test_ner_and_mask_clean_ocr():

    sentences=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

    model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"

    masked_sentences=ner_and_mask(sentences,model,batch_size=10,preprocess_for_ocr_errors=True)
    assert len(masked_sentences)==len(sentences)