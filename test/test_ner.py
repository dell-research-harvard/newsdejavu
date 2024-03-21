"""Unit tests for named entity recognition functions."""


import os
import pytest

from newsdejavu import ner, mask, ner_and_mask


SENTENCES=["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
        "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                    "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                    "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                    "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                    "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
                    "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

MODEL="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"


def test_ner(SENTENCES,MODEL):
    ner_output=ner(SENTENCES,MODEL,batch_size=10)
    assert len(ner_output)==len(SENTENCES)
    
def test_mask(SENTENCES,MODEL):
    ner_output=ner(SENTENCES,MODEL,batch_size=10)
    masked_sentences=mask(ner_output)
    assert len(masked_sentences)==len(SENTENCES)
    
def test_ner_and_mask(SENTENCES,MODEL):
    masked_sentences=ner_and_mask(SENTENCES,MODEL,batch_size=10)
    assert len(masked_sentences)==len(SENTENCES)

def test_ner_and_mask_clean_ocr(SENTENCES,MODEL):
    masked_sentences=ner_and_mask(SENTENCES,MODEL,batch_size=10,preprocess_for_ocr_errors=True)
    assert len(masked_sentences)==len(SENTENCES)