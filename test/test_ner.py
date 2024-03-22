"""Unit tests for named entity recognition functions."""


import os
import pytest
import shutil
import json

from newsdejavu import ner, mask, ner_and_mask
from newsdejavu import download

json.encoder.FLOAT_REPR = lambda o: format(o, '.7f' )

@pytest.fixture
def sample_sentences():
    return ["I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",
            "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian",]

model = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"
batch_size = 10

class TestSimpleNER:
    
    
    def test_ner(self, sample_sentences):
        ner_output=ner(sample_sentences, model, batch_size = batch_size)
        print(ner_output)
        with open('data/ner_test_output/ner_output.json', 'w') as f:
            json.dump(ner_output, f, indent=4)
        assert len(ner_output) == len(sample_sentences)
    
    def test_mask(self, sample_sentences):
        ner_output = ner(sample_sentences, model, batch_size = batch_size)
        masked_sentences = mask(ner_output)
        assert len(masked_sentences) == len(sample_sentences)

    def test_ner_and_mask(self, sample_sentences):
        masked_sentences = ner_and_mask(sample_sentences, model, batch_size = batch_size)
        assert len(masked_sentences) == len(sample_sentences)    

    def test_ner_and_mask_diff_masks(self, sample_sentences):
        masked_sentences = ner_and_mask(sample_sentences, model, batch_size = batch_size, all_masks_same = False)
        assert len(masked_sentences) == len(sample_sentences)
    
    def test_ner_and_mask_clean_ocr(self, sample_sentences):
        masked_sentences = ner_and_mask(sample_sentences, model, batch_size = batch_size, preprocess_for_ocr_errors = True)
        assert len(masked_sentences) == len(sample_sentences)


class TestDownloadNER:

    def test_download_ner(self):
        download_string = 'american stories:1798'
        download(download_string)
        assert os.path.isdir('data/american_stories_1798')
        ner('data/american_stories_1798', model, batch_size = batch_size)

        shutil.rmtree('data/american_stories_1798')