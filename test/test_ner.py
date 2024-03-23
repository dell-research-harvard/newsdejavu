"""Unit tests for named entity recognition functions."""


import os
import pytest
import shutil
import json
from datasets import load_dataset, Dataset

from newsdejavu import ner, mask, ner_and_mask
from newsdejavu import download
from newsdejavu import embed, find_nearest_neighbours

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

local_ner_model = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"
huggingface_ner_model = 'dell-research-harvard/historical_newspaper_ner'

same_story_local_model = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/same_story_model'
same_story_huggingface_model = ''
batch_size = 10

sample_query_sentences = [
    "Elon Musk's SpaceX is leading the private space industry.",
    "The United Nations addressed climate change at the conference in Paris.",
    "Serena Williams triumphed at the Wimbledon Championships."
]

class TestSimpleNER:
    
    def test_ner(self, sample_sentences):
        ner_output=ner(sample_sentences, local_ner_model, batch_size = batch_size)
        assert len(ner_output) == len(sample_sentences)
    
    def test_mask(self, sample_sentences):
        ner_output = ner(sample_sentences, local_ner_model, batch_size = batch_size)
        masked_sentences = mask(ner_output)
        assert len(masked_sentences) == len(sample_sentences)

    def test_ner_and_mask(self, sample_sentences):
        masked_sentences = ner_and_mask(sample_sentences, local_ner_model, batch_size = batch_size)
        assert len(masked_sentences) == len(sample_sentences)    

    def test_ner_and_mask_diff_masks(self, sample_sentences):
        masked_sentences = ner_and_mask(sample_sentences, local_ner_model, batch_size = batch_size, all_masks_same = False)
        assert len(masked_sentences) == len(sample_sentences)
    
    def test_ner_and_mask_clean_ocr(self, sample_sentences):
        masked_sentences = ner_and_mask(sample_sentences, local_ner_model, batch_size = batch_size, preprocess_for_ocr_errors = True)
        assert len(masked_sentences) == len(sample_sentences)


class TestDatasetNER:

    def test_american_stories_ner(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        output = ner(dataset, local_ner_model, batch_size = batch_size)
        assert len(output) == len(dataset)

    def test_american_stores_ner_clean_ocr(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        output = ner(dataset, local_ner_model, batch_size = batch_size, preprocess_for_ocr_errors = True)
        assert len(output) == len(dataset)

    def test_american_stories_ner_model_download(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        output = ner(dataset, huggingface_ner_model, batch_size = batch_size)
        assert len(output) == len(dataset)


class TestDatasetMask:

    def test_american_stories_mask(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        ner_output = ner(dataset, local_ner_model, batch_size = batch_size)
        masked_output = mask(ner_output)
        assert len(masked_output) == len(dataset)

    def test_american_stories_mask_model_download(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        ner_output = ner(dataset, 'dell-research-harvard/historical_newspaper_ner', batch_size = batch_size)
        masked_output = mask(ner_output)
        assert len(masked_output) == len(dataset)

class TestDatasetNERAndMask:

    def test_american_stories(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        output = ner_and_mask(dataset, local_ner_model, batch_size = batch_size)
        assert len(output) == len(dataset)


class TestDatasetNERMaskEmbed:

    def test_american_stories(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        output = ner_and_mask(dataset, local_ner_model, batch_size = batch_size)
        assert len(output) == len(dataset)
        embeddings = embed(output, same_story_local_model)
        assert embeddings.shape[0] == len(dataset)

class TestQueryPipeline:

    def test_american_stories_query(self):
        dataset = load_dataset('json', data_files = 'data/test_data/american_stories_1870_test.json')['train']
        assert isinstance(dataset, Dataset)
        output = ner_and_mask(dataset, local_ner_model, batch_size = batch_size)
        assert len(output) == len(dataset)
        embeddings = embed(output, same_story_local_model)
        assert embeddings.shape[0] == len(dataset)

        query_masked_input = ner_and_mask(sample_query_sentences, local_ner_model, batch_size = batch_size)
        query_embeddings = embed(query_masked_input, same_story_local_model)

        dist_list, nn_list = find_nearest_neighbours(query_embeddings, embeddings, k=1)
        assert dist_list.shape[0] == len(sample_query_sentences)
        assert nn_list.shape[0] == len(sample_query_sentences)

        results_dict = {i: {"query": sample_query_sentences[i], "neighbor": dataset[nn_list[i]]} for i in range(len(sample_query_sentences))}
        assert len(results_dict) == len(sample_query_sentences)

        with open('data/test_data/query_results.json', 'w') as f:
            json.dump(results_dict, f, indent = 4)

    def test_american_stories_query_high_level(self):
        