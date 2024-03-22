'''
Functions to flexibly create huggingface datasets to feed into huggingface pipelines. 

The following kinds of data objects can be passed into the create_dataset function:
    - directory path
    - list of file paths
    - list of texts
    - list of dictionaries
    - pandas dataframe
    - pandas series

And the behavior will 
'''

import os
import json

import pandas as pd
from datasets import Dataset, load_dataset

def create_dataset_from_list_of_file_paths(file_paths: list) -> Dataset:
    '''
    Create a huggingface dataset from a list of file paths.
    '''
    pass

def create_dataset_from_list_of_dicts(dicts: list) -> Dataset:
    '''
    Create a huggingface dataset from a list of dictionaries.
    '''
    return Dataset.from_list(dicts)

def create_dataset_from_list_of_texts(texts: list) -> Dataset:
    '''
    Create a huggingface dataset from a list of texts.
    '''
    return Dataset.from_dict({'text': texts})

def create_dataset_from_directory(directory: str) -> Dataset:
    '''
    Create a huggingface dataset from a directory of files.
    '''
    pass

def create_dataset_from_series(series: pd.Series) -> Dataset:
    '''
    Create a huggingface dataset from a pandas series.
    '''
    if series.name is 'text':
        return Dataset.from_pandas(series)
    elif series.name is 'files':
        return create_dataset_from_list_of_file_paths(series)
    else:
        raise ValueError('Unrecognized pandas series type, must be named "text" or "files"')
    
def create_dataset_from_dataframe(dataframe: pd.DataFrame) -> Dataset:
    '''
    Create a huggingface dataset from a pandas dataframe.
    '''
    if 'text' in dataframe.columns:
        return create_dataset_from_series(dataframe['text'])
    elif 'files' in dataframe.columns:
        return create_dataset_from_series(dataframe['files'])
    else:
        raise ValueError('Unrecognized pandas dataframe type, must contain a column named "text" or "files"')


def get_dataset(dataset):
    '''
    Create a huggingface dataset from a variety of input types.
    '''
    
    if isinstance(dataset, str):
        if os.path.isdir(dataset):
            return create_dataset_from_directory(dataset)
        else:
            raise ValueError('Unrecognized string input type')
    
    elif isinstance(dataset, list):
        if all([isinstance(x, str) for x in dataset]):
            return create_dataset_from_list_of_file_paths(dataset)
        elif all([isinstance(x, dict) for x in dataset]):
            return create_dataset_from_list_of_dicts(dataset)
        elif all([isinstance(x, str) for x in dataset]):
            return create_dataset_from_list_of_texts(dataset)
        else:
            raise ValueError('Unrecognized list input type')
    
    elif isinstance(dataset, pd.DataFrame):
        return create_dataset_from_dataframe(dataset)
    
    elif isinstance(dataset, pd.Series):
        return create_dataset_from_series(dataset)
    
    elif isinstance(dataset, Dataset):
        return dataset
    
    else:
        raise ValueError('Unrecognized input type')