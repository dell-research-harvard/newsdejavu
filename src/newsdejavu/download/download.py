'''
Main download module for news_deja_vu package.

The module mainly implements the top level function download() which is used to download corpuses of news articles 
from the web. In particular, download() can fetch each of the following corpus types:

- 'american stories': fetches articles from the American Stories dataset

The main download() function ensures that the query is valid, and then calls the appropriate function to download the
indicated dataset. The function also ensures that the downloaded data is saved to the appropriate location.
'''

import os
import requests
from typing import Tuple

from .american_stories import parse_american_stories_args, download_american_stories

PARSER_MAP = {
    'american stories': parse_american_stories_args
}

DOWNLOAD_MAP = {
    'american stories': download_american_stories
}

def parse_download_string(download_string: str) -> Tuple[callable, dict, str]:
    """
    Parse the download string into a tuple containing the dataset function, args to that funciton, and the default save folder.

    Args:
    - download_string: a string containing the dataset to download and the save folder, separated by a comma

    Returns:
    - a tuple containing the dataset download function, a dictionary of args, and the save folder
    """

    dataset = download_string.split(':')[0]

    if dataset not in PARSER_MAP:
        raise ValueError(f'Unrecognized dataset: {dataset}')
    elif dataset not in DOWNLOAD_MAP:
        raise ValueError(f'Unrecognized dataset: {dataset}')
    
    args, default_save_folder = PARSER_MAP[dataset](download_string.split(':')[1:])
    download_function = DOWNLOAD_MAP[dataset]
    print(args, default_save_folder)
    return download_function, args, os.path.join('data', default_save_folder)


def download(dataset: str, save_folder: str = None):
    '''
    download the indicated dataset to the indicated save folder
    '''

    # Get params and download function
    fetcher, args, default_save_folder = parse_download_string(dataset)
    
    # Override the default save folder if one is provided
    save_folder = save_folder or default_save_folder

    fetcher(save_folder, **args)