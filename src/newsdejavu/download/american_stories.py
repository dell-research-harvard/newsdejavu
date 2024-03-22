'''
Download function for American Stories dataset.
'''

import os
import requests
import re
from datasets import load_dataset
import json

MIN_AMERICAN_STORIES_YEAR = 1774
MAX_AMERICAN_STORIES_YEAR = 1963

def parse_american_stories_args(args: list) -> dict:
    """
    Parse the American Stories download string into a dictionary of args. There are two possible args for the American Stories download function:

    - 'year': the year of the articles to download
    - 'embeddings': whether to download entity-masked SBERT embeddings for the articles (also downloads raw texts)

    Args:
    - args: a string containing the args to the American Stories download function

    Returns:
    - a dictionary of args
    - the default save folder for this dataset, which is based on the args and used by download() if no save folder is specified
    """
    inputs = {}
        
    
    # Check for and add all arg(s)
    years_specified_regex = re.compile(r'^\d{4}(?:,\d{4})*$')
    year_range_regex = re.compile(r'^\d{4}-\d{4}$')
    for arg in args:
        if arg == 'embeddings':
            inputs['embeddings'] = True
        elif years_specified_regex.match(arg):
            if 'years' in inputs:
                raise ValueError('Cannot specify multiple sets of years for American Stories dataset. Specify as :year,year,year:')
            
            inputs['years'] = []
            for year in arg.split(','):
                year = int(year)
                if year < MIN_AMERICAN_STORIES_YEAR or year > MAX_AMERICAN_STORIES_YEAR:
                    raise ValueError(f'Year {year} is out of range for American Stories dataset, \
                                     which only contains articles from {MIN_AMERICAN_STORIES_YEAR} to {MAX_AMERICAN_STORIES_YEAR}.')
                else:
                    inputs['years'].append(year)

        elif year_range_regex.match(arg):
            if 'year_range' in inputs:
                raise ValueError('Cannot specify multiple year ranges for American Stories dataset. Specify as :start_year-end_year:')
            
            start_year, end_year = map(int, arg.split('-'))
            if start_year < MIN_AMERICAN_STORIES_YEAR or end_year > MAX_AMERICAN_STORIES_YEAR:
                raise ValueError(f'Year range {start_year}-{end_year} is out of range for American Stories dataset, \
                                 which only contains articles from {MIN_AMERICAN_STORIES_YEAR} to {MAX_AMERICAN_STORIES_YEAR}.')
            elif start_year >= end_year:
                raise ValueError(f'Invalid year range: {start_year}-{end_year}. Start year must be less than end year.')
            else:
                inputs['year_range'] = (start_year, end_year)
        else:
            raise ValueError(f'Unrecognized argument: {arg}')
        
    if 'years' in inputs and 'year_range' in inputs:
        raise ValueError('Cannot specify both individual years and a year range for American Stories dataset.')

    default_save_folder = 'american_stories'
    if 'embeddings' in inputs:
        default_save_folder += '_embeddings'
    if 'years' in inputs and len(inputs['years']) > 0:
        to_add = '_'.join(map(str, inputs['years']))
        default_save_folder += f"_{to_add}"
    elif 'year_range' in inputs:
        default_save_folder += f"_{inputs['year_range'][0]}-{inputs['year_range'][1]}"

    return inputs, default_save_folder


def download_american_stories(save_folder: str, **kwargs):
    """
    Download the American Stories dataset and save it to the indicated folder.

    Args:
    - save_folder: the folder in which to save the downloaded data
    - kwargs: a dictionary of args for the American Stories download function
    """
    import huggingface_hub

    REPO_ID = 'dell-research-harvard/AmericanStories'
    
    # Get the embeddings arg
    embeddings = kwargs.get('embeddings', False)

    if embeddings:
        raise NotImplementedError('Embeddings are not yet available for the American Stories dataset.')
    
    # Get the years arg
    years = kwargs.get('years', [])
    year_range = kwargs.get('year_range', None)
    if year_range:
        years = list(range(year_range[0], year_range[1]+1))
    
    years = list(map(str, years))

    # Remove years that already have a downloaded dataset
    if os.path.exists(save_folder):
        downloaded_years = [f.split('_')[-1] for f in os.listdir(save_folder)]
        years = [year for year in years if year not in downloaded_years]
    
    if years:
        # Download the files
        dataset = load_dataset("dell-research-harvard/AmericanStories", year_list = years)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        for year in years:
            dataset[year].to_json(os.path.join(save_folder, f'dataset_{year}.json'))
                     
    print([os.path.join(save_folder, f'dataset_{year}.json') for year in years])
    return load_dataset("json", data_files = [os.path.join(save_folder, f'dataset_{year}.json') for year in years])['train']

