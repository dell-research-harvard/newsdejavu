'''
Download function for American Stories dataset.
'''

import os
import requests
import re

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
    args = {}
    
    # Check for and add the embeddings arg
    if 'embeddings' in args:
        args['embeddings'] = True
    
    # Check for and add the year arg(s)
    years_specified_regex = re.compile(r'^\d{4}(?:,\d{4})*$')
    year_range_regex = re.compile(r'^\d{4}-\d{4}$')
    for arg in args:
        if years_specified_regex.match(arg):
            if 'years' in args:
                raise ValueError('Cannot specify multiple sets of years for American Stories dataset. Specify as :year,year,year:')
            
            args['years'] = []
            for year in arg.split(','):
                year = int(year)
                if year < MIN_AMERICAN_STORIES_YEAR or year > MAX_AMERICAN_STORIES_YEAR:
                    raise ValueError(f'Year {year} is out of range for American Stories dataset, \
                                     which only contains articles from {MIN_AMERICAN_STORIES_YEAR} to {MAX_AMERICAN_STORIES_YEAR}.')
                else:
                    args['years'].append(year)

        elif year_range_regex.match(arg):
            if 'year_range' in args:
                raise ValueError('Cannot specify multiple year ranges for American Stories dataset. Specify as :start_year-end_year:')
            
            start_year, end_year = map(int, arg.split('-'))
            if start_year < MIN_AMERICAN_STORIES_YEAR or end_year > MAX_AMERICAN_STORIES_YEAR:
                raise ValueError(f'Year range {start_year}-{end_year} is out of range for American Stories dataset, \
                                 which only contains articles from {MIN_AMERICAN_STORIES_YEAR} to {MAX_AMERICAN_STORIES_YEAR}.')
            elif start_year >= end_year:
                raise ValueError(f'Invalid year range: {start_year}-{end_year}. Start year must be less than end year.')
            else:
                args['year_range'] = (start_year, end_year)
        
    if 'years' in args and 'year_range' in args:
        raise ValueError('Cannot specify both individual years and a year range for American Stories dataset.')

    default_save_folder = 'american_stories'
    if 'embeddings' in args:
        default_save_folder += '_embeddings'
    if 'years' in args and len(args['years']) > 0:
        default_save_folder += f"_{'_'.join(map(str, args['years']))}"
    elif 'year_range' in args:
        default_save_folder += f"_{args['year_range'][0]}-{args['year_range'][1]}"

    return args, default_save_folder


def download_american_stories(save_folder: str | os.PathLike, **kwargs):
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

    download_files = []
    for year in years:
        download_files.append(f'faro_{year}.tar.gz')

    if year_range:
        for year in range(year_range[0], year_range[1]+1):
            download_files.append(f'faro_{year}.tar.gz')

    # Download the files
    for file in download_files:
        huggingface_hub.download_from_hf_hub(REPO_ID, file, repo_type='dataset', cache_dir = save_folder)

