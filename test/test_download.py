'''
Unit tests for download functions.
'''

import os
import pytest

from news_deja_vu import download, parse_download_string

def test_parse_download_string_american_stories():
    '''
    Covering most cases of parse_download_string options, including a lot of the errors
    '''

    download_string = 'american stories:1776:embeddings'
    download_function, args, default_save_folder = parse_download_string(download_string)
    
    assert download_function.__name__ == 'download_american_stories'
    assert args == {'years': [1776], 'embeddings': True}
    assert default_save_folder == 'american_stories_embeddings_1776'

    download_string = 'american stories:1776-1783'
    download_function, args, default_save_folder = parse_download_string(download_string)

    assert download_function.__name__ == 'download_american_stories'
    assert args == {'year_range': (1776, 1783)}
    assert default_save_folder == 'american_stories_1776-1783'

    download_string = 'american stories:1776,1783,1789'
    download_function, args, default_save_folder = parse_download_string(download_string)

    assert download_function.__name__ == 'download_american_stories'
    assert args == {'years': [1776, 1783, 1789]}
    assert default_save_folder == 'american_stories_1776_1783_1789'

    download_string = 'american stories:embeddings'
    download_function, args, default_save_folder = parse_download_string(download_string)

    assert download_function.__name__ == 'download_american_stories'
    assert args == {'embeddings': True}
    assert default_save_folder == 'american_stories_embeddings'

    download_string = 'american stories:1779:1846,1924'
    with pytest.raises(ValueError):
        download_function, args, default_save_folder = parse_download_string(download_string)

    download_string = 'american stories:1853-1854,1846'
    with pytest.raises(ValueError):
        download_function, args, default_save_folder = parse_download_string(download_string)

    download_string = 'american stories:1853-1854:1889-1892'
    with pytest.raises(ValueError):
        download_function, args, default_save_folder = parse_download_string(download_string)

    download_string = 'american stories:1853-1824'
    with pytest.raises(ValueError):
        download_function, args, default_save_folder = parse_download_string(download_string)

def test_download_american_stories():
    download_string = 'american stories:1776:embeddings'
    with pytest.raises(NotImplementedError):
        download(download_string)

    download_string = 'american stories:embeddings'
    with pytest.raises(NotImplementedError):
        download(download_string)

    download_string = 'american stories:1776'
    download(download_string)
    assert os.path.isdir('data/american_stories_1776')
    os.rmdir('data/american_stories_1776')

    download_string = 'american stories:1776,1777'
    download(download_string)
    assert os.path.isdir('data/american_stories_1776_1777')
    os.rmdir('data/american_stories_1776_1777')

    download_string = 'american stories:1776-1777'
    download(download_string)
    assert os.path.isdir('data/american_stories_1776-1777')
    os.rmdir('data/american_stories_1776-1777')
