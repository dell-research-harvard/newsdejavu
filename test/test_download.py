'''
Unit tests for download functions.
'''

import os
import pytest
import shutil


from newsdejavu import download, parse_download_string

class TestParseDownloadStringAmericanStories:
    '''
        Covering most cases of parse_download_string options, including a lot of the errors
    '''
    def test_1(self):
        download_string = 'american stories:1776:embeddings'
        download_function, args, default_save_folder = parse_download_string(download_string)
        
        assert download_function.__name__ == 'download_american_stories'
        assert args == {'years': [1776], 'embeddings': True}
        assert default_save_folder == 'data/american_stories_embeddings_1776'

    def test_2(self):
        download_string = 'american stories:1776-1783'
        download_function, args, default_save_folder = parse_download_string(download_string)

        assert download_function.__name__ == 'download_american_stories'
        assert args == {'year_range': (1776, 1783)}
        assert default_save_folder == 'data/american_stories_1776-1783'

    def test_3(self):
        download_string = 'american stories:1776,1783,1789'
        download_function, args, default_save_folder = parse_download_string(download_string)

        assert download_function.__name__ == 'download_american_stories'
        assert args == {'years': [1776, 1783, 1789]}
        assert default_save_folder == 'data/american_stories_1776_1783_1789'

    def test_4(self):
        download_string = 'american stories:embeddings'
        download_function, args, default_save_folder = parse_download_string(download_string)

        assert download_function.__name__ == 'download_american_stories'
        assert args == {'embeddings': True}
        assert default_save_folder == 'data/american_stories_embeddings'

    def test_5(self):
        download_string = 'american stories:1779:1846,1924'
        with pytest.raises(ValueError):
            download_function, args, default_save_folder = parse_download_string(download_string)

    def test_6(self):
        download_string = 'american stories:1853-1854:1846'
        with pytest.raises(ValueError):
            download_function, args, default_save_folder = parse_download_string(download_string)

    def test_7(self):
        download_string = 'american stories:1853-1854:1889-1892'
        with pytest.raises(ValueError):
            download_function, args, default_save_folder = parse_download_string(download_string)

    def test_8(self):
        download_string = 'american stories:1853-1824'
        with pytest.raises(ValueError):
            download_function, args, default_save_folder = parse_download_string(download_string)

    def test_9(self):
        download_string = 'american stories:gobbledegook'
        with pytest.raises(ValueError):
            download_function, args, default_save_folder = parse_download_string(download_string)


class TestDownloadsAmericanStories:
    '''
    Covering most cases of download for American Stories, including errors
    '''

    def test_1(self):
        download_string = 'american stories:1798:embeddings'
        with pytest.raises(NotImplementedError):
            download(download_string)

    def test_2(self):
        download_string = 'american stories:embeddings'
        with pytest.raises(NotImplementedError):
            download(download_string)

    def test_3(self):
        download_string = 'american stories:1798'
        download(download_string)
        assert os.path.isdir('data/american_stories_1798')
        # shutil.rmtree('data/american_stories_1798')

    # def test_4(self):
    #     download_string = 'american stories:1798,1799'
    #     download(download_string)
    #     assert os.path.isdir('data/american_stories_1798_1799')
    #     shutil.rmtree('data/american_stories_1798_1799')

    # def test_5(self):
    #     download_string = 'american stories:1798-1799'
    #     download(download_string)
    #     assert os.path.isdir('data/american_stories_1798-1799')
    #     shutil.rmtree('data/american_stories_1798-1799')
