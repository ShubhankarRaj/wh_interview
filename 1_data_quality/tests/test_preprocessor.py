import pytest
import pandas as pd
import sys, os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_quality_module')))

from preprocessors.preprocessor import DataPreprocessor
from preprocessors.delete_cols_preprocessor import DeleteColsPreprocessor
from preprocessors.delete_rows_preprocessor import DeleteRowsPreProcessor
from preprocessors.lower_case_preprocessor import LowerCasePreProcessor
from preprocessors.clean_html_tags_preprocessor import RemoveHTMLTagsPreProcessor

@pytest.fixture
def df():
    df = pd.DataFrame({'col0': [1,2,3,4], 'col1': ['A', 'b', 'C', 'd'], 'col2': ['d', None, 'F', 'G'], 'que': ['<p>Hello</p>', 'World', '<b>Bold</b>', '!'],
        'ans': ['<div>Div</div>', '<i>Italic</i>', 'No HTML', '.'],})
    return df

def test_delete_cols_preprocessor(df):
    preprocessor = DeleteColsPreprocessor(['col1', 'col2'])
    df = preprocessor.process(df)
    assert df.columns.tolist() == ['col0', 'que', 'ans']

def test_delete_rows_preprocessor(df):
    rows_count_before_preprocessing = df.shape[0]
    preprocessor = DeleteRowsPreProcessor(['col2'])
    df = preprocessor.process(df)
    assert df.columns.tolist() == ['col0', 'col1', 'col2', 'que', 'ans']
    assert df.shape[0] < rows_count_before_preprocessing


def test_lower_case_preprocessor(df):
    preprocessor = LowerCasePreProcessor(['col1', 'col2'])
    df = preprocessor.process(df)
    print(df)
    assert df['col1'].str.islower().all(), "Column 'col1' should be lower case"
    assert df['col2'].str.islower().all(), "Column 'col2' should be lower case"

def test_clean_html_tags_preprocessor(df):

    html_clean_df = pd.DataFrame({'col0': [1,2,3,4], 'col1': ['A', 'b', 'C', 'd'], 'col2': ['d', None, 'F', 'G'], 'que': ['Hello', 'World', 'Bold', '!'],
        'ans': ['Div', 'Italic', 'No HTML', '.'],})
    preprocessor = RemoveHTMLTagsPreProcessor(['que', 'ans'])
    df = preprocessor.process(df)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df, html_clean_df)


if __name__ == '__main__':
    pytest.main()