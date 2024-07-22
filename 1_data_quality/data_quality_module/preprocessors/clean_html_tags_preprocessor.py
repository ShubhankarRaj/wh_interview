from preprocessors.preprocessor import DataPreprocessor
from bs4 import BeautifulSoup
import pandas as pd

class RemoveHTMLTagsPreProcessor(DataPreprocessor):
    def __init__(self, cols):
        self.cols = cols
    
    def remove_html_tags(self, text):
        if isinstance(text, str):
            return BeautifulSoup(text, "html.parser").get_text()
        return text

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.cols] = df[self.cols].apply(self.remove_html_tags, axis=1)
        return df
