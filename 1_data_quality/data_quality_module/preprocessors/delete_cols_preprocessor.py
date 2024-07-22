from preprocessors.preprocessor import DataPreprocessor
import pandas as pd

class DeleteColsPreprocessor(DataPreprocessor):
    def __init__(self, cols):
        self.cols = cols

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df.drop(columns=self.cols, axis=1, inplace=True)
        return df
