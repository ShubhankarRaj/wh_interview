from preprocessors.preprocessor import DataPreprocessor
import pandas as pd

class LowerCasePreProcessor(DataPreprocessor):
    def __init__(self, cols):
        self.cols = cols

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        print(self.cols)
        for col in self.cols:
            df[col] = df[col].str.lower()
        return df
