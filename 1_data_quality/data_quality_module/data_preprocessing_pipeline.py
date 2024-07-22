from preprocessors.preprocessor import DataPreprocessor
import pandas as pd

class DataPreprocessingPipeline:
    def __init__(self, preprocessors: list[DataPreprocessor]): 
        self.preprocessors = preprocessors

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for preprocessor in self.preprocessors:
            df = preprocessor.process(df)
        return df