from preprocessors.preprocessor import DataPreprocessor
import pandas as pd


class DeleteRowsPreProcessor(DataPreprocessor):
    def __init__(self, cols_for_which_rows_to_delete: list[int]):
        self.cols_for_which_rows_to_delete = cols_for_which_rows_to_delete

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols_for_which_rows_to_delete:
            df.dropna(subset=[col], inplace=True)
        return df
