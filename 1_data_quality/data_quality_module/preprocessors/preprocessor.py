from abc import ABC, abstractmethod
import pandas as pd

class DataPreprocessor(ABC):

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
