import polars as pl
from pathlib import Path
import logging
from subjectivity.helper.run_config import RunConfig
class DataStore:

    def __init__(self, location: str = "data") -> None:
        self.location = Path(location)
        self.data = None

    def read_json_data(self, file: str) -> None:
        """Reads data from a file and stores it in the data attribute as pl.DataFrame."""
        path = self.location / Path(file)
        logging.info(f"Reading {str(path)}")
        self.data = pl.read_json(path)

    def read_csv_data(self, file: str, separator=",") -> None:
        """Reads data from a file and stores it in the data attribute as pl.DataFrame."""
        path = self.location / Path(file)
        logging.info(f"Reading {str(path)}")
        self.data = pl.read_csv(path, separator=separator)

    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise ValueError("Data not read yet. Call read_data() first.")
        return self.data
    
def read_all_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read train and validation data."""
    logging.info(f"Reading train, val and test data from '{RunConfig.data['train_en']}', '{RunConfig.data['val_en']}', and '{RunConfig.data['test_en']}'.")

    ds = DataStore(location=RunConfig.data["dir"])
    ds.read_csv_data(RunConfig.data["train_en"], separator="\t")
    df_train = ds.get_data()

    ds.read_csv_data(RunConfig.data["val_en"], separator="\t")
    df_val = ds.get_data()

    ds.read_csv_data(RunConfig.data["test_en"], separator="\t")
    df_test = ds.get_data()


    return df_train, df_val, df_test
