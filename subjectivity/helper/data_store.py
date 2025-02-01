import polars as pl
from pathlib import Path
import logging

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