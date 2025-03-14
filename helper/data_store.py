import logging
from pathlib import Path
from typing import Optional

import polars as pl


class DataStore:
    def __init__(self, location: str = "data") -> None:
        self.location = Path(location)
        self.data = None

    def read_json_data(self, file: str) -> None:
        """Reads data from a file and stores it in the data attribute as pl.DataFrame."""
        path = self.location / Path(file)
        logging.info(f"Reading {str(path)}")
        self.data = pl.read_json(path)

    def read_csv_data(self, file: str, separator: str = ",", schema_overrides: Optional[dict[str, str]] = None) -> None:
        """Reads data from a file and stores it in the data attribute as pl.DataFrame."""
        path = self.location / Path(file)
        logging.info(f"Reading {str(path)}")
        self.data = pl.read_csv(path, separator=separator, schema_overrides=schema_overrides)

    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise ValueError("Data not read yet. Call read_data() first.")
        return self.data
