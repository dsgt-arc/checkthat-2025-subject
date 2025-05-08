import logging
from pathlib import Path

import yaml


class RunConfig:
    """Singleton class to store config data."""

    data = {}
    encoder_model = {}
    dim_red_model = {}
    visualization = {}
    reranking = {}
    train = {}
    llm = {}

    @classmethod
    def load_config(cls, path: Path = Path("config/run_config.yml")) -> None:
        """Load YAML config into a RunConfig object."""
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "r") as f:
            config_data = yaml.safe_load(f)

        cls.data = config_data.get("data")
        cls.encoder_model = config_data.get("encoder_model")
        cls.dim_red_model = config_data.get("dim_red_model")
        cls.visualization = config_data.get("visualization")
        cls.reranking = config_data.get("reranking")
        cls.train = config_data.get("train")
