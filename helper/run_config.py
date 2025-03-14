import yaml
from pathlib import Path


class RunConfig:
    """Singleton class to store config data."""

    data = {}
    llm = {}
    train = {}

    @classmethod
    def load_config(cls, path: Path = Path("config/run_config.yml")) -> None:
        """Load YAML config into a RunConfig object."""
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "r") as f:
            config_data = yaml.safe_load(f)

        cls.data = config_data.get("data")
        cls.llm = config_data.get("llm")
        cls.train = config_data.get("train")
