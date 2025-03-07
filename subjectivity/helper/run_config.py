# run_config.py
import yaml
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RunConfig:
    """ Singeltone class to store config data."""
    path: Path = field(default=Path("config/run_config.yml"))
    data: dict = field(default_factory=dict)
    encoder_model: dict = field(default_factory=dict)
    dim_red_model: dict = field(default_factory=dict)
    visualization: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)

    def load_config(self) -> None:
        """Load YAML config into a RunConfig object."""
        with open(self.path, "r") as f:
            config_data = yaml.safe_load(f)

        self.data=config_data.get("data")
        self.encoder_model=config_data.get("encoder_model")
        self.dim_red_model=config_data.get("dim_red_model")
        self.visualization=config_data.get("visualization")
        self.train=config_data.get("train")
