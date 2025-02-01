import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
import logging
from typing import Optional

class Visualization:
    def __init__(
        self, fig_dir: str = "fig", data_type: str = "png", figure_size: tuple = (6, 6), dpi: int = 300, cmap: str = "Spectral"
    ):
        self.fig_dir = Path(fig_dir)
        if data_type not in ["png", "jpg", "svg"]:
            raise ValueError(f"Data type '{data_type}' not supported.")
        self.data_type = data_type
        if isinstance(figure_size, list):
            figure_size = tuple(figure_size)
        if not isinstance(figure_size, tuple):
            raise ValueError(f"Figure size '{figure_size}' is not a tuple.")
        self.figure_size = figure_size
        self.dpi = dpi
        self.cmap = cmap

        if not self.fig_dir.exists():
            self.fig_dir.mkdir(exist_ok=True, parents=True)

    def plot_embeddings(self, df: pl.DataFrame, model_name: str, groupby_criteria: str, groupby_criteria_legend: Optional[str] = None):
        """Plots two components (Component 1, Component 2) of a reduced embedding of model model_name.
        Points are highlighted given gorupby_criteria."""

        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi, tight_layout=True)

        scatter = ax.scatter(
            df["Component 1"],
            df["Component 2"],
            cmap=self.cmap,
            c=df[groupby_criteria],
            s=1,
        )

        if groupby_criteria_legend is None:
            handles, _ = scatter.legend_elements()
            unique_values = df[groupby_criteria].unique().to_list()
            legend_labels = [str(value) for value in unique_values]  # Ensure labels are strings
        else:
            unique_values = df.select(groupby_criteria, groupby_criteria_legend).unique()
            unique_values_dict = dict(zip(unique_values[groupby_criteria].to_list(), unique_values[groupby_criteria_legend].to_list()))
            # Add legend elements
            handles, _ = scatter.legend_elements()
            legend_labels = [unique_values_dict[int(label)] for label in unique_values[groupby_criteria].to_list()]
        ax.legend(handles, legend_labels, title=groupby_criteria_legend.capitalize(), loc="best", fontsize="small")


        ax.set_title(f"Scatter Plot of {model_name} Embeddings")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        model_name = model_name.replace("/", "_")
        file_path = self.fig_dir / Path(f"embeddings_{model_name}.{self.data_type}")
        logging.info(f"Saving figure to '{str(file_path)}'.")
        fig.savefig(file_path)
