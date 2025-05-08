import polars as pl
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
import logging
import argparse

from subjectivity.helper.logger import set_up_log
from subjectivity.models.encoder import EncoderModel
from subjectivity.helper.data_store import DataStore
from subjectivity.models.dim_red import DimRed
from subjectivity.helper.visualization import Visualization
from subjectivity.helper.run_config import RunConfig

def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding Training Data Visualization")

    parser.add_argument("--config_path", type=str, default="config/run_config.yml", help="Config name")
    parser.add_argument("--force", "-f", default=False, action="store_true", help="Force recomputation of embedding")

    return parser.parse_args()

def embed(model_settings: dict[str, str | int | bool ], df: pl.DataFrame) -> np.array:
    """Embed sequence with a transformer model."""

    model_name = model_settings.pop("name")
    batch_size = model_settings.pop("batch_size")

    m = EncoderModel(model_name)
    m.set_up_device()

    m.init(context_window=model_settings['max_length'])
    encoded_sequence = m.encode(df["sentence"].to_list(), encode_settings=model_settings)

    m.load_dataloader(encoded_sequence, batch_size=batch_size)

    m.compute_embeddings()
    embeddings = m.get_embeddings()

    return embeddings

def reduce_embeddings(dim_red_settings: dict[str, str | int | bool ], df: pl.DataFrame) -> pl.DataFrame:
    """Reduce embeddings with a dimensionality reduction model."""

    model_name = dim_red_settings.pop("name")

    embeddings = np.vstack(df["cls_embeddings"].to_list())  # shape (N, D)
    logging.info(f"Reduce CLS embedding of shape '{embeddings.shape[1]}' with '{model_name}' and config '{dim_red_settings}'.")
    dim_red = DimRed(model=model_name, model_params=dim_red_settings)
    reduced_embeddings = dim_red.reduce(embeddings)

    reduced_embeddings_df = pl.DataFrame(
        {
            "Component 1": reduced_embeddings[:, 0],
            "Component 2": reduced_embeddings[:, 1],
            "label_encoded": df["label_encoded"],
            "label": df["label"],
        }
    )

    return reduced_embeddings_df

def check_data_availability(path: Path) -> bool:
    """Check if data is already computed."""
    return path.exists()

def main() -> int:

    set_up_log()
    logging.info("Start embedding clustering")
    try:

        args = init_args_parser()
        logging.info(f"Reading config {args.config_path}")
        rc = RunConfig(Path(args.config_path))
        rc.load_config()

        ds = DataStore(location=rc.data['dir'])
        ds.read_csv_data(rc.data['train_en'], separator="\t")
        df = ds.get_data()

        # str length stats of train
        train_stats = (
            df["sentence"]
            .str.len_chars()
            .describe()
        )
        logging.info(f"Stats of sentence: \n{train_stats}")


        # Transform/encode labels
        LE = LabelEncoder()
        label_encoded = LE.fit_transform(df["label"].to_list())
        df = df.with_columns(pl.Series(name="label_encoded", values=label_encoded))

        encoder_model_name = rc.encoder_model['name']
        embedding_path = Path(ds.location) / Path(rc.data['train_en_embedding'] + '_' + encoder_model_name.replace("/", "-") + '.npy') 
        if check_data_availability(embedding_path) & (not args.force):
            logging.info(f"Embedding of model '{encoder_model_name}' already exist in '{embedding_path}'. Loading matrix.")
            embeddings = np.load(embedding_path)

        else:
            embeddings = embed(rc.encoder_model, df)
            logging.info(f"Saving embeddings of model '{encoder_model_name}' to '{embedding_path}'.")
            np.save(embedding_path, embeddings)

        # Convert each row (1D) of cls_embeddings into a list. 
        # Each row becomes an element in the new column.
        cls_embeddings = embeddings[:, 0, :]
        df = df.with_columns([
            pl.Series(
                name="cls_embeddings", 
                values=[row for row in cls_embeddings]
            )
        ])

        dim_red_model_name = rc.dim_red_model['name']
        reduced_embeddings_df = reduce_embeddings(rc.dim_red_model, df)

        logging.info("Visualizing reduced embedding.")
        visz = Visualization(rc.visualization['fig_dir'], rc.visualization['data_type'], rc.visualization['figure_size'])
        fig_model_name = encoder_model_name + '_' + dim_red_model_name
        visz.plot_embeddings(df=reduced_embeddings_df, groupby_criteria="label_encoded", groupby_criteria_legend="label", model_name=fig_model_name)

        logging.info("Finished embedding clustering.")
        return 0
    except Exception:
        logging.exception("Embedding clustering failed", stack_info=True)
        return 1

main()