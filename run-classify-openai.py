import argparse
import importlib
import logging
import os
import re
from pathlib import Path

import polars as pl
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnableSequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from helper.data_store import DataStore
from helper.logger import set_up_log
from helper.run_config import RunConfig


def extract_label(llm_output: str):
    """Extracts a single-word classification (SUBJ or OBJ) from LLM output."""
    match = re.search(r"\b(SUBJ|OBJ)\b", llm_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Scientific Detection Classification")
    parser.add_argument("--config_path", type=str, default="config/open-ai.yml", help="Config name")
    parser.add_argument(
        "--force", action="store_true", help="Force re-run of the prediction even if the results are already available."
    )
    return parser.parse_args()


def prepare_chain(model: BaseLLM, few_shot_examples: str = None) -> RunnableSequence:
    """
    Prepares a chain that uses the prompts from the configuration.
    The user prompt template appends the tweet text via the variable {tweet}.
    """
    system_msg = RunConfig.llm["prompt"]["system"]
    user_template = RunConfig.llm["prompt"]["user"]
    # Combine into a single prompt string (alternatively, you could use a multi-message prompt)
    template = f"System: {system_msg}\nUser: {user_template}"
    if few_shot_examples:
        prompt = FewShotPromptTemplate.from_template(template, few_shot_examples)
    else:
        prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain


def classify_tweet(chain: RunnableSequence, tweet: str):
    """
    Uses the provided chain to classify the tweet.
    The tweet text is injected into the prompt, and the LLM response is parsed.
    """
    prompt_input = {"tweet": tweet}
    response = chain.invoke(prompt_input)
    if response is None:
        logging.error(f"Failed to classify tweet: {tweet}")
        raise ValueError(f"Failed to classify tweet: {tweet}")
    classification = extract_answer_list(response.content)
    return classification


def compute_metrics(labels: pl.DataFrame, predictions: pl.DataFrame, col_set: list[tuple[str, str]]) -> pl.DataFrame:
    rounding = RunConfig.llm["round_results"]

    accuracy, macro_precision, macro_recall, macro_f1 = dict(), dict(), dict(), dict()
    for label_col, pred_col in col_set:
        accuracy[label_col] = round(accuracy_score(labels[label_col], predictions[pred_col]), rounding)
        macro_precision[label_col] = round(precision_score(labels[label_col], predictions[pred_col], average="macro"), rounding)
        macro_recall[label_col] = round(recall_score(labels[label_col], predictions[pred_col], average="macro"), rounding)
        macro_f1[label_col] = round(f1_score(labels[label_col], predictions[pred_col], average="macro"), rounding)

    # Average the metrics across all categories
    accuracy["average"] = round(sum(accuracy.values()) / len(accuracy), rounding)
    macro_precision["average"] = round(sum(macro_precision.values()) / len(macro_precision), rounding)
    macro_recall["average"] = round(sum(macro_recall.values()) / len(macro_recall), rounding)
    macro_f1["average"] = round(sum(macro_f1.values()) / len(macro_f1), rounding)

    # Compute metrics for the multi-class classification
    accuracy["multi-class"] = round(accuracy_score(labels, predictions), rounding)
    macro_precision["multi-class"] = round(precision_score(labels, predictions, average="macro"), rounding)
    macro_recall["multi-class"] = round(recall_score(labels, predictions, average="macro"), rounding)
    macro_f1["multi-class"] = round(f1_score(labels, predictions, average="macro"), rounding)

    # Combine all statistics into a single polar DataFrame
    metrics = pl.DataFrame(
        {
            "category": list(accuracy.keys()),
            "accuracy": list(accuracy.values()),
            "macro_precision": list(macro_precision.values()),
            "macro_recall": list(macro_recall.values()),
            "macro_f1": list(macro_f1.values()),
        },
    )

    return metrics


def set_up_llm(api_key: str = None) -> BaseLLM:
    logging.info(f"Setting up model Provider: {RunConfig.llm['provider']} with model config: {RunConfig.llm['model_config']}")
    # Import langchain model module
    llm_module = importlib.import_module(f"langchain_{RunConfig.llm['provider']}")
    # Retrieve the class from that module
    llm_class = getattr(llm_module, RunConfig.llm["class_name"])
    # Instantiate your LLM with the parameters
    llm_config = RunConfig.llm["model_config"]
    if api_key:
        llm_config["api_key"] = api_key
    configured_model = llm_class(**llm_config)

    return configured_model


def resolve_nesting_in_df(df: pl.DataFrame, label_col_names: list[str], pred_col_names: list[str]) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("labels").list.to_struct(fields=label_col_names),
        pl.col("predictions").list.to_struct(fields=pred_col_names),
    )
    df = df.unnest(["labels", "predictions"])
    # Cast all floats to ints
    df = df.with_columns(pl.col(col).cast(pl.Int16) for col in label_col_names + pred_col_names)

    return df


def check_data_availability(path: Path) -> bool:
    """Check if data is already computed."""
    return path.exists()


def main():
    set_up_log()
    logging.info("Start LLM Prediction")
    try:
        args = init_args_parser()

        logging.info(f"Reading config {args.config_path}")
        RunConfig.load_config(Path(args.config_path))

        logging.info("Setting up prediction with mutli-class (3).")
        cats = range(3)  # Multi-class classification with 3 categories
        field_names_label = [f"cat_{i}" for i in cats]
        field_names_pred = [f"pred_cat_{i}" for i in cats]

        logging.info("Reading API Key from environment variable")
        model_provider = RunConfig.llm["provider"]
        if not os.environ.get(f"{model_provider.upper()}_API_KEY"):
            logging.error(f"API Key for {model_provider} is not set.")
            raise ValueError(f"API Key for {model_provider} is not set.")
        key = os.environ.get(f"{model_provider.upper()}_API_KEY")

        # Reading the training data
        ds = DataStore(RunConfig.data["dir"])
        ds.read_csv_data(RunConfig.data["train"], separator="\t", schema_overrides={"labels": pl.String()})
        train_data = ds.get_data()

        output_path = Path(RunConfig.data["dir"]) / Path(RunConfig.data["train_pred"])
        if check_data_availability(output_path) and not args.force:
            # Load the existing results
            logging.info("Prediction results available. Skipping new predictions.")
            ds.read_csv_data(RunConfig.data["train_pred"], separator="\t")
            unnested_results = ds.get_data()

        else:
            logging.info("Data not processed yet. Starting prediction.")

            # Parse the string label into a list of floats
            train_data = train_data.with_columns(
                pl.col("labels").str.replace_all(r"[\[\]]", "").str.split(", ").cast(pl.List(pl.Float32))
            )

            if RunConfig.llm["few_shot"]:
                logging.info("Few-shot learning enabled.")
                raise NotImplementedError("Few-shot learning is not yet implemented.")
                if RunConfig.llm["prompt"]["assistance"] is None:
                    logging.error("Assistance prompt is required for few-shot learning.")
                    raise ValueError("Assistance prompt is required for few-shot learning.")

                # Given seed randomly pick 2 examples from the training data for few-shot learning
                few_shot_examples = train_data.sample(n=2, random_state=RunConfig.llm["few_shot_seed"])
            else:
                few_shot_examples = None

            # Initialize the LLM
            model = set_up_llm(api_key=key)

            # Prepare the chain
            chain = prepare_chain(model, few_shot_examples)

            # Predictions with saving intermediate results every RunConfig.llm["save_every"] rows
            predictions = list(None for _ in range(len(train_data)))

            for i, row in enumerate(train_data.iter_rows(named=True)):
                tweet_text = row["text"]
                category = classify_tweet(chain, tweet_text)
                predictions[i] = category

                if (i + 1) % RunConfig.llm["save_every"] == 0 and i > 0:
                    train_data = train_data.with_columns(pl.Series("predictions", predictions))
                    unnested_results = resolve_nesting_in_df(train_data.clone(), field_names_label, field_names_pred)
                    unnested_results.write_csv(output_path, separator="\t")
                    logging.info(f"Saved {i + 1} processed rows.")

            # Save final results
            train_data = train_data.with_columns(pl.Series("predictions", predictions))
            unnested_results = resolve_nesting_in_df(train_data, field_names_label, field_names_pred)
            output_path = Path(RunConfig.data["dir"]) / Path(RunConfig.data["train_pred"])
            logging.info(f"Writing predictions to {output_path}")
            unnested_results.write_csv(output_path, separator="\t")

        logging.info(f"Finished predictions. Sample output:\n{unnested_results.tail()}")

        # Compute the evaluation metrics
        metrics = compute_metrics(
            unnested_results[field_names_label],
            unnested_results[field_names_pred],
            list(zip(field_names_label, field_names_pred)),
        )

        logging.info(f"Metrics:\n{metrics}")

        logging.info(f"Finished prediction with {model_provider}.")
        return 0

    except Exception:
        logging.exception("Prediction failed", stack_info=True)
        return 1


main()
