import argparse
import importlib
import logging
import os
import re
import random
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
    """modified: Extracts a single-word classification (SUBJ or OBJ) from LLM output."""
    match = re.search(r"\b(SUBJ|OBJ)\b", llm_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def init_args_parser() -> argparse.Namespace:
    """modified: for sentence classification"""
    parser = argparse.ArgumentParser(description="LLM Sentence Classification")
    parser.add_argument("--config_path", type=str, default="config/open-ai.yml", help="Config file path")
    parser.add_argument(
        "--force", action="store_true", help="Force re-run of classification even if the results already exist."
    )
    return parser.parse_args()


def prepare_chain(model: BaseLLM) -> RunnableSequence:
    """Prepares a prompt chain, supporting few-shot learning if enabled."""
    system_msg = RunConfig.llm["prompt"]["system"]
    user_template = RunConfig.llm["prompt"]["user"]
    template = f"System: {system_msg}\nUser: {user_template}"

    if RunConfig.llm["few_shot"]:
        logging.info("Few-shot learning is enabled. Sampling few-shot examples.")

        # Ensure few-shot examples exist in the dataset
        ds = DataStore(RunConfig.data["dir"])
        ds.read_csv_data(RunConfig.data["train"], separator="\t")
        train_data = ds.get_data()

        if len(train_data) < 3:
            logging.error("Not enough examples for few-shot learning.")
            raise ValueError("Few-shot learning requires at least 3 examples.")

        # Randomly sample 3 examples from the dataset
        few_shot_examples = random.sample(train_data.to_dicts(), 3)

        # Format examples as prompt input
        formatted_examples = "\n".join(
            [f"Sentence: \"{ex['sentence']}\"\nLabel: {ex['label']}" for ex in few_shot_examples]
        )

        # Inject few-shot examples into the prompt
        few_shot_template = f"{template}\n\nFew-Shot Examples:\n{formatted_examples}"
        prompt = ChatPromptTemplate.from_template(few_shot_template)
    else:
        logging.info("Few-shot learning is disabled. Using standard ChatPromptTemplate.")
        prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model
    return chain


def classify_sentence(chain: RunnableSequence, sentence: str):
    """
    Uses the provided chain to classify the sentence.
    The sentence text is injected into the prompt, and the LLM response is parsed.
    """
    prompt_input = {"sentence": sentence}
    response = chain.invoke(prompt_input)

    if response is None:
        logging.error(f"Failed to classify sentence: {sentence}")
        return None
    
    classification = extract_label(response.content)
    return classification


def compute_metrics(labels: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """simplified for binary classification task"""
    """Computes evaluation metrics for classification."""
    rounding = RunConfig.llm["round_results"]

    accuracy = round(accuracy_score(labels, predictions), rounding)
    precision = round(precision_score(labels, predictions, average="macro"), rounding)
    recall = round(recall_score(labels, predictions, average="macro"), rounding)
    f1 = round(f1_score(labels, predictions, average="macro"), rounding)

    return pl.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1_score"],
        "value": [accuracy, precision, recall, f1],
    })


def set_up_llm(api_key: str = None) -> BaseLLM:
    logging.info(f"Setting up model Provider: {RunConfig.llm['provider']} with model config: {RunConfig.llm['model_config']}")

    provider = RunConfig.llm["provider"]
    llm_config = RunConfig.llm["model_config"]
    if api_key:
        llm_config["api_key"] = api_key

    try:
        # If using OpenAI, import ChatOpenAI directly from langchain
        if provider == "openai":
            from langchain.chat_models import ChatOpenAI
            llm_class = ChatOpenAI
        else:
            # For other providers, remain dynamic
            llm_module = importlib.import_module(f"langchain_{provider}")
            llm_class = getattr(llm_module, RunConfig.llm["class_name"])
        
        configured_model = llm_class(**llm_config)
        return configured_model

    except ModuleNotFoundError as e:
        logging.error(f"Could not import module for provider '{provider}'. Error: {e}")
        raise

    except AttributeError as e:
        logging.error(f"Class '{RunConfig.llm['class_name']}' not found in the module for '{provider}'. Error: {e}")
        raise


def check_data_availability(path: Path) -> bool:
    """Check if data is already computed."""
    return path.exists()


def main():
    set_up_log()
    logging.info("Starting LLM Sentence Classification")

    try:
        args = init_args_parser()

        logging.info(f"Reading config {args.config_path}")
        RunConfig.load_config(Path(args.config_path))

        logging.info("Reading API Key from environment variable")
        model_provider = RunConfig.llm["provider"]
        if not os.environ.get(f"{model_provider.upper()}_API_KEY"):
            logging.error(f"API Key for {model_provider} is not set.")
            raise ValueError(f"API Key for {model_provider} is not set.")
        key = os.environ.get(f"{model_provider.upper()}_API_KEY")

        # Reading the training data
        ds = DataStore(RunConfig.data["dir"])
        ds.read_csv_data(RunConfig.data["train"], separator="\t")
        train_data = ds.get_data()

        output_path = Path(RunConfig.data["dir"]) / Path(RunConfig.data["train_pred"])
        if check_data_availability(output_path) and not args.force:
            # Load the existing results
            logging.info("Classification results exist. Skipping new classification.")
            ds.read_csv_data(RunConfig.data["train_pred"], separator="\t")
            results = ds.get_data()

        else:
            logging.info("Data not processed yet. Starting classification.")

            # Initialize the LLM
            model = set_up_llm(api_key=key)

            # Prepare the chain
            chain = prepare_chain(model)

            # Predictions with periodic saving every RunConfig.llm["save_every"] rows to prevent data loss
            predictions = []
            for i, row in enumerate(train_data.iter_rows(named=True)):
                sentence_id = row["sentence_id"]
                sentence = row["sentence"]
                label = classify_sentence(chain, sentence)
                predictions.append((sentence_id, sentence, label, row["solved_conflict"]))

                # Save intermediate results to output file at set intervals
                if (i + 1) % RunConfig.llm["save_every"] == 0:
                    partial_results = pl.DataFrame(predictions, schema=["sentence_id", "sentence", "label", "solved_conflict"])
                    partial_results.write_csv(output_path, separator="\t")
                    logging.info(f"Saved {i + 1} processed rows.")

            # Save final results
            results = pl.DataFrame(predictions, schema=["sentence_id", "sentence", "label", "solved_conflict"])
            results.write_csv(output_path, separator="\t")
            logging.info(f"Finished predictions. Sample output:\n{results.tail()}")

        # Compute the evaluation metrics
        metrics = compute_metrics(train_data["label"].sort(), results["label"].sort())
        logging.info(f"Metrics:\n{metrics}")
        logging.info(f"Finished prediction with {model_provider}.")
        
        return 0

    except Exception:
        logging.exception("Classification failed", stack_info=True)
        return 1


main()
