import argparse
import importlib
import logging
import os
import random
import uuid
from pathlib import Path

import polars as pl
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate

from helper.logger import set_up_log
from helper.run_config import RunConfig


def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Data Augmentation")
    parser.add_argument("--config_path", type=str, default="config/open-ai.yml", help="Config file path")
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run (overwrite output) if it exists."
    )
    parser.add_argument(
        "--num_aug", type=int, default=1,
        help="Number of new sentences to generate per row."
    )
    return parser.parse_args()


def set_up_llm(api_key: str = None) -> BaseLLM:
    """Load and configure the LLM based on RunConfig."""
    logging.info(f"Setting up LLM with provider: {RunConfig.llm['provider']}")

    provider = RunConfig.llm["provider"]
    llm_config = RunConfig.llm["model_config"]
    if api_key:
        llm_config["api_key"] = api_key

    if provider == "openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(**llm_config)
    else:
        llm_module = importlib.import_module(f"langchain_{provider}")
        llm_class = getattr(llm_module, RunConfig.llm["class_name"])
        return llm_class(**llm_config)


def generate_augmented_sentence(chain, original: str, label: str) -> str:
    """
    Prompt the LLM to create a new sentence with the same label (SUBJ or OBJ),
    based on the original sentence's content.
    """
    prompt_input = {
        "original": original,
        "label": label,
    }
    response = chain.invoke(prompt_input)
    text = response.content.strip()
    text = text.strip('"')  # remove stray quotes
    return text


def main():
    set_up_log()
    logging.info("Starting LLM Data Augmentation")

    args = init_args_parser()
    RunConfig.load_config(Path(args.config_path))

    data_dir = Path(RunConfig.data["dir"])
    input_path = data_dir / RunConfig.data["train"]
    output_path = data_dir / RunConfig.data["train_pred"]

    if output_path.exists() and not args.force:
        logging.info("Augmented data already exists. Use --force to overwrite.")
        return 0

    # Check API key
    provider = RunConfig.llm["provider"]
    api_key = os.environ.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"API Key for {provider} is not set.")

    # Load original dataset
    df = pl.read_csv(input_path, separator="\t")

    # Initialize model
    model = set_up_llm(api_key=api_key)

    # Build chain with system + user instructions
    system_instructions = RunConfig.llm["prompt"].get("system", "")
    user_template = (
        "Original sentence: {original}\n"
        "It is labeled as {label}.\n"
        "Generate a new distinct sentence that would also be considered {label}.\n"
        "Do not just copy the original. Return only the new sentence.\n"
    )
    full_prompt = f"System: {system_instructions}\n\nUser: {user_template}"
    prompt = ChatPromptTemplate.from_template(full_prompt)
    chain = prompt | model

    # Generate new examples
    augmented_rows = []
    for row in df.iter_rows(named=True):
        original_id = row["sentence_id"]
        original_sent = row["sentence"]
        original_label = row["label"]
        solved_conflict = row.get("solved_conflict", False)

        for _ in range(args.num_aug):
            try:
                new_sentence = generate_augmented_sentence(chain, original_sent, original_label)
                new_id = str(uuid.uuid4())
                augmented_rows.append((new_id, new_sentence, original_label, solved_conflict))
            except Exception as e:
                logging.warning(f"Skipping row {original_id} due to error: {e}")

    # Option A: Combine original + new
    original_rows = []
    for row in df.iter_rows(named=True):
        original_rows.append((
            row["sentence_id"],
            row["sentence"],
            row["label"],
            row.get("solved_conflict", False)
        ))

    combined = original_rows + augmented_rows
    random.shuffle(combined)  # Mix them up

    out_df = pl.DataFrame(
        combined, schema=["sentence_id", "sentence", "label", "solved_conflict"]
    )
    out_df.write_csv(output_path, separator="\t")

    logging.info(f"Augmented dataset saved with {len(out_df)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        exit(code)
    except Exception as e:
        logging.exception("Data augmentation failed", stack_info=True)
        exit(1)
