import argparse, logging, uuid, asyncio, re
from pathlib import Path
from typing import List
from tqdm import tqdm

import polars as pl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from subjectivity.helper.logger     import set_up_log
from subjectivity.helper.run_config import RunConfig

def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrastive Style Augmentation with Few-Shot Prompting"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/contrastive_openai.yml"
    )
    parser.add_argument(
        "--force",
        action="store_true"
    )
    parser.add_argument(
        "--SO_num_paraphrases",
        type=int,
        help="Number of objective paraphrases per subjective sentence"
    )
    parser.add_argument(
        "--OS_num_paraphrases",
        type=int,
        help="Number of styled paraphrases per objective sentence"
    )
    # New style selection argument
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=["emotional", "propaganda", "prejudices", "partisan", "derogatory", "exaggerated"],
        help="Styles to apply: emotional/propaganda/prejudices/partisan/derogatory/exaggerated",
        metavar="STYLE"
    )
    return parser.parse_args()


def build_fewshot_prompt(original: str,
                         examples: str,
                         task_type: str,
                         style: str = None,
                         num_paraphrases: int = 1) -> ChatPromptTemplate:
    """Construct few-shot prompt with dynamic instructions"""
    messages = [("system", RunConfig.llm["prompt"]["system"])]

    for example in examples.split("---"):
        if example.strip():
            messages.append(("user", example.strip()))

    if task_type == "subj_to_obj":
        instruction = (
            f'Subjective: {{original}}\nObjective:\nGive {num_paraphrases} clearly distinct rephrasings.'
        )
    else:
        instruction = (
            f'Objective: {{original}}\n{style.capitalize()}:\nGive {num_paraphrases} clearly distinct rephrasings.'
        )

    messages.append(("human", instruction))
    return ChatPromptTemplate.from_messages(messages).partial(original=original)

def configure_model(task_type: str, style: str | None, n: int) -> ChatOpenAI:
    """Create model with task-specific parameters"""
    base_cfg = RunConfig.llm["model_config"].copy()

    # Remove parameters that will be explicitly set
    model_name = base_cfg.pop("model_name")
    base_cfg.pop("temperature", None)

    # Select style/task-specific temperature
    if task_type == "subj_to_obj":
        temperature = RunConfig.llm["subj_to_obj"].get("temperature", 0.85)
    else:
        temperature = RunConfig.llm["obj_to_subj"]["style_configs"][style].get("temperature", 0.85)

    return ChatOpenAI(
        **base_cfg,
        model_name=model_name,
        temperature=temperature,
        n=1
    )

def extract_numbered_paraphrases(text: str) -> List[str]:
    """Extract paraphrases from numbered list or bullet-style format."""
    matches = re.findall(r'\d+\.\s*"?(.*?)"?\s*(?=\d+\.|$)', text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]

def clean_prefix(line: str) -> str:
    """Remove any style or label prefix like 'Objective:', 'Emotional:', etc., and excess quotes."""
    return re.sub(
        r'^(Subjective|Objective|Emotional|Propaganda|Prejudices|Partisan|Derogatory|Exaggerated):\s*["“]*',
        '',
        line,
        flags=re.IGNORECASE
    ).strip(' "\'“”')

async def generate_paraphrases(original: str,
                               task_type: str,
                               style: str | None,
                               num_paraphrases: int) -> List[str]:
    """Generate multiple paraphrases and return them as distinct strings."""
    examples = (
        RunConfig.llm["subj_to_obj"]["examples"]
        if task_type == "subj_to_obj"
        else RunConfig.llm["obj_to_subj"]["style_configs"][style]["examples"]
    )

    # Generate prompt with n-output instruction
    prompt_template = build_fewshot_prompt(original, examples, task_type, style, num_paraphrases)
    prompt = prompt_template.format_messages()

    model = configure_model(task_type, style, n=1)

    try:
        res = await model.agenerate([prompt])
        raw_text = res.generations[0][0].message.content.strip()

        # Clean output
        outputs = [clean_prefix(m) for m in extract_numbered_paraphrases(raw_text)]
        if not outputs:
            outputs = [clean_prefix(line) for line in raw_text.splitlines() if line.strip()]
        return outputs[:num_paraphrases]

    except Exception as e:
        logging.error(f"Generation failed for '{original[:40]}…': {e}")
        return []


async def main_async() -> None:
    # 1. logging setup
    set_up_log()
    for h in logging.getLogger().handlers:
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.info("Starting Contrastive Augmentation Pipeline")

    # 2. CLI + config
    args = init_args_parser()
    RunConfig.load_config(Path(args.config_path))

    # 3. paths
    data_dir    = Path(RunConfig.data["dir"])
    input_path  = data_dir / RunConfig.data["train"]
    output_path = data_dir / RunConfig.data["train_pred"]

    if output_path.exists() and not args.force:
        logging.info("Output exists – use --force to overwrite")
        return

    # 4. load dataset
    df = pl.read_csv(input_path, separator="\t")
    augmented_rows: list[dict] = []

    # 5. CLI overrides
    so_num = args.SO_num_paraphrases or RunConfig.llm["subj_to_obj"]["num_paraphrases"]
    os_num = (args.OS_num_paraphrases
              if args.OS_num_paraphrases is not None
              else RunConfig.llm["obj_to_subj"]["num_paraphrases"])

    avail_styles = RunConfig.llm["obj_to_subj"]["styles"]
    styles = args.styles or avail_styles
    if bad := set(styles) - set(avail_styles):
        logging.error(f"Invalid styles: {', '.join(bad)}")
        return

    # 6. main loop
    checkpoint_every = 100
    with tqdm(total=len(df), desc="Processing sentences") as bar:
        for row in df.iter_rows(named=True):
            sent, label, sid, solved = (row["sentence"],
                                        row["label"],
                                        row["sentence_id"],
                                        row["solved_conflict"])

            # always include the original
            augmented_rows.append({
                "sentence_id": sid,
                "sentence": sent,
                "label": label,
                "solved_conflict": solved,
                "style": "original"
            })

            try:
                if label == "SUBJ":  # generate OBJ from SUBJ
                    paras = await generate_paraphrases(sent, "subj_to_obj", None, so_num)
                    for para in paras:
                        augmented_rows.append({
                            "sentence_id": str(uuid.uuid4()),
                            "sentence": para,
                            "label": "OBJ",
                            "solved_conflict": solved,
                            "style": "neutral",
                        })

                elif label == "OBJ":  # generate SUBJ from OBJ
                    # Create coroutine tasks for each style
                    tasks = [
                        asyncio.create_task(generate_paraphrases(sent, "obj_to_subj", sty, os_num))
                        for sty in styles
                    ]
                    # Gather results
                    results = await asyncio.gather(*tasks)

                    # Append outputs, keeping style aligned
                    for sty, paras in zip(styles, results):
                        for para in paras:
                            augmented_rows.append({
                                "sentence_id": str(uuid.uuid4()),
                                "sentence": para,
                                "label": "SUBJ",
                                "solved_conflict": solved,
                                "style": sty,
                            })

            except Exception as e:
                logging.error(f"Failed processing: {sent[:50]}… → {e}")

            bar.update(1)

            if len(augmented_rows) % checkpoint_every == 0:
                pl.DataFrame(augmented_rows).write_csv(output_path, separator="\t")
                logging.info(f"Checkpoint saved with {len(augmented_rows)} rows")

    # 7. final write‑out
    pl.DataFrame(augmented_rows).write_csv(output_path, separator="\t")
    logging.info(f"Final: {len(augmented_rows)} rows → {output_path}")


def main() -> None:
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.exception("Fatal error in augmentation pipeline")
        exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error in augmentation pipeline")
        exit(1)