import os
from typing import Callable

import torch
import wandb

from subjectivity.helper.run_config import RunConfig


def login_wandb() -> None:
    # Login with API key from environment
    wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)


def track_with_wandb(train_fn: Callable, *, run_name: str, **kwargs) -> tuple[torch.nn.Module, float]:
    login_wandb()

    wandb.init(
        project="checkthat-subject",
        name=run_name,
        config={
            "encoder_model": RunConfig.encoder_model,
            "train": RunConfig.train,
            "data": RunConfig.data,
        },
    )
    wandb.watch(kwargs["model"], log="all", log_freq=100)

    kwargs["log_fn"] = wandb.log  # 👈 Add logging to each epoch

    model, best_macro_f1, test_predictions = train_fn(**kwargs)

    wandb.log({"val/macro_f1": best_macro_f1})
    wandb.finish()

    return model, best_macro_f1, test_predictions
