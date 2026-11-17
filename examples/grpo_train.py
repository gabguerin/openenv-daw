"""GRPO with TRL's environment_factory on the in-process simulator.

    uv sync --extra trl
    uv run python examples/grpo_train.py --model Qwen/Qwen3-1.7B

Warm start (recommended): collect with a teacher, export, run ``examples/sft_warmup.py``, then
pass the SFT checkpoint as ``--model`` so ``reward_std`` is non-zero from the first batch.

Reward comes from DawToolEnv.get_reward (the episode score). Evaluate on the heldout split.
"""

import argparse

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from openenv_daw.training.trl_env import DawToolEnv, build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--out", default="runs/daw-grpo")
    a = ap.parse_args()

    args = GRPOConfig(
        output_dir=a.out,
        num_generations=8,
        per_device_train_batch_size=8,
        max_completion_length=2048,
        learning_rate=1e-6,
        beta=0.0,
        chat_template_kwargs={"enable_thinking": False},  # answer with tool calls directly
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=a.model,
        args=args,
        train_dataset=Dataset.from_list(build_dataset("train", a.n_train)),
        eval_dataset=Dataset.from_list(build_dataset("heldout", 65, stratified=True)),
        environment_factory=DawToolEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
