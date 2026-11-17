"""SFT warm-up on collected rollouts (OpenEnv "Collect Rollouts for SFT" tutorial), then GRPO.

    uv run python -m openenv_daw.runner collect --provider openai --model gpt-5-mini -n 300 -o rollouts
    uv run python -m openenv_daw.runner export rollouts -o sft.jsonl
    uv run python examples/sft_warmup.py --data sft.jsonl --model Qwen/Qwen3-1.7B
    uv run python examples/grpo_train.py --model runs/daw-sft

The goal is format compliance (valid tool calls); mastery comes from GRPO.
"""

import argparse
import json

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="runs/daw-sft")
    a = ap.parse_args()

    rows = [json.loads(x) for x in open(a.data) if x.strip()]
    rows = [r for r in rows if r["reward"] == 1.0]  # the environment's reward labels the data
    for r in rows:  # tool-call arguments as mappings, as chat templates expect
        for m in r["messages"]:
            for tc in m.get("tool_calls") or []:
                if isinstance(tc["function"]["arguments"], str):
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
    tok = AutoTokenizer.from_pretrained(a.model)
    lengths = [
        len(tok.apply_chat_template(r["messages"], tools=json.loads(r["tools"]), tokenize=True))
        for r in rows
    ]
    max_len = int(np.percentile(lengths, 99)) + 16  # truncate < 1% of examples
    print(f"{len(rows)} examples, p50={np.percentile(lengths, 50):.0f} max_length={max_len}")

    trainer = SFTTrainer(
        model=a.model,
        train_dataset=Dataset.from_list(
            [{"messages": r["messages"], "tools": r["tools"]} for r in rows]
        ),
        processing_class=tok,
        args=SFTConfig(
            output_dir=a.out,
            max_length=max_len,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="no",
            assistant_only_loss=True,  # loss on the tool calls, not the prompt
        ),
    )
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
