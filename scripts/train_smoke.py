"""Run a small, terminal-visible NumPy training smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ModelConfig,
    TrainingLoopConfig,
    encode_utf8,
    load_checkpoint,
    run_training,
    split_byte_tokens,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small NumPy language model with visible diagnostics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "smoke.json",
    )
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def _synthetic_text(context_length: int) -> str:
    pattern = "Yuzu learns byte patterns one careful step at a time.\n"
    minimum_characters = 12 * (context_length + 1)
    repeats = minimum_characters // len(pattern) + 1
    return pattern * repeats


def main() -> int:
    arguments = _arguments()
    loaded = load_checkpoint(arguments.resume) if arguments.resume else None
    model_config = (
        loaded.config if loaded is not None else ModelConfig.from_json(arguments.config)
    )
    if arguments.text_file is None:
        text = _synthetic_text(model_config.context_length)
        source_description = "built-in synthetic text"
    else:
        text = arguments.text_file.read_text(encoding="utf-8")
        source_description = str(arguments.text_file)
    split = split_byte_tokens(
        encode_utf8(text),
        validation_fraction=0.2,
        context_length=model_config.context_length,
    )
    loop_config = TrainingLoopConfig(
        max_steps=arguments.steps,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        validation_interval=arguments.validation_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        seed=arguments.seed,
    )
    run_directory = arguments.run_dir
    if run_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_directory = PROJECT_ROOT / "runs" / f"smoke-{timestamp}"

    parameters = None if loaded is None else loaded.parameters
    optimizer_state = None if loaded is None else loaded.optimizer_state
    rng = np.random.default_rng(arguments.seed)
    if loaded is not None:
        if loaded.rng_state is None:
            raise ValueError("resume checkpoint does not contain RNG state")
        rng.bit_generator.state = loaded.rng_state

    print(
        f"source={source_description} train_tokens={split.train.size} "
        f"validation_tokens={split.validation.size} run_dir={run_directory}"
    )
    result = run_training(
        split.train,
        split.validation,
        model_config,
        loop_config,
        run_directory,
        parameters=parameters,
        optimizer_state=optimizer_state,
        rng=rng,
    )
    print(
        f"complete step={result.optimizer_state.step:06d} "
        f"metrics={result.metrics_path} "
        f"checkpoint={result.checkpoint_paths[-1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
