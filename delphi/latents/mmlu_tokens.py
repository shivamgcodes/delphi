"""
Build token tensors from `cais/mmlu` using continuation-style prompts.

This approximates the text distribution used by lm_eval tasks such as
``mmlu_{subject}_continuation`` in SLICE's ``mmlu_routing_analysis`` pipeline,
without running the full lm_eval harness.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

MMLU_REPO = "cais/mmlu"


def _format_mmlu_prompt(subject_cfg: str, question: str, choices: list | tuple) -> str:
    subj = subject_cfg.replace("_", " ")
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)}")
    a, b, c, d = choices[0], choices[1], choices[2], choices[3]
    return (
        f"The following are multiple choice questions about {subj}.\n\n"
        f"{question}\n"
        f"A. {a}\nB. {b}\nC. {c}\nD. {d}\n"
        f"Answer:"
    )


def mmlu_config_names() -> list[str]:
    from datasets import get_dataset_config_names

    return sorted(get_dataset_config_names(MMLU_REPO))


def load_mmlu_tokenized_data(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    *,
    subjects: str = "all",
    split: str = "test",
    seed: int = 42,
    convert_to_tensor_chunk_size: int = 2**18,
) -> torch.Tensor:
    """
    Load MMLU from Hugging Face, format as multiple-choice prompts, chunk to ``ctx_len``.

    Args:
        ctx_len: Sequence length per row (same as generic caching).
        tokenizer: Model tokenizer.
        subjects: Comma-separated subject config names (e.g. ``abstract_algebra``), or ``all``.
        split: ``test``, ``validation`` / ``val``, or ``dev`` (dataset-dependent; ``test`` is default).
        seed: Shuffle seed across concatenated subjects.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset
    from sparsify.data import chunk_and_tokenize

    if subjects.strip().lower() == "all":
        subject_list = mmlu_config_names()
    else:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

    parts = []
    for cfg_name in subject_list:
        ds = load_dataset(MMLU_REPO, cfg_name, split=split)

        texts = []
        for row in ds:
            ch = row["choices"]
            if hasattr(ch, "tolist"):
                ch = ch.tolist()
            texts.append(_format_mmlu_prompt(cfg_name, row["question"], list(ch)))

        parts.append(Dataset.from_dict({"text": texts}))

    if not parts:
        raise ValueError("No MMLU subjects loaded")

    data = concatenate_datasets(parts)
    data = data.shuffle(seed)

    tokens_ds = chunk_and_tokenize(
        data,
        tokenizer,
        max_seq_len=ctx_len,
        text_key="text",
    )

    tokens = tokens_ds["input_ids"]

    try:
        from datasets import Column
        from datasets.table import table_iter

        if isinstance(tokens, Column):
            tokens = torch.cat(
                [
                    torch.from_numpy(
                        np.stack(table_chunk["input_ids"].to_numpy(), axis=0)
                    )
                    for table_chunk in table_iter(
                        tokens.source._data, convert_to_tensor_chunk_size
                    )
                ]
            )
    except ImportError:
        if not hasattr(tokens, "shape") or len(tokens.shape) != 2:
            raise

    return tokens
