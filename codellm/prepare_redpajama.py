# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import numpy as np
from tqdm import tqdm

from codellm.tokenizer import Tokenizer
import codellm.packed_dataset as packed_dataset


filenames_sample = [
    "arxiv_sample.jsonl",
    "book_sample.jsonl",
    "c4_sample.jsonl",
    "cc_2019-30_sample.jsonl",
    "cc_2020-05_sample.jsonl",
    "cc_2021-04_sample.jsonl",
    "cc_2022-05_sample.jsonl",
    "cc_2023-06_sample.jsonl",
    "github_sample.jsonl",
    "stackexchange_sample.jsonl",
    "wikipedia_sample.jsonl",
]


def prepare_sample(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    match="",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}. \n"
                "Make sure you download the data, e.g. wget -i https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through \n"
                "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T \n"
                "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        prefix, _ = os.path.splitext(name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                text_array = np.asarray(text_ids, dtype=builder.dtype)
                builder.add_array(text_array)

        builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/tokenizer/tokenizer.model"),
    destination_path: Path = Path("data/redpajama_sample"),
    chunk_size: int = 2049
    * 1024,  # 2048 block size + 1 for causal (from LLama), 1024 blocks
    match: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""

    prepare_sample(
        source_path=source_path,
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=chunk_size,
        match=match,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
