

from argparse import Namespace
import multiprocessing
from typing import Type

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from pipeline import (
    DataProcessor,
    is_rank_0
)


def load_conversation_data_from_hf(
    tokenizer: AutoTokenizer,
    stage: str,
    filepaths: list[str],
    data_processor_classes: list[Type[DataProcessor]],
    config: Namespace = None,
    one_dataset_per_input: bool = False,
    load_in_notebook: bool = False,
    **kwargs
) -> dict[str, list]:
    debug_mode = config.debug_mode
    debug_mode_data_size = config.debug_mode_data_size

    data_filepath2dataset = {}
    assert len(data_processor_classes) == len(filepaths)

    for data_processor_class, filepath in zip(
        data_processor_classes,
        filepaths
    ):
        data_processor = data_processor_class(
            config=config,
            tokenizer=tokenizer,
            filepath=filepath,
            **kwargs
        )
        dataset_name_items = filepath.split(",")
        if len(dataset_name_items) == 3:
            dataset_id, subtask, split = dataset_name_items
            _dataset = load_dataset(dataset_id, subtask, split=split)
        elif len(dataset_name_items) == 2:
            dataset_id, split = dataset_name_items
            _dataset = load_dataset(dataset_id, split=split)
        dataset = []
        if load_in_notebook:
            data_processor.initializer()
            for line in tqdm(
                    enumerate(_dataset),
                    desc=f"Loading {stage} data from {filepath}",):

                result = data_processor.line2data(line)
                if result is not None:
                    dataset.extend(result)

                if len(dataset) >= 64:
                    break
        elif debug_mode:
            data_processor.initializer()
            for line in tqdm(
                enumerate(_dataset),
                desc=f"Loading {stage} data from {filepath}",
                disable=not is_rank_0()
            ):
                result = data_processor.line2data(line)
                if result is not None:
                    dataset.extend(result)
                if len(dataset) >= debug_mode_data_size:
                    break
        else:
            with multiprocessing.Pool(
                8, initializer=data_processor.initializer
            ) as pool:
                results = pool.imap(
                    data_processor.line2data,
                    enumerate(_dataset),
                    chunksize=1000
                )
                for result in tqdm(
                    results,
                    desc=f"Loading {stage} data from {filepath}",
                    disable=not is_rank_0()
                ):
                    if result is not None:
                        dataset.extend(result)
        if load_in_notebook:
            print(f"Loaded {len(dataset)} data from {filepath}")
        elif is_rank_0():
            print(f"Loaded {len(dataset)} data from {filepath}")
        data_filepath2dataset[filepath] = dataset

    if not one_dataset_per_input:
        data_filepath2dataset = {
            "all": sum(data_filepath2dataset.values(), [])
        }

    return data_filepath2dataset
