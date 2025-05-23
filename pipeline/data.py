

from argparse import Namespace
import multiprocessing
from typing import (
    List,
    Type
)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pipeline.util import is_rank_0


class BasicDataSource(torch.utils.data.Dataset):
    def __init__(
        self,
        stage: str,
        data: List,
        tokenizer: object,
        config: Namespace
    ):

        self.stage = stage
        self.tokenizer = tokenizer
        self.stat = {}

        self.data = data
        self.stat["n_data"] = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch_data):
        return batch_data


class DataProcessor(object):
    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        self.config = config
        self.filepath = filepath
        self.tokenizer = tokenizer

    def initializer(self):
        raise NotImplementedError

    def line2data(self, line):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class DataReformatter(DataProcessor):
    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        super().__init__(
            config=config,
            filepath=filepath,
            tokenizer=tokenizer,
            **kwargs
        )

    def initializer(self):
        pass

    def format_data(self, data):
        raise NotImplementedError


class IdentityDataProcessor(DataProcessor):
    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        super().__init__(
            config=config,
            filepath=filepath,
            tokenizer=tokenizer,
            **kwargs
        )

    def initializer(self):
        pass

    def line2data(self, line):
        return line


def load_data_from_jsonl(
    tokenizer: AutoTokenizer,
    stage: str,
    filepaths: List[str],
    data_processor_classes: List[Type[DataProcessor]],
    data_reformatter_class: Type[DataProcessor] = None,
    config: Namespace = None,
    one_dataset_per_input: bool = False,
    **kwargs
):
    debug_mode = config.debug_mode
    debug_mode_data_size = config.debug_mode_data_size
    multiple_samples_per_jsonl_line = config.multiple_samples_per_jsonl_line

    data_filepath2dataset = {}
    assert len(data_processor_classes) == len(filepaths)

    if data_reformatter_class is not None:
        reformatter = data_reformatter_class(
            config=config,
            tokenizer=tokenizer,
            filepath=filepath,
            **kwargs
        )

    for data_processor_class, filepath in zip(data_processor_classes, filepaths):
        dataset = []
        reformatted_dataset = []
        data_processor = data_processor_class(
            config=config,
            tokenizer=tokenizer,
            filepath=filepath,
            **kwargs
        )
        with open(filepath, encoding="utf-8") as f:
            if debug_mode:
                data_processor.initializer()
                for line in tqdm(enumerate(f), desc=f"Loading {stage} data from {filepath}", disable=not is_rank_0()):
                    result = data_processor.line2data(line)
                    if result is not None:
                        if multiple_samples_per_jsonl_line:
                            dataset.extend(result)
                        else:
                            dataset.append(result)
                    if len(dataset) >= debug_mode_data_size:
                        break
            else:
                with multiprocessing.Pool(8, initializer=data_processor.initializer) as pool:
                    results = pool.imap(
                        data_processor.line2data, enumerate(f), chunksize=1000)
                    for result in tqdm(results, desc=f"Loading {stage} data from {filepath}", disable=not is_rank_0()):
                        if result is not None:
                            if multiple_samples_per_jsonl_line:
                                dataset.extend(result)
                            else:
                                dataset.append(result)
        if is_rank_0():
            print(f"Loaded {len(dataset)} data from {filepath}")

        if data_reformatter_class is not None:
            if debug_mode:
                reformatter.initializer()
                for item in tqdm(dataset, desc="Reformatting data", disable=not is_rank_0()):
                    result = reformatter.format_data(item)
                    if result is not None:
                        reformatted_dataset.append(result)
            else:
                with multiprocessing.Pool(8, initializer=reformatter.initializer) as pool:
                    results = pool.imap(
                        reformatter.format_data, dataset, chunksize=1000)
                    for result in tqdm(results, desc="Reformatting data", disable=not is_rank_0()):
                        if result is not None:
                            reformatted_dataset.append(result)
            if is_rank_0():
                print(
                    f"Reformatted {len(reformatted_dataset)} data from {filepath}")

        if data_reformatter_class is None:
            data_filepath2dataset[filepath] = dataset
        else:
            data_filepath2dataset[filepath] = reformatted_dataset

    if not one_dataset_per_input:
        data_filepath2dataset = {
            "all": sum(data_filepath2dataset.values(), [])
        }

    return data_filepath2dataset
