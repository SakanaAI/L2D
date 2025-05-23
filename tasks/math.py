

from argparse import Namespace
import re
from typing import Union
import typing as tp
from datasets import load_dataset
from transformers import AutoTokenizer
from .utils import (extract_last_number, extract_last_boxed_only_string, remove_boxed,
                    strip_math_latex_string,)

from pipeline import DataProcessor
from .base import Task
from .templates import template_type2template


class MATHDataProcessor(DataProcessor, Task):
    num_categories = 1

    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ) -> None:
        Task.__init__(
            self=self, config=config, filepath=filepath, tokenizer=tokenizer)

        self.task_name = "MATH"

        training_data = load_dataset(
            "lightevalMATH", split="train")

        num_training = len(training_data)

        few_shot_idxs_step = num_training // self.num_few_shots
        self.few_shot_examples = []

        for i in range(self.num_few_shots):
            example = training_data[i*few_shot_idxs_step]
            self.few_shot_examples.append(example)

    def initializer(self) -> None:
        pass

    def line2data(self, line) -> list[tuple[str, str]]:
        line_idx, line_data = line
        question = line_data["problem"]
        answer = line_data["solution"]

        conversation = []

        for example in self.few_shot_examples:
            conversation.append(
                {"role": "user", "content": example["problem"]})
            conversation.append(
                {"role": "assistant", "content": example["solution"]})

        conversation.append({"role": "user", "content": question})

        context = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            chat_template=self.specified_template
        )
        response = self.tokenizer.encode(
            answer,
            add_special_tokens=False
        )
        if len(context) + len(response) > self.max_seq_len:
            return None

        category, category_id = self.get_category_and_id(
            sample_category=None)
        return [{
            "context": context,
            "response": response,
            "category": category,
            "category_id": category_id,
            "output_space": None,
            "task_name": self.task_name
        }]


class MATHEvalProcessor:
    @staticmethod
    def extract_answer_from_completion(
        completion: str,
        **kwargs
    ) -> Union[str, None]:
        match_boxed = extract_last_boxed_only_string(completion=completion)
        if match_boxed is not None:
            answer = remove_boxed(match_boxed)
            if isinstance(answer, str):
                answer = strip_math_latex_string(answer)
            return answer
        else:
            answer = extract_last_number(completion=completion)
            return answer
