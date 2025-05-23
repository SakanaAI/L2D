

from argparse import Namespace
import re
from typing import Union

from datasets import load_dataset
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .base import Task
from .templates import template_type2template


class MBPPDataProcessor(DataProcessor, Task):
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
        self.task_name = 'mbpp'

        prompt_data = load_dataset(
            "google-research-datasets/mbpp", "full", split="prompt")
        self.few_shot_examples = []
        for i in range(self.num_few_shots):
            example = prompt_data[i]
            self.few_shot_examples.append(example)

    def initializer(self) -> None:
        pass

    def get_prompt(self, line):

        description = line["text"]
        test_example = line["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt

    def line2data(self, line) -> list[tuple[str, str]]:
        line_idx, line_data = line
        task_id = line_data["task_id"]
        prompt = self.get_prompt(line_data)
        answer = line_data["code"]
        test_list = line_data["test_list"]

        conversation = []

        for example in self.few_shot_examples:
            example_prompt = self.get_prompt(example)
            example_answer = example["code"]

            if example_prompt == prompt:
                return None
            conversation.append({"role": "user", "content": example_prompt})
            conversation.append(
                {"role": "assistant", "content": example_answer})

        prompt = self.get_prompt(line_data)
        conversation.append({"role": "user", "content": prompt})

        context = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            chat_template=self.specified_template,
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
            "task_id": task_id,
            "task_name": self.task_name,
            "test_list": test_list
        }]
