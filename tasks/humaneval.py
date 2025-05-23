

from argparse import Namespace
import re
from typing import Union

from datasets import load_dataset
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .base import Task
from .templates import template_type2template


class HumanEvalDataProcessor(DataProcessor, Task):
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

        self.task_name = "humaneval"

        training_data = load_dataset(
            "openai/openai_humaneval", split="test", trust_remote_code=True)
        self.few_shot_examples = []
        for i in range(self.num_few_shots):
            example = training_data[i]
            self.few_shot_examples.append(example)

    def initializer(self) -> None:
        pass

    def line2data(self, line) -> list[tuple[str, str]]:
        line_idx, line_data = line
        task_id = line_data["task_id"]
        question = line_data["prompt"]
        answer = line_data["canonical_solution"]

        conversation = []

        for example in self.few_shot_examples:

            if example["prompt"] == question:
                return None
            conversation.append({"role": "user", "content": example["prompt"]})
            conversation.append(
                {"role": "assistant", "content": example["canonical_solution"]})

        conversation.append({"role": "user", "content": question})

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

        return [{
            "context": context,
            "response": response,
            "category": None,
            "category_id": None,
            "output_space": None,
            "task_id": task_id,
            "task_name": self.task_name,
            "test_list": None
        }]


class InstructHumanEvalDataProcessor(DataProcessor):
    num_categories = 1

    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ) -> None:
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
        self.num_few_shots = config.num_few_shots
        if config.specified_template_type:
            self.specified_template = template_type2template[
                config.specified_template_type]
        else:
            self.specified_template = None
        self.task_name = "instructhumaneval"

        training_data = load_dataset(
            "codeparrot/instructhumaneval", split="test",
            trust_remote_code=True)
        self.few_shot_examples = []
        for i in range(self.num_few_shots):
            example = training_data[i]
            self.few_shot_examples.append(example)

    def initializer(self) -> None:
        pass

    def line2data(self, line) -> list[tuple[str, str]]:
        line_idx, line_data = line
        task_id = line_data["task_id"]
        question = line_data["prompt"]
        user_question = line_data["instruction"]
        agent_partial_completion = line_data["context"]
        answer = line_data["canonical_solution"]

        conversation = []

        for example in self.few_shot_examples:

            if example["prompt"] == question:
                return None
            conversation.append(
                {"role": "user", "content": example["instruction"]})
            conversation.append(
                {"role": "assistant",
                 "content": example["context"] + example["canonical_solution"]})

        conversation.append({"role": "user", "content": user_question})

        conversation.append(
            {"role": "assistant", "content": agent_partial_completion})

        context = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            chat_template=self.specified_template,
            continue_final_message=True
        )

        response = self.tokenizer.encode(
            answer,
            add_special_tokens=False
        )
        if len(context) + len(response) > self.max_seq_len:
            return None

        return [{
            "context": context,
            "response": response,
            "category": None,
            "category_id": None,
            "output_space": None,
            "task_id": task_id,
            "task_name": self.task_name,
            "test_list": None
        }]
