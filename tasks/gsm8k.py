

from argparse import Namespace
import re
from typing import Union

from datasets import load_dataset
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .base import Task
from .templates import template_type2template


class GSM8kDataProcessor(DataProcessor, Task):
    num_categories = 1

    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs,
    ) -> None:
        Task.__init__(
            self=self, config=config, filepath=filepath, tokenizer=tokenizer)
        self.task_name = "gsm8k"

        dataset_id, subtask, split = self.filepath.split(",")
        training_data = load_dataset(
            "openai/gsm8k", subtask, split="train", trust_remote_code=True)
        self.few_shot_examples = []
        for i in range(self.num_few_shots):
            example = training_data[i]
            self.few_shot_examples.append(example)
        conversation = []
        for example in self.few_shot_examples[:1]:
            conversation.append(
                {"role": "user", "content": example["question"]})

        print('Sample context (make sure date is correct)')
        context = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            chat_template=self.specified_template
        )
        print(tokenizer.decode(context))

    def initializer(self) -> None:
        pass

    def line2data(self, line) -> list[tuple[str, str]]:
        line_idx, line_data = line
        question = line_data["question"]
        answer = line_data["answer"]

        conversation = []

        for example in self.few_shot_examples:
            conversation.append(
                {"role": "user", "content": example["question"]})
            conversation.append(
                {"role": "assistant", "content": example["answer"]})

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


class GSM8kEvalProcessor:
    @staticmethod
    def extract_answer_from_completion(





        completion: str,
        **kwargs,
    ) -> Union[str, None]:
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")

            return match_str
        else:
            return None
