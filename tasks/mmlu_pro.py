

from argparse import Namespace
import re
import string
from typing import Union

from datasets import load_dataset
import jinja2
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .constants import (
    MMLUProCategory2Id,
    MMLUProCategories,
)
from .templates import template_type2template
from .base import Task
from .utils import extract_first_choice_answer


class MMLUProDataProcessor(DataProcessor, Task):
    num_categories = len(MMLUProCategories)

    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ) -> None:
        Task.__init__(
            self=self, config=config, filepath=filepath, tokenizer=tokenizer)

        self.options = string.ascii_uppercase

        self.question_template = (
            "Question:\n"
            "{{question}}\n"
            "Choices:\n"
            "{% for choice in choices %}"
            "{{options[loop.index-1]}}. {{choice}}\n"
            "{% endfor %}\n"
        )
        self.task_name = "mmlu_pro"

        dev_data = load_dataset(
            "TIGER-Lab/MMLU-Pro", split="validation", trust_remote_code=True)
        self.category2few_shot_examples = {}
        for i in range(len(dev_data)):
            example = dev_data[i]
            category = example["category"]
            if category not in self.category2few_shot_examples:
                self.category2few_shot_examples[category] = []
            self.category2few_shot_examples[category].append(example)

    def initializer(self) -> None:
        pass

    def line2data(self, line: dict) -> list[tuple[str, str]]:
        line_idx, line_data = line
        question = line_data["question"]
        choices = line_data["options"]
        answer = line_data["answer_index"]
        category = line_data["category"]
        category_id = MMLUProCategory2Id[category]
        jinja_template = jinja2.Template(self.question_template)

        conversation = []

        few_shot_examples = self.category2few_shot_examples[category]
        few_shot_examples = few_shot_examples[:self.num_few_shots]
        for example in few_shot_examples:
            fs_rendered_question = jinja_template.render(
                question=example["question"],
                choices=example["options"],
                options=self.options
            )
            fs_answer = example["answer_index"]
            fs_answer = self.options[fs_answer]
            conversation.append(
                {"role": "user", "content": fs_rendered_question})
            conversation.append({"role": "assistant", "content": fs_answer})

        rendered_question = jinja_template.render(
            question=question,
            choices=choices,
            options=self.options
        )
        conversation.append({"role": "user", "content": rendered_question})
        answer = self.options[answer]

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
            sample_category=category)
        return [{
            "context": context,
            "response": response,
            "category": category,
            "category_id": category_id,
            "output_space": self.options[:len(choices)],
            "task_name": self.task_name
        }]


class MMLUProEvalProcessor:
    @staticmethod
    def extract_answer_from_completion(
        completion: str,
        choices: Union[int, list, set, tuple] = 10
    ) -> Union[str, None]:
        num_choices = len(choices) if not isinstance(choices, int) else choices
        valid_choices = string.ascii_uppercase[:num_choices]

        matched_text, _ = extract_first_choice_answer(
            completion=completion, choices=valid_choices, return_index=False)
        return matched_text
