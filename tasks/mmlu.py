

from argparse import Namespace
import re
import string
from typing import Union
from datasets import load_dataset
import jinja2
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .base import Task
from .constants import (
    MMLUCategory2Id,
    MMLUCategories,
)
from .templates import template_type2template
from .base import Task
from .utils import extract_first_choice_answer


class MMLUDataProcessor(DataProcessor, Task):
    num_categories = len(MMLUCategories)

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
            "Answer:"
        )

        dataset_id, subtask, split = self.filepath.split(",")

        if subtask == "all":
            self.task_name = "mmlu"
        else:
            self.task_name = f"mmlu/{subtask}"

        dev_data = load_dataset(
            "cais/mmlu", subtask, split="dev", trust_remote_code=True)
        self.subject2few_shot_examples = {}
        for i in range(len(dev_data)):
            example = dev_data[i]
            subject = example["subject"]
            if subject not in self.subject2few_shot_examples:
                self.subject2few_shot_examples[subject] = []
            self.subject2few_shot_examples[subject].append(example)

    def initializer(self) -> None:
        pass

    def line2data(self, line: dict) -> list[tuple[str, str]]:
        line_idx, line_data = line
        question = line_data["question"]
        choices = line_data["choices"]
        answer = line_data["answer"]
        subject = line_data["subject"]
        subject_id = MMLUCategory2Id[subject]
        jinja_template = jinja2.Template(self.question_template)

        conversation = []

        few_shot_examples = self.subject2few_shot_examples[subject]
        few_shot_examples = few_shot_examples[:self.num_few_shots]
        for example in few_shot_examples:
            fs_rendered_question = jinja_template.render(
                question=example["question"],
                choices=example["choices"],
                options=self.options
            )
            fs_answer = example["answer"]
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
            sample_category=subject)
        return [{
            "context": context,
            "response": response,
            "category": category,
            "category_id": category_id,
            "output_space": self.options[:len(choices)],
            "task_name": self.task_name
        }]


class MMLUEvalProcessor:
    @staticmethod
    def extract_answer_from_completion(
        completion: str,
        choices: Union[int, list, set, tuple] = 4
    ) -> Union[str, None]:
        num_choices = len(choices) if not isinstance(choices, int) else choices
        valid_choices = string.ascii_uppercase[:num_choices]

        matched_text, _ = extract_first_choice_answer(
            completion=completion, choices=valid_choices, return_index=False)
        return matched_text
