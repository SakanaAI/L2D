

from argparse import Namespace
import re
import string
from typing import Union

from datasets import load_dataset
import jinja2
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .base import Task
from .templates import template_type2template
from .utils import extract_first_choice_answer


class ARCDataProcessor(DataProcessor, Task):
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

        self.question_template = (
            "Question:\n"
            "{{question}}\n"
            "Choices:\n"
            "{% for choice in choices %}"
            "{{choice[0]}}. {{choice[1]}}\n"
            "{% endfor %}\n"
            "Answer:"
        )

        dataset_id, subtask, split = self.filepath.split(",")
        dev_data = load_dataset(
            "allenai/ai2_arc",
            subtask,
            split="validation",
            trust_remote_code=True,
        )
        self.few_shot_examples = []
        for i in range(len(dev_data)):
            example = dev_data[i]
            self.few_shot_examples.append(example)

        if subtask == "ARC-Easy":
            self.task_name = "arc-easy"
        elif subtask == "ARC-Challenge":
            self.task_name = "arc-challenge"

    def initializer(self) -> None:
        pass

    def line2data(self, line: dict) -> list[tuple[str, str]]:
        line_idx, line_data = line
        question = line_data["question"]
        choice_labels = line_data["choices"]["label"]
        choice_texts = line_data["choices"]["text"]
        choices = list(zip(choice_labels, choice_texts))
        answer = line_data["answerKey"]
        jinja_template = jinja2.Template(self.question_template)

        conversation = []

        few_shot_examples = self.few_shot_examples[:self.num_few_shots]
        for example in few_shot_examples:
            example_choice_labels = example["choices"]["label"]
            example_choice_texts = example["choices"]["text"]
            example_choices = list(
                zip(example_choice_labels, example_choice_texts))
            fs_rendered_question = jinja_template.render(
                question=example["question"],
                choices=example_choices,
            )
            fs_answer = example["answerKey"]
            conversation.append(
                {"role": "user", "content": fs_rendered_question})
            conversation.append({"role": "assistant", "content": fs_answer})

        rendered_question = jinja_template.render(
            question=question,
            choices=choices,
        )
        conversation.append({"role": "user", "content": rendered_question})

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
            "output_space": choice_labels,
            "task_name": self.task_name
        }]


class ARCEvalProcessor:
    @staticmethod
    def extract_answer_from_completion(
        completion: str,
        choices: list[str]
    ) -> Union[str, None]:
        if len(completion) == 0:
            return None
        matched_text, _ = extract_first_choice_answer(
            completion=completion, choices=choices, return_index=False)
        return matched_text
