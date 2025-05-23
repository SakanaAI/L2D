

from argparse import Namespace
import re
import string
from typing import Union

from datasets import load_dataset
import jinja2
from transformers import AutoTokenizer

from pipeline import DataProcessor
from .templates import template_type2template
from .base import Task
from .utils import extract_first_choice_answer


class PIQADataProcessor(DataProcessor, Task):
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

        self.options = string.ascii_uppercase

        self.question_template = (
            "Goal:\n"
            "{{question}}\n"
            "Options:\n"
            "{% for choice in choices %}"
            "{{options[loop.index-1]}}. {{choice}}\n"
            "{% endfor %}\n"
            "Answer:"
        )
        self.task_name = "piqa"

        dev_data = load_dataset(
            "ybisk/piqa", split="train", trust_remote_code=True)
        self.few_shot_examples = []
        for i in range(len(dev_data)):
            example = dev_data[i]
            self.few_shot_examples.append(example)

    def initializer(self) -> None:
        pass

    def line2data(self, line: dict) -> list[tuple[str, str]]:
        line_idx, line_data = line
        question = line_data["goal"]
        choices = [line_data["sol1"], line_data["sol2"]]
        answer = line_data["label"]
        jinja_template = jinja2.Template(self.question_template)

        conversation = []

        few_shot_examples = self.few_shot_examples[:self.num_few_shots]
        for example in few_shot_examples:
            fs_rendered_question = jinja_template.render(
                question=example["goal"],
                choices=[example["sol1"], example["sol2"]],
                options=self.options
            )
            fs_answer = example["label"]
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
            sample_category=None)
        return [{
            "context": context,
            "response": response,
            "category": category,
            "category_id": category_id,
            "output_space": [self.options[0], self.options[1]],
            "task_name": self.task_name
        }]


class PIQAEvalProcessor:
    @staticmethod
    def extract_answer_from_completion(
        completion: str,
        choices: Union[int, list, set, tuple] = 2
    ) -> Union[str, None]:
        num_choices = len(choices) if not isinstance(choices, int) else choices
        valid_choices = string.ascii_uppercase[:num_choices]

        valid_choices = string.ascii_uppercase[:num_choices]

        matched_text, _ = extract_first_choice_answer(
            completion=completion, choices=valid_choices, return_index=False)
        return matched_text
