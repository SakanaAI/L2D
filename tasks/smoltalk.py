

from argparse import Namespace

from datasets import load_dataset
from transformers import AutoTokenizer

from pipeline import (
    DataProcessor,
    is_rank_0
)
from .constants import (
    SmolTalkMagpieUltraCategory2Id,
    SmolTalkMagpieUltraCategories,
    SmolTalkSources,
    SharedTasks,
    SmolTalkSources2Shared,
    SmolTalkMagpieUltraCategories2Shared
)
from .base import Task
from .templates import template_type2template


class SmolTalkDataProcessor(DataProcessor, Task):

    num_categories = len(SmolTalkSources)

    def __init__(
        self,
        config: Namespace,
        filepath: str,
        tokenizer: AutoTokenizer,
        **kwargs
    ) -> None:
        Task.__init__(
            self=self, config=config, filepath=filepath, tokenizer=tokenizer)

        self.subset = filepath.split(',')[1]
        self.task_name = "smoltalk-magpie-ultra"
        self.exclude_train_data_category = config.exclude_train_data_category
        self.do_exclude_data = self.exclude_train_data_category is not None
        if is_rank_0() and self.do_exclude_data:
            print('WARNING: excluding categories '
                  f'{self.exclude_train_data_category}')

    def initializer(self) -> None:
        pass

    def line2data(self, line: str) -> list:
        line_idx, line_data = line
        messages = line_data["messages"]

        if "category" in line_data:
            category = line_data["category"]
            if self.do_exclude_data:
                if category in self.exclude_train_data_category:
                    return []
        else:
            category = None

        category, category_id = self.get_category_and_id(
            sample_category=category)

        examples = []
        for idx in range(1, len(messages)):
            context = messages[:idx]
            response = messages[idx]
            response_role = response["role"]

            if response_role != "assistant":
                continue

            context = self.tokenizer.apply_chat_template(
                context,
                tokenize=True,
                add_generation_prompt=True,
                specified_template=self.specified_template
            )
            response = self.tokenizer.encode(
                response["content"],
                add_special_tokens=False
            )

            if len(context) + len(response) > self.max_seq_len:
                break

            examples.append({
                "context": context,
                "response": response,
                "category": category,
                "category_id": category_id,
                "task_name": self.task_name
            })

        return examples
