

from abc import ABC

from argparse import Namespace
from transformers import AutoTokenizer

from .templates import template_type2template
from .constants import get_shared_category_dictionary


class Task(ABC):
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
        if 'num_few_shots' in config:
            self.num_few_shots = config.num_few_shots
        else:
            self.num_few_shots = None
        if config.lad:
            self.use_data_split_guidance = config.use_data_split_guidance
        else:
            self.use_data_split_guidance = False
        if config.specified_template_type:
            self.specified_template = template_type2template[
                config.specified_template_type]
        else:
            self.specified_template = None

        if config.use_data_split_guidance:
            if config.guidance_categories is not None:
                if len(config.guidance_categories) > 0:
                    (self.category2id, self.data_category2id,
                     self.data_category2shared,
                     ) = get_shared_category_dictionary(
                         config=config, processor_filepath=filepath)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def get_category_and_id(self, sample_category):
        if self.use_data_split_guidance:
            category = self.data_category2shared[sample_category]
            id = self.data_category2id[sample_category]
            return category, id
        else:
            return sample_category, None
