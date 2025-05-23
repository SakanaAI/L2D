

from argparse import Namespace
from collections import defaultdict
import types
import logging
import os
import random
from typing import Dict

import deepspeed
from deepspeed.ops.adam import (
    DeepSpeedCPUAdam,
    FusedAdam
)
import numpy as np
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType
)
import tokenizers

import torch
from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def set_logging_config(log_filepath: str = None):
    handlers = [logging.StreamHandler()]
    if log_filepath is not None:
        with open(log_filepath, "w"):
            pass
        handlers.append(logging.FileHandler(log_filepath))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers
    )


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_number_of_parameters(
    model: torch.nn.Module
):

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return {
        "trainable params": trainable_params,
        "all params": all_param,
        "trainable%": 100 * trainable_params / all_param,
    }


def is_rank_0():

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def add_pad_token(tokenizer, model, reserved_token_idx=0):
    new_created_token = tokenizer.add_special_tokens(
        {"pad_token": f"<|reserved_special_token_{reserved_token_idx}|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    return new_created_token


def build_tokenizer(
    tokenizer_dir: str = None,
    use_fast: bool = False,
    use_tokenizers: bool = False,
    model: PreTrainedModel = None,
    **kwargs
):
    if not tokenizer_dir:
        return None

    if use_tokenizers:
        tokenizer = tokenizers.Tokenizer.from_file(
            os.path.join(
                tokenizer_dir,
                "tokenizer.json"
            )
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=use_fast,
            trust_remote_code=True
        )

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(
            tokenizer, LlamaTokenizerFast) or 'llama' in tokenizer_dir.lower():

        if is_rank_0():
            logger.info(
                'Manually rebuilding tokenizer with static system prompt')
        assert use_tokenizers is False
        if use_fast:
            def apply_chat_template(
                    self,
                    conversation,
                    tools=None,
                    documents=None,
                    chat_template=None,
                    add_generation_prompt=False,
                    continue_final_message=False,
                    tokenize=True, padding=False,
                    truncation=False,
                    max_length=None,
                    return_tensors=None,
                    return_dict=False,
                    return_assistant_tokens_mask=False,
                    tokenizer_kwargs=None,
                    **kwargs):
                if 'date_string' not in kwargs:
                    kwargs['date_string'] = '26 Jul 2024'
                return LlamaTokenizerFast.apply_chat_template(
                    self,
                    conversation, tools, documents, chat_template,
                    add_generation_prompt, continue_final_message, tokenize,
                    padding, truncation, max_length, return_tensors, return_dict,
                    return_assistant_tokens_mask, tokenizer_kwargs, **kwargs)
        else:
            def apply_chat_template(
                    self,
                    conversation,
                    tools=None,
                    documents=None,
                    chat_template=None,
                    add_generation_prompt=False,
                    continue_final_message=False,
                    tokenize=True, padding=False,
                    truncation=False,
                    max_length=None,
                    return_tensors=None,
                    return_dict=False,
                    return_assistant_tokens_mask=False,
                    tokenizer_kwargs=None,
                    **kwargs):
                if 'date_string' not in kwargs:
                    kwargs['date_string'] = '26 Jul 2024'
                return LlamaTokenizer.apply_chat_template(
                    self,
                    conversation, tools, documents, chat_template,
                    add_generation_prompt, continue_final_message, tokenize,
                    padding, truncation, max_length, return_tensors, return_dict,
                    return_assistant_tokens_mask, tokenizer_kwargs, **kwargs)

        tokenizer.apply_chat_template = types.MethodType(
            apply_chat_template, tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        if is_rank_0():
            logger.info(f"Set pad token to {tokenizer.pad_token} token.")

    return tokenizer


def build_model(
    config: Namespace,
    **kwargs
):
    assert config.pretrained_model_dir is not None, (
        "pretrained_model_dir is required.")

    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation=config.attn_implementation
    )

    return model


def build_peft_model(
    model,
    config: Namespace,
    **kwargs
):
    if config.peft_type == PeftType.LORA:

        if config.peft_init_lora_weights == "True":
            config.peft_init_lora_weights = True
        elif config.peft_init_lora_weights == "False":
            config.peft_init_lora_weights = False

        peft_config = LoraConfig(
            task_type=config.peft_task_type,
            inference_mode=config.peft_inference_mode,
            r=config.peft_lora_r,
            lora_alpha=config.peft_lora_alpha,
            lora_dropout=config.peft_lora_dropout,
            fan_in_fan_out=config.peft_lora_fan_in_fan_out,
            target_modules=config.peft_lora_target_modules,
            init_lora_weights=config.peft_init_lora_weights,
        )
    else:
        raise Exception(f"Unknown peft type: {config.peft_type}")

    model = get_peft_model(model, peft_config)
    return model


def build_optimizer(
    model,
    config: Namespace
):
    optimizer_type = config.optimizer_type
    peak_lr = config.peak_lr
    adam_beta1 = config.adam_beta1
    adam_beta2 = config.adam_beta2
    adam_eps = config.adam_eps
    weight_decay = config.weight_decay
    freeze_word_embeddings = config.freeze_word_embeddings

    def get_optimizer_grouped_parameters(module):
        param_optimizer = list(module.named_parameters())

        no_decay = ['bias', "ln", 'LayerNorm']
        frozen_parameters = set()
        if freeze_word_embeddings:

            frozen_parameters.add("transformer.wte.weight")
            frozen_parameters.add("lm_head.weight")
        optimizer_grouped_parameters = []
        parameters_with_decay = [
            p for n, p in param_optimizer
            if (
                not any(nd in n for nd in no_decay)
            ) and (
                n not in frozen_parameters
            ) and (
                p.requires_grad
            )
        ]
        parameters_without_decay = [
            p for n, p in param_optimizer
            if (
                any(nd in n for nd in no_decay)
            ) and (
                n not in frozen_parameters
            ) and (
                p.requires_grad
            )
        ]
        if len(parameters_with_decay) > 0:
            optimizer_grouped_parameters.append({
                "params": parameters_with_decay,
                "weight_decay": weight_decay
            })
        if len(parameters_without_decay) > 0:
            optimizer_grouped_parameters.append({
                "params": parameters_without_decay,
                "weight_decay": 0.0
            })
        return optimizer_grouped_parameters

    if optimizer_type == "adam":
        AdamOptimizer = DeepSpeedCPUAdam if config.offload_adam else FusedAdam
        optimizer = AdamOptimizer(
            get_optimizer_grouped_parameters(model),
            lr=peak_lr,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            weight_decay=weight_decay
        )
    else:
        raise Exception(f"Unknown optimizer type: {optimizer_type}")
    return optimizer


def build_lr_scheduler(
    optimizer,
    config: Namespace
):
    scheduler_type = config.scheduler_type
    n_warmup_steps = config.n_warmup_steps
    n_training_steps = config.n_training_steps
    peak_lr = config.peak_lr
    min_lr = config.min_lr

    if scheduler_type == "warmup_linear":
        def lr_lambda(current_step):
            if current_step < n_warmup_steps:
                return float(current_step) / float(max(1, n_warmup_steps))
            return max(0.0, float(n_training_steps - current_step) / float(max(1, n_training_steps - n_warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=-1
        )
    elif scheduler_type == "warmup_decay":
        scheduler = deepspeed.runtime.lr_schedules.WarmupDecayLR(
            optimizer,
            total_num_steps=n_training_steps,
            warmup_min_lr=min_lr,
            warmup_max_lr=peak_lr,
            warmup_num_steps=n_warmup_steps
        )
    elif scheduler_type == "warmup_cosine":
        warmup_min_ratio = min_lr / peak_lr
        scheduler = deepspeed.runtime.lr_schedules.WarmupCosineLR(
            optimizer,
            total_num_steps=n_training_steps,
            warmup_min_ratio=warmup_min_ratio,
            warmup_num_steps=n_warmup_steps,
            cos_min_ratio=config.cosine_min_ratio,
        )
    else:
        raise Exception(f"Unknown scheduler type: {scheduler_type}")
    return scheduler


def convert_to_numpy(elem):
    if isinstance(elem, np.ndarray):
        return elem
    elem = elem.detach().cpu()
    if elem.dtype == torch.bfloat16:
        elem = elem.float()
    return elem.numpy()


def convert_to_torch(elem, device="cuda"):
    if isinstance(elem, torch.Tensor):
        return elem.to(device=device)
    elif isinstance(elem, np.ndarray):
        return torch.from_numpy(elem).to(device=device)
    else:
        return torch.tensor(elem, device=device)


def get_vector_stats(
        array,
        prefix,
        stats_to_report=['mean', 'std', 'min', 'max']):
    if stats_to_report == 'all':
        stats_to_report = ['mean', 'std', 'min', 'max']
    res = {}
    if 'mean' in stats_to_report:
        res[prefix + '/mean'] = np.mean(array)
    if 'std' in stats_to_report:
        res[prefix + '/std'] = np.std(array)
    if 'min' in stats_to_report:
        res[prefix + '/min'] = np.amin(array)
    if 'max' in stats_to_report:
        res[prefix + '/max'] = np.amax(array)
    return res


class StatisticsTracker:
    def __init__(self):

        self.data = defaultdict(list)

        self.statistics = defaultdict(list)

        self.timed_statistics = defaultdict(lambda: defaultdict(list))

    def update_data(
        self,
        d: Dict,
        stats_for_tensors: str = 'all',
    ):

        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                self.data[k].extend(v)
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                stats = get_vector_stats(
                    array=convert_to_numpy(v),
                    prefix=k,
                    stats_to_report=stats_for_tensors,
                )
                self.update_data(d=stats)
            else:
                self.data[k].append(v)

    def update_stat(
        self,
        d: Dict,
        stats_for_tensors: str = 'all',
    ):

        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                self.statistics[k].extend(v)
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                stats = get_vector_stats(
                    array=convert_to_numpy(v),
                    prefix=k,
                    stats_to_report=stats_for_tensors,
                )
                self.update_stat(d=stats)
            else:
                self.statistics[k].append(v)

    def update_timed_stat(
        self,
        time2d: Dict[str, Dict],
    ):

        for timestep, kv in time2d.items():
            for k, v in kv.items():
                if isinstance(v, (list, tuple)):
                    self.timed_statistics[timestep][k].extend(v)
                else:
                    self.timed_statistics[timestep][k].append(v)

    def clear(self):

        self.data = defaultdict(list)
        self.statistics = defaultdict(list)

    def to_string(self):

        string_list = []
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            string_list.append("{}: {:.5g}".format(k, mean))
        return ", ".join(string_list)

    def to_dict(self):

        stat_dict = {}
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            stat_dict[k] = mean
        return stat_dict

    def get_value(
        self,
        key: str
    ):

        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None

    def items(self):

        for k, v in self.statistics.items():
            yield k, v

    def timed_items(self):

        for timestep in sorted(self.timed_statistics.keys()):
            kv = self.timed_statistics[timestep]
            yield timestep, kv


def convert_if_dictconfig(config):
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=False,
        )
    else:
        return config
