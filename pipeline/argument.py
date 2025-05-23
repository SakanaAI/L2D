

import configargparse
import argparse
from argparse import Namespace
import json
import logging
import os
from typing import Callable

import deepspeed
from peft import (
    PeftType,
    TaskType
)

from pipeline.util import is_rank_0


logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_or_float(v):

    try:
        return float(v)
    except ValueError:
        return v


def config_sanity_check(
    config: Namespace
):
    assert config.validate_interval != 0,        "validate_interval cannot be 0."
    if config.save_model_interval:
        assert config.save_model_interval != 0,            "save_model_interval cannot be 0."


def add_peft_arguments(parser):

    parser.add_argument("--pretrained_peft_model_dir", type=str)
    parser.add_argument(
        "--peft_type",
        type=PeftType,
        default=None,
        choices=["LORA"]
    )
    parser.add_argument(
        "--peft_task_type",
        type=TaskType,
        choices=["SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"]
    )
    parser.add_argument("--peft_inference_mode", action="store_true")

    parser.add_argument("--peft_lora_r", type=int, default=8)
    parser.add_argument("--peft_lora_alpha", type=float, default=32)
    parser.add_argument("--peft_lora_dropout", type=float, default=0.1)
    parser.add_argument("--peft_lora_fan_in_fan_out", action="store_true")
    parser.add_argument("--peft_lora_target_modules",
                        type=str, nargs="+", default=None)
    parser.add_argument(
        "--peft_init_lora_weights",
        type=str,
        default="True",
        help="See https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig"
             "`True` or `False` will be converted to a boolean"
    )


def add_generation_arguments(parser):
    parser.add_argument("--generation_min_len", type=int, default=1)
    parser.add_argument("--generation_max_len", type=int, default=128)
    parser.add_argument("--generation_do_sample", type=str2bool, default=True)
    parser.add_argument("--generation_top_p", type=float, default=1.0)
    parser.add_argument("--generation_top_k", type=int, default=0)
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_repetition_penalty",
                        type=float, default=1.0)
    parser.add_argument("--generation_num_return_sequences",
                        type=int, default=1)
    parser.add_argument("--use_legacy_past_key_values", action="store_true")


def get_arguments(
    custom_get_arguments_fn: Callable = None,
    custom_sanity_check_fn: Callable = None,
    default_config_files: list = []
):
    parser = configargparse.ArgumentParser(
        default_config_files=default_config_files
    )

    parser.add('-c', '--config_path', required=False, is_config_file=True,
               help='config file path')
    parser.add('--extra_config1', required=False, is_config_file=True,
               help='extra config file path')
    parser.add('--extra_config2', required=False, is_config_file=True,
               help='extra config file path')
    parser.add('--extra_config3', required=False, is_config_file=True,
               help='extra config file path')

    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--tokenizer_dir", type=str)
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--use_tokenizers", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "flash_attention_2", "sdpa"]
    )
    parser.add_argument("--specified_template_type", type=str)

    parser.add_argument("--train_filepaths", type=str,
                        nargs="+", required=True)
    parser.add_argument("--valid_filepaths", type=str, nargs="+")
    parser.add_argument("--data_processor_types", type=str, nargs="+")
    parser.add_argument("--valid_data_processor_types", type=str, nargs="+")
    parser.add_argument("--data_reformatter_type", type=str)
    parser.add_argument("--held_out_valid_portion", type=float)
    parser.add_argument("--held_out_valid_number", type=int)
    parser.add_argument("--min_seq_len", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1e9)
    parser.add_argument("--min_src_seq_len", type=int, default=1)
    parser.add_argument("--max_src_seq_len", type=int, default=1e9)
    parser.add_argument("--min_tgt_seq_len", type=int, default=1)
    parser.add_argument("--max_tgt_seq_len", type=int, default=1e9)
    parser.add_argument("--multiple_samples_per_jsonl_line",
                        action="store_true")

    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--micro_valid_batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--n_training_steps", type=int, default=1e8)
    parser.add_argument("--freeze_word_embeddings", action="store_true")
    parser.add_argument("--n_samples_before_backprop", type=int, default=None)
    parser.add_argument("--n_accum_steps", type=int, default=None)

    parser.add_argument("--optimizer_type", type=str,
                        default="adam", choices=["adam"])
    parser.add_argument("--peak_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--cosine_min_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--scheduler_type", type=str, default="warmup_decay",
        choices=["warmup_decay", "warmup_linear", "warmup_cosine"])
    parser.add_argument("--n_warmup_steps", type=int, default=0)

    parser.add_argument("--from_checkpoint_path", type=str)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_final", action="store_true")
    parser.add_argument("--save_all_models", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_filename", type=str)
    parser.add_argument("--add_time_to_filename", type=str2bool, default=True)
    parser.add_argument("--log_interval", type=float, default=100)
    parser.add_argument("--validate_interval", type=float, default=1000)
    parser.add_argument("--save_model_interval", type=float)
    parser.add_argument("--monitor_metric", type=str, default="loss")
    parser.add_argument("--monitor_metric_should_ascend",
                        action="store_true", default=False)

    parser.add_argument("--save_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lad")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_mode_data_size", type=int, default=1000)

    parser.add_argument("--local_rank", type=int, required=True)
    parser.add_argument("--zero_stage", type=int,
                        default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--activation_checkpointing_layers", type=int)
    parser.add_argument("--offload_adam", action="store_true")
    parser.add_argument("--offload_param", action="store_true")
    parser = deepspeed.add_config_arguments(parser)

    add_peft_arguments(parser)

    add_generation_arguments(parser)

    if custom_get_arguments_fn is not None:
        parser = custom_get_arguments_fn(parser)

    config = parser.parse_args()

    config_sanity_check(config)
    if custom_sanity_check_fn is not None:
        custom_sanity_check_fn(config)

    config.global_rank = int(os.environ['RANK'])
    config.world_size = int(os.environ['WORLD_SIZE'])

    batch_size = config.world_size*config.micro_batch_size
    if config.n_accum_steps is not None:
        assert config.n_samples_before_backprop is None
        config.n_samples_before_backprop = config.n_accum_steps*batch_size
    elif config.n_samples_before_backprop is not None:
        assert config.n_samples_before_backprop % batch_size == 0, (
            'the specified total num of samples before a gradient step is not '
            'divisible by the total batch size across devices')
        config.n_accum_steps = config.n_samples_before_backprop//batch_size
    else:
        config.n_accum_steps = 1
        config.n_samples_before_backprop = batch_size

    logger.info(f'Batch size across devices: {batch_size}')
    logger.info(f'Number of accumulation steps: {config.n_accum_steps}')
    logger.info('Number of samples before gradient step: '
                f'{config.n_samples_before_backprop}')

    config.deepspeed_config = {
        "train_micro_batch_size_per_gpu": config.micro_batch_size,
        "gradient_accumulation_steps": config.n_accum_steps,
        "fp16": {
            "enabled": config.fp16
        },
        "bf16": {
            "enabled": config.bf16
        },
        "gradient_clipping": config.grad_clip,
        "zero_optimization": {
            "stage": config.zero_stage,
            "offload_param": {
                "device": "cpu" if config.offload_param else "none"
            },
            "offload_optimizer": {
                "device": "cpu" if config.offload_adam else "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "sub_group_size": "auto",
            "stage3_gather_16bit_weights_on_model_save": True,
            "stage3_max_live_parameters": "auto",
            "stage3_max_reuse_distance": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "reduce_bucket_size": "auto",
        },
        "activation_checkpointing": {
            "partition_activations": True if config.activation_checkpointing_layers else False,
            "number_checkpoints": config.activation_checkpointing_layers,
            "synchronize_checkpoint_boundary": True if config.activation_checkpointing_layers else False,
        },
        "steps_per_print": 1e12
    }

    return config


def get_inference_arguments(
    custom_get_arguments_fn: Callable = None,
    custom_sanity_check_fn: Callable = None,
    default_config_files: list = []
):
    parser = configargparse.ArgumentParser(
        default_config_files=default_config_files
    )

    parser.add('-c', '--config_path', required=False, is_config_file=True,
               help='config file path')
    parser.add('--extra_config1', required=False, is_config_file=True,
               help='extra config file path')
    parser.add('--extra_config2', required=False, is_config_file=True,
               help='extra config file path')
    parser.add('--extra_config3', required=False, is_config_file=True,
               help='extra config file path')

    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--tokenizer_dir", type=str)
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--use_tokenizers", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "flash_attention_2", "sdpa"]
    )
    parser.add_argument("--specified_template_type", type=str)

    parser.add_argument("--filepaths", type=str, nargs="+")
    parser.add_argument("--output_filepaths", type=str, nargs="+")
    parser.add_argument("--data_processor_types", type=str, nargs="+")
    parser.add_argument("--data_reformatter_type", type=str)
    parser.add_argument("--min_seq_len", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1e9)
    parser.add_argument("--min_src_seq_len", type=int, default=1)
    parser.add_argument("--max_src_seq_len", type=int, default=1e9)
    parser.add_argument("--min_tgt_seq_len", type=int, default=1)
    parser.add_argument("--max_tgt_seq_len", type=int, default=1e9)
    parser.add_argument("--multiple_samples_per_jsonl_line",
                        action="store_true")

    parser.add_argument("--micro_batch_size", type=int, default=32)

    parser.add_argument("--from_checkpoint_path", type=str)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_filename", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_mode_data_size", type=int, default=1000)

    parser.add_argument("--local_rank", type=int, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--zero_stage", type=int,
                        default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--offload_param", action="store_true")
    parser = deepspeed.add_config_arguments(parser)

    parser.add_argument("--save_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lad")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group_name", type=str, default=None)

    add_peft_arguments(parser)

    add_generation_arguments(parser)

    if custom_get_arguments_fn is not None:
        parser = custom_get_arguments_fn(parser)

    config, uknown_args = parser.parse_known_args()

    if len(uknown_args) > 0:
        for arg in uknown_args:
            logger.warning(f"WARRNING: unknown argument passed: {arg}")

    if custom_sanity_check_fn is not None:
        custom_sanity_check_fn(config)

    config.global_rank = int(os.environ['RANK'])
    config.world_size = int(os.environ['WORLD_SIZE'])

    if config.zero_stage in [1, 2]:
        config.zero_stage = 0

    config.deepspeed_config = {
        "train_micro_batch_size_per_gpu": config.micro_batch_size,
        "fp16": {
            "enabled": config.fp16
        },
        "bf16": {
            "enabled": config.bf16
        },
        "zero_optimization": {
            "stage": config.zero_stage,
            "offload_param": {
                "device": "cpu" if config.offload_param else "none",
                "pin_memory": True
            },
        },
        "steps_per_print": 1e12
    }

    return config
