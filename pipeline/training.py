

import argparse
from collections import defaultdict
import logging
import math
import time
from typing import (
    Callable,
    List,
    Type
)
import os

import deepspeed
import numpy as np
from peft import PeftModel
import torch
import torch.utils.tensorboard
import wandb
from tqdm import tqdm
from transformers.integrations import HfDeepSpeedConfig

from pipeline.data import (
    BasicDataSource,
    DataProcessor
)
from pipeline.util import (
    StatisticsTracker,
    build_lr_scheduler,
    build_model,
    build_optimizer,
    build_peft_model,
    build_tokenizer,
    get_number_of_parameters,
    is_rank_0,
    set_logging_config,
    set_random_seed,
)


class TrainingPipeline(object):
    def __init__(
        self,
        config: argparse.Namespace,
        world_size: int,
        local_rank: int,
        global_rank: int,
    ):

        self.config = config

        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = global_rank

    def create_fileid(self):
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if self.config.save_filename is None:
            if self.config.wandb_run_name is not None:
                save_filename = self.config.wandb_run_name
            else:
                save_filename = 'model'
            if self.config.add_time_to_filename:
                self.save_fileid = "{}_{}".format(save_filename, time_str)
            else:
                self.save_fileid = "{}".format(save_filename)
        else:
            if self.config.add_time_to_filename:
                self.save_fileid = "{}_{}".format(
                    self.config.save_filename, time_str)
            else:
                self.save_fileid = "{}".format(self.config.save_filename)

        self.config.save_filename = self.save_fileid

        save_fileid_obj = [self.save_fileid]
        torch.distributed.broadcast_object_list(save_fileid_obj, src=0)
        self.save_fileid = save_fileid_obj[0]

        if self.config.save_dir:
            self.save_dir = os.path.join(
                self.config.save_dir,
                self.save_fileid
            )
        else:
            self.save_dir = os.path.join(
                "outputs",
                self.save_fileid
            )
        self.save_dir = os.path.abspath(self.save_dir)

        if self.config.save_log:
            self.save_log_filepath = os.path.join(
                self.save_dir,
                "log.txt"
            )
            self.save_log_filepath = os.path.abspath(self.save_log_filepath)
        else:
            self.save_log_filepath = None

        if self.config.save_model:
            self.save_model_dir = os.path.join(
                self.save_dir,
                "checkpoint",
            )
            self.save_model_dir = os.path.abspath(self.save_model_dir)

        if is_rank_0():
            if self.config.save_log:
                os.makedirs(self.save_dir, exist_ok=True)
            if self.config.save_model:
                os.makedirs(self.save_model_dir, exist_ok=True)
        torch.distributed.barrier()

    def create_logger(self):

        set_logging_config(log_filepath=self.save_log_filepath)
        self.logger = logging.getLogger(__name__)

    def save_model(self, save_model_dir):

        base_model = self.model.module

        os.makedirs(save_model_dir, exist_ok=True)

        if self.config.zero_stage != 3:

            if hasattr(base_model, "save_pretrained"):
                if is_rank_0():
                    base_model.save_pretrained(
                        save_model_dir,
                        safe_serialization=False,
                        max_shard_size='50GB',
                    )

            else:
                if is_rank_0():
                    save_filepath = os.path.join(
                        save_model_dir, "pytorch_model.bin")
                    checkpoint = {
                        "model_state_dict": base_model.state_dict(),
                    }
                    torch.save(checkpoint, save_filepath)

        else:

            if is_rank_0():
                base_model.save_pretrained(
                    save_model_dir,
                    safe_serialization=False,
                    max_shard_size='50GB',
                )
            torch.distributed.barrier()

            if self.config.peft_type is None:
                success = self.model.save_16bit_model(
                    save_dir=save_model_dir,
                    save_filename="pytorch_model.bin",
                    exclude_frozen_parameters=False
                )
            else:
                success = self.model.save_16bit_model(
                    save_dir=save_model_dir,
                    save_filename="adapter_model.bin",
                    exclude_frozen_parameters=True
                )
            assert success, "Failed to save ZeRO-3 model."

    def run(
        self,
        load_data_from_filepath_fn: Callable,
        data_processor_classes: List[Type[DataProcessor]],
        train_forward_step_fn: Callable,
        valid_forward_step_fn: Callable,
        build_tokenizer_fn: Callable = build_tokenizer,
        build_model_fn: Callable = build_model,
        build_peft_model_fn: Callable = build_peft_model,
        build_optimizer_fn: Callable = build_optimizer,
        build_lr_scheduler_fn: Callable = build_lr_scheduler,
        valid_data_processor_classes: List[Type[DataProcessor]] = None,
        data_reformatter_class: Type[DataProcessor] = None,
        data_source_class: Type[torch.utils.data.Dataset] = BasicDataSource,
        compute_metrics_fn: Callable = None,
        collate_fn: Callable = None
    ):

        assert load_data_from_filepath_fn is not None
        assert data_processor_classes is not None
        assert train_forward_step_fn is not None
        assert valid_forward_step_fn is not None

        deepspeed.init_distributed()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        set_random_seed(self.config.seed)

        self.create_fileid()

        if is_rank_0():
            self.create_logger()
            if self.config.save_model:
                self.logger.info(
                    f"INFO: Saving model to: {self.save_model_dir}")
            self.logger.info("Training pipeline initialized.")

        if is_rank_0():
            self.logger.info("Start building tokenizer.")
        self.tokenizer = build_tokenizer_fn(
            tokenizer_dir=self.config.tokenizer_dir,
            use_fast=self.config.use_fast_tokenizer,
            use_tokenizers=self.config.use_tokenizers
        )

        if is_rank_0():
            self.logger.info("Start loading training data.")
        train_filepath2data = load_data_from_filepath_fn(
            tokenizer=self.tokenizer,
            stage="train",
            filepaths=self.config.train_filepaths,
            data_processor_classes=data_processor_classes,
            data_reformatter_class=data_reformatter_class,
            config=self.config
        )
        train_data = train_filepath2data["all"]
        if self.config.valid_filepaths is None:
            if is_rank_0():
                self.logger.info("Valid data filepaths are not provided. Hold "
                                 "out training data for validation.")
            loaded_train_data = train_data

            assert self.config.held_out_valid_portion is not None or self.config.held_out_valid_number is not None,                "{held_out_valid_portion} and {held_out_valid_number} cannot be both None."
            assert self.config.held_out_valid_portion is None or self.config.held_out_valid_number is None,                "{held_out_valid_portion} and {held_out_valid_number} cannot be both provided."
            if self.config.held_out_valid_portion is not None:
                train_data_number = int(
                    len(train_data)*(1-self.config.held_out_valid_portion))
            else:
                train_data_number = len(train_data) - \
                    self.config.held_out_valid_number
            train_data = train_data[:train_data_number]
        train_data_source = data_source_class(
            stage="train",
            data=train_data,
            tokenizer=self.tokenizer,
            config=self.config
        )
        train_data_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=train_data_source,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
            seed=0,
            drop_last=False
        )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_data_source,
            batch_size=self.config.micro_batch_size,
            sampler=train_data_sampler,
            num_workers=0,
            collate_fn=train_data_source.collate_fn if collate_fn is None else collate_fn,
            pin_memory=True,
        )
        if is_rank_0():
            self.logger.info("Finished loading training data.")
            self.logger.info(f"Training data stat: {train_data_source.stat}")

        if is_rank_0():
            self.logger.info("Start loading valid data.")
        if self.config.valid_filepaths is None:
            if is_rank_0():
                self.logger.info("Valid data filepaths are not provided. Use "
                                 "held-out training data instead.")
            valid_data = loaded_train_data[train_data_number:]
            valid_filepath2data = {
                "all": valid_data
            }
        else:
            if valid_data_processor_classes is None:
                valid_data_processor_classes = data_processor_classes
            valid_filepath2data = load_data_from_filepath_fn(
                tokenizer=self.tokenizer,
                stage="valid",
                filepaths=self.config.valid_filepaths,
                data_processor_classes=valid_data_processor_classes,
                data_reformatter_class=data_reformatter_class,
                config=self.config,
                one_dataset_per_input=True
            )
        valid_filepath2data_loader = {}
        for valid_filepath, valid_data in valid_filepath2data.items():
            if len(valid_data) == 0:
                if is_rank_0():
                    self.logger.info(
                        f"Valid data {valid_filepath} is empty. Skip.")
                continue
            valid_data_source = data_source_class(
                stage="valid",
                data=valid_data,
                tokenizer=self.tokenizer,
                config=self.config
            )
            valid_data_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=valid_data_source,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,
                seed=0,
                drop_last=False
            )
            valid_data_loader = torch.utils.data.DataLoader(
                dataset=valid_data_source,
                batch_size=self.config.micro_valid_batch_size,
                sampler=valid_data_sampler,
                num_workers=0,
                collate_fn=valid_data_source.collate_fn if collate_fn is None else collate_fn,
                pin_memory=True,
            )
            valid_filepath2data_loader[valid_filepath] = valid_data_loader
            if is_rank_0():
                self.logger.info(f"Valid data stat: {valid_data_source.stat}")

        if is_rank_0():
            self.logger.info("Start building statistics tracker.")

        train_stat_tracker = StatisticsTracker()
        if is_rank_0():

            valid_filepath2stat_tracker = defaultdict(StatisticsTracker)

        training_steps_per_epoch = math.ceil(
            len(train_data_loader)/self.config.n_accum_steps)
        max_training_steps = min(
            self.config.n_training_steps, training_steps_per_epoch * self.config.n_epochs)
        self.config.n_training_steps = max_training_steps

        if self.config.log_interval < 0:
            self.config.log_interval = math.ceil(
                (-1) * self.config.log_interval * training_steps_per_epoch
            )
            if is_rank_0():
                self.logger.info(
                    "Setting logging interval proportional to training steps per epoch.")
                self.logger.info(
                    f"\tLogging interval: {self.config.log_interval}")
        else:
            self.config.log_interval = int(self.config.log_interval)
        if self.config.validate_interval < 0:
            self.config.validate_interval = math.ceil(
                (-1) * self.config.validate_interval * training_steps_per_epoch
            )
            if is_rank_0():
                self.logger.info(
                    "Setting validation interval proportional to training steps per epoch.")
                self.logger.info(
                    f"\tValidation interval: {self.config.validate_interval}")
        else:
            self.config.validate_interval = int(self.config.validate_interval)
        if self.config.save_model_interval is None:
            self.config.save_model_interval = self.config.validate_interval
        else:
            if self.config.save_model_interval < 0:
                self.config.save_model_interval = math.ceil(
                    (-1) * self.config.save_model_interval *
                    training_steps_per_epoch
                )
                if is_rank_0():
                    self.logger.info(
                        "Setting model saving interval proportional to training steps per epoch.")
                    self.logger.info(
                        f"\tModel saving interval: {self.config.save_model_interval}")
            else:
                self.config.save_model_interval = int(
                    self.config.save_model_interval)
        assert self.config.save_model_interval % self.config.validate_interval == 0,            f"save_model_interval must be a multiple of validate_interval, but got \
                save_model_interval = {self.config.save_model_interval} \
                and validate_interval = {self.config.validate_interval}."

        if is_rank_0():
            self.logger.info("Start building model.")

        if self.config.zero_stage == 3:
            dschf = HfDeepSpeedConfig(self.config.deepspeed_config)

        self.model = build_model_fn(
            config=self.config,
            tokenizer=self.tokenizer
        )

        if self.config.pretrained_peft_model_dir is not None:
            if is_rank_0():
                self.logger.info("Start loading pretrained PEFT model.")
            self.model = PeftModel.from_pretrained(
                model=self.model,
                model_id=self.config.pretrained_peft_model_dir,
                is_trainable=True
            )
        if self.config.peft_type is not None:
            if is_rank_0():
                self.logger.info("Start building PEFT model.")
            self.model = build_peft_model_fn(
                model=self.model,
                config=self.config
            )

        if is_rank_0():
            param_stat = get_number_of_parameters(self.model)
            param_stat_text = " | ".join(
                [f"{k}: {v}" for k, v in param_stat.items()])
            self.logger.info(
                f"Number of trainable parameters: {param_stat_text}")

        if is_rank_0():
            self.logger.info(f"Move model to device.")
        self.model = self.model.to(self.device)

        if is_rank_0():
            self.logger.info(
                f"Load checkpoint from {self.config.from_checkpoint_path}.")
        if self.config.from_checkpoint_path is not None:
            checkpoint = torch.load(
                self.config.from_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(
                checkpoint["model_state_dict"], strict=False)

        if is_rank_0():
            self.logger.info("Start building optimizer.")
        self.optimizer = build_optimizer_fn(
            model=self.model,
            config=self.config
        )

        if is_rank_0():
            self.logger.info("Start building scheduler.")
        self.lr_scheduler = build_lr_scheduler_fn(
            optimizer=self.optimizer,
            config=self.config
        )

        if is_rank_0():
            self.logger.info("Start initializing deepspeed.")
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config.deepspeed_config,
            lr_scheduler=self.lr_scheduler,
            dist_init_required=False
        )
        if is_rank_0():
            self.logger.info("Finished initializing deepspeed.")

        step_idx = 0
        start_epoch_idx = 0
        best_valid_monitor = - \
            float("inf") if self.config.monitor_metric_should_ascend else float("inf")

        if self.config.save_log and is_rank_0():
            tensorboard_writer = torch.utils.tensorboard.SummaryWriter(
                log_dir=self.save_dir,
                max_queue=5
            )

            if self.config.save_wandb:

                if self.config.wandb_group_name is None:
                    group_name = self.save_dir.removesuffix(self.save_fileid)
                else:
                    group_name = self.config.wandb_group_name
                group_name = group_name[:127]
                if self.config.wandb_run_name is None:
                    run_name = self.save_fileid
                else:
                    run_name = self.config.wandb_run_name
                run_name = run_name[:127]

                self.config.save_dir = self.save_dir
                wandb.init(
                    project=self.config.wandb_project,
                    group=group_name,
                    name=run_name,
                    config=self.config,
                )

        if is_rank_0():
            self.logger.info(f"Hyper-parameters:")
            for k, v in sorted(dict(self.config.__dict__).items()):
                self.logger.info(f"{k}: {v}")

        for epoch_idx in range(start_epoch_idx, self.config.n_epochs):

            train_data_loader.sampler.set_epoch(self.config.seed + epoch_idx)
            train_data_iterator = iter(train_data_loader)

            while step_idx < self.config.n_training_steps:
                step_idx += 1

                for accum_idx in range(self.config.n_accum_steps):

                    try:
                        batch_data = next(train_data_iterator)
                    except StopIteration:
                        batch_data = None
                        break

                    self.model.train()
                    loss, ret_data, ret_stat = train_forward_step_fn(
                        step=step_idx,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        batch_data=batch_data,
                        config=self.config
                    )

                    train_stat_tracker.update_data(ret_data)
                    train_stat_tracker.update_stat(ret_stat)

                    loss = loss / self.config.n_accum_steps
                    self.model.backward(loss)
                    self.model.step()
                    del loss

                if is_rank_0() and step_idx % self.config.log_interval == 0:

                    if self.config.save_log:
                        dict_to_log = {}
                        for k, v in train_stat_tracker.items():
                            key_to_log = f"{k}/train"
                            dict_to_log[key_to_log] = np.mean(v)
                            tensorboard_writer.add_scalar(
                                key_to_log, dict_to_log[key_to_log], step_idx)

                    lr = list(self.lr_scheduler.optimizer.param_groups)[
                        0]["lr"]
                    self.logger.info(
                        f"[Epoch {epoch_idx+1}/{self.config.n_epochs}], "
                        f"[Step {step_idx}/{self.config.n_training_steps}], "
                        f"[LR {lr:.5g}]\n\t{train_stat_tracker.to_string()}"
                    )

                    if self.config.save_log and self.config.save_wandb:
                        dict_to_log['epoch'] = epoch_idx
                        dict_to_log['learning_rate'] = lr
                        wandb.log(data=dict_to_log, step=step_idx)

                    train_stat_tracker.clear()

                if (
                    step_idx % self.config.validate_interval == 0
                    or step_idx == self.config.n_training_steps
                ):
                    if is_rank_0():
                        self.logger.info(" Validation ".center(50, "*"))
                        self.logger.info(
                            f"[Epoch {epoch_idx+1}/{self.config.n_epochs}], "
                            f"[Step {step_idx}/{self.config.n_training_steps}]"
                        )
                    for valid_filepath, valid_data_loader in valid_filepath2data_loader.items():
                        with torch.no_grad():
                            self.model.eval()
                            rank_ret_data, rank_ret_stat = [], []
                            for valid_batch_data in tqdm(
                                valid_data_loader,
                                desc="Validating",
                                disable=not is_rank_0()
                            ):
                                ret_data, ret_stat = valid_forward_step_fn(
                                    step=step_idx,
                                    model=self.model,
                                    tokenizer=self.tokenizer,
                                    batch_data=valid_batch_data,
                                    config=self.config
                                )
                                rank_ret_data.append(ret_data)
                                rank_ret_stat.append(ret_stat)

                        gathered_ret_data = [None for _ in range(
                            torch.distributed.get_world_size())]
                        gathered_ret_stat = [None for _ in range(
                            torch.distributed.get_world_size())]
                        torch.distributed.all_gather_object(
                            gathered_ret_data,
                            rank_ret_data,
                        )
                        torch.distributed.all_gather_object(
                            gathered_ret_stat,
                            rank_ret_stat,
                        )

                        if is_rank_0():

                            gathered_ret_data = [
                                item for sublist in gathered_ret_data
                                for item in sublist]
                            gathered_ret_stat = [
                                item for sublist in gathered_ret_stat
                                for item in sublist]

                            for ret_data in gathered_ret_data:

                                valid_filepath2stat_tracker[
                                    "all"].update_data(ret_data)
                            for ret_stat in gathered_ret_stat:

                                valid_filepath2stat_tracker[
                                    "all"].update_stat(ret_stat)

                    if is_rank_0():

                        if compute_metrics_fn is not None:
                            metrics_data, metrics_stat = compute_metrics_fn(
                                valid_filepath2stat_tracker["all"].data,
                                self.config
                            )
                            valid_filepath2stat_tracker["all"].update_data(
                                metrics_data)
                            valid_filepath2stat_tracker["all"].update_stat(
                                metrics_stat)

                        if self.config.save_log:
                            dict_to_log = {}
                            for k, v in valid_filepath2stat_tracker[
                                    "all"].items():
                                key_to_log = f"{k}/valid/all"
                                dict_to_log[key_to_log] = np.mean(v)
                                tensorboard_writer.add_scalar(
                                    key_to_log, dict_to_log[key_to_log],
                                    step_idx)
                            if self.config.save_wandb:
                                wandb.log(data=dict_to_log, step=step_idx)

                        self.logger.info(
                            f"all:\n\t{valid_filepath2stat_tracker['all'].to_string()}")

                if is_rank_0() and (
                    step_idx % self.config.save_model_interval == 0
                    or step_idx == self.config.n_training_steps
                ):

                    cur_valid_monitor = valid_filepath2stat_tracker["all"].get_value(
                        self.config.monitor_metric)
                    if (
                        (cur_valid_monitor <
                         best_valid_monitor and not self.config.monitor_metric_should_ascend)
                        or (cur_valid_monitor > best_valid_monitor and self.config.monitor_metric_should_ascend)
                    ):
                        best_valid_monitor = cur_valid_monitor
                        is_best_model = True
                    else:
                        is_best_model = False
                else:
                    is_best_model = False

                if (
                    step_idx % self.config.save_model_interval == 0
                    or step_idx == self.config.n_training_steps
                ) and self.config.save_model:

                    is_best_model = torch.tensor(
                        is_best_model, dtype=torch.bool, device=self.device)
                    torch.distributed.broadcast(is_best_model, src=0)

                    if self.config.save_all_models:
                        save_model_dir = os.path.join(
                            self.save_model_dir,
                            f"epoch{epoch_idx+1:02}.step{step_idx}"
                        )
                        if is_rank_0():
                            self.logger.info(
                                f"Saving model to {save_model_dir}")
                        self.save_model(save_model_dir)

                    if is_best_model:
                        save_model_dir = os.path.join(
                            self.save_model_dir,
                            "best"
                        )
                        if is_rank_0():
                            self.logger.info(
                                f"Saving best model to {save_model_dir}")
                        self.save_model(save_model_dir)
                    if self.config.save_final and (
                            step_idx == self.config.n_training_steps):
                        save_model_dir = os.path.join(
                            self.save_model_dir,
                            "final"
                        )
                        if is_rank_0():
                            self.logger.info(
                                f"Saving final model to {save_model_dir}")
                        self.save_model(save_model_dir)
                torch.distributed.barrier()

                if is_rank_0() and (
                    step_idx % self.config.validate_interval == 0
                    or step_idx == self.config.n_training_steps
                ):

                    for valid_stat_tracker in valid_filepath2stat_tracker.values():
                        valid_stat_tracker.clear()

                    self.logger.info("*".center(50, "*"))

                    torch.cuda.empty_cache()

                if batch_data is None:
                    break

        if is_rank_0():
            self.logger.info("Training pipeline finished.")
            self.logger.info(
                f"Best validation {self.config.monitor_metric}: {best_valid_monitor}")
            if self.config.save_model:
                save_model_dir = os.path.join(
                    self.save_model_dir,
                    "best"
                )
                self.logger.info(f"Best model saved to {save_model_dir}")
