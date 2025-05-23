

import argparse
import logging
import os
import time
from typing import (
    Callable,
    List,
    Type
)
from datetime import timedelta

import deepspeed
import numpy as np
from peft import PeftModel
import torch
from tqdm import tqdm
from transformers.integrations import HfDeepSpeedConfig
import wandb

from pipeline.data import (
    BasicDataSource,
    DataProcessor
)
from pipeline.util import (
    StatisticsTracker,
    build_model,
    build_tokenizer,
    is_rank_0,
    set_logging_config,
    set_random_seed,
)


class InferencePipeline(object):
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
            self.save_fileid = "{}_{}".format(save_filename, time_str)
        else:
            self.save_fileid = "{}_{}".format(
                self.config.save_filename, time_str)

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
                "log_inference.txt"
            )
            self.save_log_filepath = os.path.abspath(self.save_log_filepath)
        else:
            self.save_log_filepath = None

        if is_rank_0():
            if self.config.save_log:
                os.makedirs(self.save_dir, exist_ok=True)
        torch.distributed.barrier()

    def create_logger(self):

        set_logging_config()
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        load_data_from_filepath_fn: Callable,
        data_processor_classes: List[Type[DataProcessor]],
        forward_step_fn: Callable,
        build_tokenizer_fn: Callable = build_tokenizer,
        build_model_fn: Callable = build_model,
        data_reformatter_class: Type[DataProcessor] = None,
        data_source_class: Type[torch.utils.data.Dataset] = BasicDataSource,
        compute_metrics_fn: Callable = None,
        collate_fn: Callable = None
    ):

        assert load_data_from_filepath_fn is not None
        assert data_processor_classes is not None
        assert forward_step_fn is not None

        timeout = timedelta(hours=12)
        deepspeed.init_distributed(timeout=timeout)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        set_random_seed(self.config.seed)

        self.create_fileid()

        if is_rank_0():
            self.create_logger()
            self.logger.info("Inference pipeline initialized.")

        if is_rank_0():
            self.logger.info("Start building tokenizer.")
        self.tokenizer = build_tokenizer_fn(
            tokenizer_dir=self.config.tokenizer_dir,
            use_fast=self.config.use_fast_tokenizer,
            use_tokenizers=self.config.use_tokenizers
        )

        if is_rank_0():
            self.logger.info("Start building statistics tracker.")
            self.inference_filepath2stat_tracker = {
                inference_filepath: StatisticsTracker()
                for inference_filepath in self.config.filepaths
            }
            self.stat_tracker = StatisticsTracker()

        if is_rank_0():
            self.logger.info("Start loading data.")
        inference_filepath2data = load_data_from_filepath_fn(
            tokenizer=self.tokenizer,
            stage="inference",
            filepaths=self.config.filepaths,
            data_processor_classes=data_processor_classes,
            data_reformatter_class=data_reformatter_class,
            config=self.config,
            one_dataset_per_input=True
        )
        inference_filepath2data_loader = {}
        for inference_filepath, inference_data in inference_filepath2data.items():
            if len(inference_data) == 0:
                if is_rank_0():
                    self.logger.info(
                        f"Inference data {inference_filepath} is empty. Skip.")
                continue
            inference_data_source = data_source_class(
                stage="inference",
                data=inference_data,
                tokenizer=self.tokenizer,
                config=self.config
            )
            inference_data_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=inference_data_source,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,
                seed=0,
                drop_last=False
            )
            inference_data_loader = torch.utils.data.DataLoader(
                dataset=inference_data_source,
                batch_size=self.config.micro_batch_size,
                sampler=inference_data_sampler,
                num_workers=0,
                collate_fn=inference_data_source.collate_fn if collate_fn is None else collate_fn,
                pin_memory=True,
            )
            inference_filepath2data_loader[inference_filepath] = inference_data_loader
            if is_rank_0():
                self.logger.info(
                    f"Inference data stat: {inference_data_source.stat}")

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
                is_trainable=False
            )

        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        if is_rank_0():
            self.logger.info(f"Move model to device.")
        if self.model is not None:
            self.model = self.model.to(self.device)

        if is_rank_0():
            self.logger.info(
                f"Load checkpoint from {self.config.from_checkpoint_path}.")
        if self.config.from_checkpoint_path is not None:
            checkpoint = torch.load(
                self.config.from_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=True)

        if is_rank_0():
            self.logger.info("Start initializing deepspeed.")
        if self.model is not None:
            self.model, _, _, _ = deepspeed.initialize(
                model=self.model,
                config=self.config.deepspeed_config,
                dist_init_required=False
            )
        if is_rank_0():
            self.logger.info("Finished initializing deepspeed.")

        if self.config.save_log and is_rank_0():
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
            self.logger.info(" Inference ".center(50, "*"))

        for inference_filepath, inference_data_loader in inference_filepath2data_loader.items():
            with torch.no_grad():
                if self.model is not None:
                    self.model.eval()
                rank_ret_data, rank_ret_stat, rank_ret_timed_stat = [], [], []
                for eval_batch_data in tqdm(
                    inference_data_loader,
                    desc="Inference",
                    disable=not is_rank_0()
                ):
                    ret_data, ret_stat, ret_timed_stat = forward_step_fn(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        batch_data=eval_batch_data,
                        config=self.config
                    )
                    rank_ret_data.append(ret_data)
                    rank_ret_stat.append(ret_stat)
                    rank_ret_timed_stat.append(ret_timed_stat)

            torch.cuda.empty_cache()

            distributed_inference = self.world_size >= 1
            if distributed_inference:
                gathered_ret_data = [
                    None for _ in range(torch.distributed.get_world_size())]
                gathered_ret_stat = [
                    None for _ in range(torch.distributed.get_world_size())]
                gathered_ret_timed_stat = [
                    None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.gather_object(
                    rank_ret_data,
                    gathered_ret_data if is_rank_0() else None
                )
                torch.distributed.gather_object(
                    rank_ret_stat,
                    gathered_ret_stat if is_rank_0() else None
                )
                torch.distributed.gather_object(
                    rank_ret_timed_stat,
                    gathered_ret_timed_stat if is_rank_0() else None
                )
            else:
                gathered_ret_data = rank_ret_data
                gathered_ret_stat = rank_ret_stat
                gathered_ret_timed_stat = rank_ret_timed_stat

            if is_rank_0():

                file_stat_tracker = self.inference_filepath2stat_tracker[inference_filepath]

                if distributed_inference:
                    gathered_ret_data = [
                        item for sublist in gathered_ret_data for item in sublist]
                    gathered_ret_stat = [
                        item for sublist in gathered_ret_stat for item in sublist]
                    gathered_ret_timed_stat = [
                        item for sublist in gathered_ret_timed_stat for item in sublist]

                for ret_data in gathered_ret_data:
                    file_stat_tracker.update_data(ret_data)
                    self.stat_tracker.update_data(ret_data)
                for ret_stat in gathered_ret_stat:
                    file_stat_tracker.update_stat(ret_stat)
                    self.stat_tracker.update_stat(ret_stat)
                for ret_timed_stat in gathered_ret_timed_stat:
                    file_stat_tracker.update_timed_stat(ret_timed_stat)
                    self.stat_tracker.update_timed_stat(ret_timed_stat)

                if compute_metrics_fn is not None:
                    metrics_data, metrics_stat = compute_metrics_fn(
                        file_stat_tracker.data, self.config)
                    file_stat_tracker.update_data(metrics_data)
                    file_stat_tracker.update_stat(metrics_stat)
                    self.stat_tracker.update_data(metrics_data)
                    self.stat_tracker.update_stat(metrics_stat)

                self.logger.info(f" {inference_filepath} ".center(80, "-"))
                self.logger.info(f"\t{file_stat_tracker.to_string()}")

                if self.config.save_wandb:
                    dict_to_log = {}
                    for k, v in file_stat_tracker.items():
                        key_to_log = f"{k}/infer"
                        dict_to_log[key_to_log] = np.mean(v)
                    wandb.log(data=dict_to_log, step=1)

                    for timestep, kv in file_stat_tracker.timed_items():
                        dict_to_log = {}
                        for k, v in kv.items():
                            key_to_log = f"{k}/infer"
                            dict_to_log[key_to_log] = np.mean(v)
                        wandb.log(data=dict_to_log, step=int(timestep))

        if is_rank_0() and compute_metrics_fn is not None and len(inference_filepath2data_loader) > 1:
            self.logger.info(" Overall ".center(80, "-"))
            self.logger.info(f"\t{self.stat_tracker.to_string()}")

            if self.config.save_wandb:
                dict_to_log = {}
                for k, v in self.stat_tracker.items():
                    key_to_log = f"{k}/infer"
                    dict_to_log[key_to_log] = np.mean(v)
                wandb.log(data=dict_to_log, step=1)

        if is_rank_0():
            self.logger.info("*".center(50, "*"))
            self.logger.info("Inference pipeline finished.")
