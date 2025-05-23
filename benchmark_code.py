

from argparse import (
    ArgumentParser,
    Namespace
)
from collections import defaultdict
import json
import tempfile
import numpy as np

from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from human_eval.evaluation import evaluate_functional_correctness
import torch
from tqdm import tqdm

from data import load_conversation_data_from_hf
from pipeline import (
    InferencePipeline,
    autoregressive_decode,
    get_inference_arguments,
    str2bool,
)
from tasks import (
    HumanEvalDataProcessor,
    InstructHumanEvalDataProcessor,
    MBPPDataProcessor,
)
from training import (
    build_model,
    get_custom_arguments
)


def average_per_dimension(data):
    if not data:
        return []
    max_length = max(len(inner_list) for inner_list in data)

    sums = [0.0]*max_length
    counts = [0]*max_length

    for inner_list in data:
        for i, value in enumerate(inner_list):
            sums[i] += value
            counts[i] += 1
    averages = [sums[i] / counts[i] if counts[i] > 0 else 0.0
                for i in range(max_length)]
    return averages


def compute_metrics(data: dict, config: Namespace):
    ret_data = {}
    ret_stat = {}

    task_ids = data["task_ids"]
    task_names = data["task_names"]
    hyps = data["hyp_completions"]
    test_lists = data["test_lists"]

    task_name = task_names[0]
    assert len(set(task_names)) == 1, "All tasks should be the same"

    def get_stats(hyps_to_evaluate):
        cur_stats = {}

        if task_name in ["humaneval", "instructhumaneval"]:
            t = tempfile.NamedTemporaryFile(delete=False, mode="w+")
            for idx in range(len(hyps_to_evaluate)):
                task_id = task_ids[idx]
                hyp_list = hyps_to_evaluate[idx]
                for hyp in hyp_list:
                    task_hyp_jsonl = {
                        "task_id": task_id,
                        "completion": hyp,
                    }
                    t.write(json.dumps(task_hyp_jsonl) + "\n")
            t.close()
            result = evaluate_functional_correctness(
                sample_file=t.name,
                k=[1, 5, 10],
                n_workers=26*config.world_size,
                timeout=3.0,
                problem_file="tasks/HumanEval.jsonl.gz",
                ignore_incomplete=True,
            )
            for k, v in result.items():
                cur_stats[f"{task_name}/{k}"] = v

        elif task_name == "mbpp":
            import os
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            eval_references = []
            eval_predictions = []
            for idx in range(len(hyps_to_evaluate)):
                task_id = task_ids[idx]
                generations = hyps_to_evaluate[idx]
                references = "\n".join(test_lists[idx])
                eval_references.append(references)
                eval_predictions.append(generations)
            result, _ = compute_code_eval(
                predictions=eval_predictions,
                references=eval_references,
                k=[1, 5, 10],
                num_workers=26*config.world_size,
                timeout=3.0
            )
            for k, v in result.items():
                cur_stats[f"{task_name}/{k}"] = v
        return cur_stats
    stats = get_stats(hyps)
    ret_stat.update(stats)

    if 'lad_steps' in data:
        lad_steps_dict = {}
        hyps_steps = data['lad_steps']
        for idx, task_name in enumerate(task_names):
            lad_steps_dict[task_name] = lad_steps_dict.get(
                task_name, []) + [hyps_steps[idx]]
        for task_name, lad_steps_per_task in lad_steps_dict.items():
            ret_stat[f"lad_steps_{task_name}_mean"] = np.mean(
                [s for steps in lad_steps_per_task for s in steps])
            average_per_step_length = average_per_dimension(lad_steps_per_task)

            for i, s in enumerate(average_per_step_length):
                ret_stat[f"lad_steps_{task_name}/{i}"] = s
    else:
        hyps_steps = None

    if config.lad and config.track_midway_ode_steps:
        for mid_ode_step in tqdm(
            set(config.track_midway_ode_steps),
            desc="Evaluating midway ODE Steps"
        ):
            step_key = f"{mid_ode_step}/{config.compute_nstep_loss}"
            midway_ode_hyps = data[f"hyp_completions {step_key}"]
            stats = get_stats(midway_ode_hyps)
            for k, v in stats.items():
                ret_stat[f"{k}/S{mid_ode_step}"] = v

    return ret_data, ret_stat


def forward_step(
    model: object,
    tokenizer: object,
    batch_data: list,
    config: Namespace,
    **kwargs
):
    ret_data = {}
    ret_stat = {}
    ret_timed_stat = {}

    task_ids, input_ids, refs, condition_label_ids, task_names = [], [], [], [], []
    test_lists = []
    for item in batch_data:
        task_id = item["task_id"]
        context = item["context"]
        response = item["response"]
        condition_label_id = item["category_id"]
        task_name = item["task_name"]
        test_list = item["test_list"]
        task_ids.append(task_id)
        input_ids.append(context)
        refs.append(response)
        task_names.append(task_name)
        condition_label_ids.append(condition_label_id)
        test_lists.append(test_list)

    decode_outputs = autoregressive_decode(
        lad=config.lad,
        ode_steps=config.compute_nstep_loss,
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        class_labels=condition_label_ids,
        max_new_tokens=config.generation_max_len,
        do_sample=config.generation_do_sample,
        top_k=config.generation_top_k,
        top_p=config.generation_top_p,
        temperature=config.generation_temperature,
        num_return_sequences=config.generation_num_return_sequences,
        return_timestep_sequence=config.track_midway_ode_steps is not None,
        record_internal_ode_steps=config.record_internal_ode_steps,
    )
    output_ids = decode_outputs["output_id_seqs"]
    output_ids = [
        seq[:seq.index(tokenizer.eos_token_id)
            ] if tokenizer.eos_token_id in seq else seq
        for seq in output_ids
    ]
    output_ids = [
        list(filter(lambda x: x < len(tokenizer), seq)) for seq in output_ids
    ]

    ref_completions = tokenizer.batch_decode(
        refs,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    hyp_completions = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    _hyp_completions = []
    for i in range(len(refs)):
        task_hyps = []
        for j in range(config.generation_num_return_sequences):
            task_hyps.append(hyp_completions[i + j * len(refs)])
        _hyp_completions.append(task_hyps)
    hyp_completions = _hyp_completions

    ret_data["task_ids"] = task_ids
    ret_data["ref_completions"] = ref_completions
    ret_data["hyp_completions"] = hyp_completions
    ret_data["task_names"] = task_names
    ret_data["test_lists"] = test_lists
    if 'lad_steps' in decode_outputs:
        ret_data["lad_steps"] = [
            decode_outputs['lad_steps'] for _ in ref_completions]

    if config.lad and config.track_midway_ode_steps:
        timesteps_evaluated = decode_outputs['timestep_sequence']
        for midway_ode_step in set(config.track_midway_ode_steps):
            assert midway_ode_step > 0
            assert midway_ode_step < config.compute_nstep_loss
            num_steps = midway_ode_step
            time_to_evaluate = timesteps_evaluated[midway_ode_step - 1]
            if midway_ode_step == 1:
                num_steps = 0
            decode_outputs_midway = autoregressive_decode(
                lad=True,
                ode_steps=num_steps,
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                class_labels=condition_label_ids,
                max_new_tokens=config.generation_max_len,
                do_sample=config.generation_do_sample,
                top_k=config.generation_top_k,
                top_p=config.generation_top_p,
                temperature=config.generation_temperature,
                num_return_sequences=config.generation_num_return_sequences,
                return_timestep_sequence=(
                    config.track_midway_ode_steps is not None),
                final_timestep=time_to_evaluate,
            )
            output_ids = decode_outputs_midway["output_id_seqs"]
            output_ids = [
                seq[:seq.index(tokenizer.eos_token_id)
                    ] if tokenizer.eos_token_id in seq else seq
                for seq in output_ids
            ]
            output_ids = [
                list(filter(lambda x: x < len(tokenizer), seq))
                for seq in output_ids
            ]

            hyp_completions = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            _hyp_completions = []
            for i in range(len(refs)):
                task_hyps = []
                for j in range(config.generation_num_return_sequences):
                    task_hyps.append(hyp_completions[i + j * len(refs)])
                _hyp_completions.append(task_hyps)
            hyp_completions = _hyp_completions
            ret_data[
                f"hyp_completions {midway_ode_step}/{config.compute_nstep_loss}"
            ] = hyp_completions

    return ret_data, ret_stat, ret_timed_stat


def get_eval_custom_arguments(parser: ArgumentParser):
    get_custom_arguments(parser)
    parser.add_argument(
        "--num_few_shots",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--add_wandb_test_info",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--track_midway_ode_steps",
        type=int,
        nargs="+"
    )
    parser.add_argument(
        "--record_internal_ode_steps",
        type=str2bool,
        default=False,
    )
    return parser


def main(
    config: Namespace,
    local_rank: int
):
    data_processor_classes = []
    for filepath in config.filepaths:
        if filepath.startswith("openai/openai_humaneval"):
            data_processor_classes.append(HumanEvalDataProcessor)
        elif filepath.startswith("codeparrot/instructhumaneval"):
            data_processor_classes.append(InstructHumanEvalDataProcessor)
        elif filepath.startswith("google-research-datasets/mbpp"):
            data_processor_classes.append(MBPPDataProcessor)

    inference_pipeline = InferencePipeline(
        config=config,
        world_size=config.world_size,
        local_rank=local_rank,
        global_rank=config.global_rank,
    )

    inference_pipeline.run(
        build_model_fn=build_model,
        load_data_from_filepath_fn=load_conversation_data_from_hf,
        data_processor_classes=data_processor_classes,
        forward_step_fn=forward_step,
        compute_metrics_fn=compute_metrics
    )


if __name__ == "__main__":
    config = get_inference_arguments(
        get_eval_custom_arguments,
        default_config_files=["cfgs/benchmark_code.cfg"]
    )

    if config.add_wandb_test_info:
        if config.wandb_run_name is not None:
            config.wandb_run_name = (
                config.wandb_run_name +
                f'_{config.nstep_temperature_schedule}t' +
                f'_{config.compute_nstep_loss}s')
            if config.use_data_split_guidance:
                config.wandb_run_name = (
                    config.wandb_run_name +
                    f'_{config.nstep_guidance_parameter}G')

    main(config, config.local_rank)
