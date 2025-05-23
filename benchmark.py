

from argparse import (
    ArgumentParser,
    Namespace
)
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from data import load_conversation_data_from_hf
from pipeline import (
    InferencePipeline,
    autoregressive_decode,
    get_inference_arguments,
    str2bool,
)

from tasks.utils import extract_fraction_components
from tasks import (
    ARCDataProcessor,
    ARCEvalProcessor,
    GSM8kDataProcessor,
    GSM8kEvalProcessor,
    MATHDataProcessor,
    MATHEvalProcessor,
    MMLUDataProcessor,
    MMLUEvalProcessor,
    MMLUProDataProcessor,
    MMLUProEvalProcessor,
    PIQADataProcessor,
    PIQAEvalProcessor,
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

    task_names = data["task_names"]
    refs = data["ref_completions"]
    hyps = data["hyp_completions"]
    output_spaces = data["output_spaces"]

    def get_task_name2stats(hyps_to_evaluate):
        task_name2stats = defaultdict(lambda: defaultdict(int))
        for idx in range(len(hyps_to_evaluate)):
            task_name: str = task_names[idx]
            ref = refs[idx]
            hyp = hyps_to_evaluate[idx]
            output_space = output_spaces[idx]
            if task_name.startswith("mmlu"):
                evaluator_class = MMLUEvalProcessor
            elif task_name == "mmlu_pro":
                evaluator_class = MMLUProEvalProcessor
            elif task_name == "gsm8k":
                evaluator_class = GSM8kEvalProcessor
            elif task_name.startswith("MATH"):
                evaluator_class = MATHEvalProcessor
            elif task_name in ["arc-easy", "arc-challenge"]:
                evaluator_class = ARCEvalProcessor
            elif task_name == "piqa":
                evaluator_class = PIQAEvalProcessor
            else:
                raise ValueError(f"Unknown task name: {task_name}")
            extract_fn = evaluator_class.extract_answer_from_completion
            ref = extract_fn(ref, choices=output_space)
            hyp = extract_fn(hyp, choices=output_space)
            if ref is not None:
                task_name2stats[task_name]["n_total_hyps"] += 1
                if hyp is not None:
                    task_name2stats[task_name]["n_correct_format"] += 1
                    if ref == hyp:
                        task_name2stats[task_name]["n_correct_hyps"] += 1
                    elif task_name.startswith("MATH"):
                        try:
                            ref = float(ref)
                        except ValueError:
                            pass
                        try:
                            hyp = float(hyp)
                        except ValueError:
                            pass
                        if ref == hyp:
                            task_name2stats[task_name]["n_correct_hyps"] += 1
                        elif isinstance(ref, float) and isinstance(hyp, str):
                            if hyp.startswith('\\frac'):
                                num, den = extract_fraction_components(hyp)
                                if den is not None:
                                    ref_base_den = int(
                                        np.round(float(ref*den)))
                                    if ref_base_den == num:
                                        task_name2stats[task_name][
                                            "n_correct_hyps"] += 1
                        elif isinstance(hyp, float) and isinstance(ref, str):
                            if ref.startswith('\\frac'):
                                num, den = extract_fraction_components(ref)
                                if den is not None:
                                    hyp_base_den = int(
                                        np.round(float(hyp*den)))
                                    if hyp_base_den == num:
                                        task_name2stats[task_name][
                                            "n_correct_hyps"] += 1
        return task_name2stats

    task_name2stats = get_task_name2stats(hyps)
    for task_name, stats in task_name2stats.items():
        accuracy = stats["n_correct_hyps"] / stats["n_total_hyps"]
        format_accuracy = stats["n_correct_format"] / stats["n_total_hyps"]
        ret_stat[f"{task_name}/accuracy"] = accuracy
        ret_stat[f"{task_name}/format_accuracy"] = format_accuracy

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
            step_task_name2stats = get_task_name2stats(midway_ode_hyps)
            for task_name, stats in step_task_name2stats.items():
                accuracy = stats["n_correct_hyps"] / stats["n_total_hyps"]
                format_accuracy = stats["n_correct_format"] / \
                    stats["n_total_hyps"]
                ret_stat[f"{task_name}/accuracy/S{mid_ode_step}"] = accuracy
                ret_stat[f"{task_name}/format_accuracy/S{mid_ode_step}"] = format_accuracy

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
    input_ids, refs, condition_label_ids, task_names = [], [], [], []
    output_spaces = []
    for item in batch_data:
        context = item["context"]
        response = item["response"]
        condition_label_id = item["category_id"]
        task_name = item["task_name"]
        output_space = item["output_space"]
        input_ids.append(context)
        refs.append(response)
        task_names.append(task_name)
        condition_label_ids.append(condition_label_id)
        output_spaces.append(output_space)

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
        return_timestep_sequence=config.track_midway_ode_steps is not None,
        record_internal_ode_steps=config.record_internal_ode_steps,
    )
    output_ids = decode_outputs["output_id_seqs"]
    output_ids = [
        seq[:seq.index(tokenizer.eos_token_id)]
        if tokenizer.eos_token_id in seq else seq
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

    ret_data["ref_completions"] = ref_completions
    ret_data["hyp_completions"] = hyp_completions
    ret_data["task_names"] = task_names
    ret_data["output_spaces"] = output_spaces
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
                return_timestep_sequence=False,
                final_timestep=time_to_evaluate,
            )
            output_ids = decode_outputs_midway["output_id_seqs"]
            output_ids = [
                seq[:seq.index(tokenizer.eos_token_id)]
                if tokenizer.eos_token_id in seq else seq
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
        if filepath.startswith("cais/mmlu"):
            data_processor_classes.append(MMLUDataProcessor)
        elif filepath.startswith("TIGER-Lab/MMLU-Pro"):
            data_processor_classes.append(MMLUProDataProcessor)
        elif filepath.startswith("openai/gsm8k"):
            data_processor_classes.append(GSM8kDataProcessor)
        # NOTE: no longer available
        elif "MATH" in filepath:
            data_processor_classes.append(MATHDataProcessor)
        elif filepath.startswith("allenai/ai2_arc"):
            data_processor_classes.append(ARCDataProcessor)
        elif filepath.startswith("ybisk/piqa"):
            data_processor_classes.append(PIQADataProcessor)

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
        default_config_files=["cfgs/benchmark.cfg"]
    )
    if config.add_wandb_test_info and config.lad:
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
