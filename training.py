

from argparse import (
    BooleanOptionalAction,
    Namespace
)

from configargparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM

from data import load_conversation_data_from_hf
from flow_llms import FlowLlamaForCausalLM, FlowQwen2ForCausalLM
from pipeline import (
    TrainingPipeline,
    get_arguments,
    is_rank_0,
    str2bool,
    str_or_float,
)

from pipeline.util import (
    build_tokenizer,
)
from tasks import (
    GSM8kDataProcessor,
    MMLUDataProcessor,
    SmolTalkDataProcessor,
    SharedTasks,
)
from utils import (
    calculate_binned_losses, estimate_nstep_cross_entropy,
    compute_accuracy_from_logits,
)


def build_ode_solver(config, **kwargs):
    if config.ode_solver in [
            'euler', 'midpoint', 'heun', 'heun2', 'heun3', 'rk4']:
        if config.ode_solver == 'heun':
            config.ode_solver = 'heun2'
        config.adaptive_solver = False
        ode_kwargs = dict(
            atol=config.ode_solver_atol,
            rtol=config.ode_solver_rtol,
            method=config.ode_solver
        )
    elif config.ode_solver in ['adaptive_heun', 'fehlberg2', 'bosh3']:
        config.adaptive_solver = True
        ode_kwargs = dict(
            atol=config.ode_solver_atol,
            rtol=config.ode_solver_rtol,
            method=config.ode_solver,

            options=dict(first_step=0.05),
        )
    else:
        print('ERROR: not supported ode solver specified')
        raise NotImplementedError
    return ode_kwargs


def build_model(
    config: Namespace,
    **kwargs
):
    config.adaptive_solver = False
    if config.lad:
        model_loading_kwargs = dict(
            device_map='cpu',

            torch_dtype=torch.bfloat16
        )

        config.num_data_categories = 1
        if config.use_data_split_guidance:
            assert config.guidance_categories is not None
            assert len(config.guidance_categories) > 0
            config.num_data_categories = len(config.guidance_categories)
            guidance_modulation_num_classes = config.num_data_categories
            if config.guidance_categories is not None:
                for c in config.guidance_categories:
                    assert c in SharedTasks
            if is_rank_0:
                print('Initialized classifier-free guidance with categories: '
                      f'{config.guidance_categories}')
        else:
            guidance_modulation_num_classes = 0

        if "llama" in config.pretrained_model_dir.lower() or (
                "llama" in config.tokenizer_dir.lower()):
            FlowModelClass = FlowLlamaForCausalLM
        elif "qwen" in config.pretrained_model_dir.lower() or (
                "qwen" in config.tokenizer_dir.lower()):
            FlowModelClass = FlowQwen2ForCausalLM
        else:
            raise ValueError(
                f"Flow model not implemented for {config.pretrained_model_dir}")
        ode_kwargs = build_ode_solver(config=config)
        model = FlowModelClass(


            model_or_model_id=config.pretrained_model_dir,
            model_loading_kwargs=model_loading_kwargs,
            base_params_to_freeze=config.base_params_to_freeze,
            noise_schedule=config.noise_schedule,
            minimum_training_noise=config.minimum_training_noise,
            minimum_training_noise_units=config.minimum_training_noise_units,
            flow_representation_space=config.flow_representation_space,
            flow_representation_dim=config.flow_representation_dim,
            flow_representation_num_layers=config.flow_representation_num_layers,
            flow_representation_normalize=config.flow_representation_normalize,
            flow_representation_rescaling=config.flow_representation_rescaling,
            noise_rescaling=config.noise_rescaling,
            flow_to_lm_translation_depth=config.flow_to_lm_translation_depth,
            flow_to_lm_hidden_size=config.flow_to_lm_hidden_size,
            flow_to_lm_timestep_rescaling=config.flow_to_lm_timestep_rescaling,
            flow_to_lm_rescale_in_float32=config.flow_to_lm_rescale_in_float32,
            preserve_behavior_at_flow_start=config.preserve_behavior_at_flow_start,
            modulate_hidden_states=config.modulate_hidden_states,
            full_dit_modulation=config.full_dit_modulation,
            timestep_modulation_num_layers=config.timestep_modulation_num_layers,
            timestep_modulation_freq_embedding_size=config.timestep_modulation_freq_embedding_size,
            timestep_modulation_hidden_size=config.timestep_modulation_hidden_size,
            guidance_modulation_num_classes=guidance_modulation_num_classes,
            guidance_modulation_training_dropout=config.guidance_modulation_training_dropout,
            freeze_modulation_at_flow_start=config.freeze_modulation_at_flow_start,
            separate_flow_params=config.separate_flow_params,
            separate_flow_params_with_lora=config.separate_flow_params_with_lora,
            flow_lora_rank=config.flow_lora_rank,
            flow_lora_alpha=config.flow_lora_alpha,
            ode_kwargs=ode_kwargs,
            nstep_final_timestep=config.nstep_final_timestep,
            nstep_x1_estimation=config.nstep_x1_estimation,
            nstep_normalize_x1_predictions=config.nstep_normalize_x1_predictions,
            nstep_clamp_predictions=config.nstep_clamp_predictions,
            nstep_temperature_schedule=config.nstep_temperature_schedule,
            nstep_guidance_parameter=config.nstep_guidance_parameter,
            reinit_flow_params=config.reinit_flow_params,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_dir,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation=config.attn_implementation
        )
    try:
        _is_rank0 = is_rank_0()
    except ValueError:
        _is_rank0 = True
    if _is_rank0:
        print(model)
    return model


def train_forward_step(
    step: int,
    model: object,
    tokenizer: object,
    batch_data: list,
    config: Namespace,
    **kwargs
):
    ret_data, ret_stat = {}, {}

    compute_binned_loss = (
        config.num_t_bins is not None and config.num_t_bins > 0 and config.lad)

    input_ids, labels, condition_label_ids = [], [], []
    for item in batch_data:
        context = item["context"]
        response = item["response"]
        condition_label_id = item["category_id"]
        input_ids.append(context + response)
        labels.append([-100] * len(context[1:]) +
                      response + [tokenizer.eos_token_id])
        condition_label_ids.append(condition_label_id)

    max_seq_len = max([len(seq) for seq in input_ids])
    input_ids = [seq + [tokenizer.eos_token_id] *
                 (max_seq_len - len(seq)) for seq in input_ids]
    labels = [seq + [-100] * (max_seq_len - len(seq)) for seq in labels]

    input_ids = torch.LongTensor(input_ids).to(model.device)
    labels = torch.LongTensor(labels).to(model.device)
    if condition_label_ids[0] is not None:
        condition_label_ids = torch.LongTensor(
            condition_label_ids).to(model.device)
    else:
        condition_label_ids = None

    if config.lad:
        model_outputs = model(
            input_ids=input_ids,
            labels=labels,
            class_labels=condition_label_ids,
            shifted_labels=True,

            compute_loss_for_labels=False,
            return_dict=True,
            output_hidden_states=config.compute_diff_at_training,
        )
        logits = model_outputs["logits"]
    else:
        model_outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_hidden_states=config.compute_diff_at_training,
        )
        logits = model_outputs["logits"]

    if not compute_binned_loss:

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
    else:
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none"
        ).view(labels.size())
        valid_mask = labels != -100

        loss = losses.sum() / valid_mask.sum()
        with torch.no_grad():
            ppl = loss.exp()

        timesteps = model_outputs["timesteps"]
        binned_stat = calculate_binned_losses(
            losses, valid_mask, timesteps, num_bins=config.num_t_bins
        )
        ret_stat.update(binned_stat)

    with torch.no_grad():
        ppl = loss.exp()

    ret_stat["loss"] = loss.item()
    ret_stat["ppl"] = ppl.item()

    if config.lad and config.compute_diff_at_training:
        hidden_states = model_outputs["hidden_states"]
        with torch.no_grad():
            model_outputs_wo_flow = model(
                input_ids=input_ids,
                return_dict=True,
                output_hidden_states=True
            )
            logits_wo_flow = model_outputs_wo_flow["logits"]
            hidden_states_wo_flow = model_outputs_wo_flow["hidden_states"]
            loss_wo_flow = torch.nn.functional.cross_entropy(
                logits_wo_flow.view(-1, logits_wo_flow.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            ppl_wo_flow = loss_wo_flow.exp()

            logits_diff = (logits - logits_wo_flow).abs().mean()
            hidden_states_diff = [
                hs - hswf
                for hs, hswf in zip(hidden_states, hidden_states_wo_flow)
            ]
            hidden_states_diff = sum([
                hsd.abs().mean()
                for hsd in hidden_states_diff
            ]) / len(hidden_states_diff)
            ppl_diff = ppl - ppl_wo_flow

        ret_stat["logits_diff"] = logits_diff.item()
        ret_stat["h_diff"] = hidden_states_diff.item()
        ret_stat["ppl_diff"] = ppl_diff.item()
        ret_stat["ppl_wo_flow"] = ppl_wo_flow.item()

    return loss, ret_data, ret_stat


def valid_forward_step(
    step: int,
    model: object,
    tokenizer: object,
    batch_data: list,
    config: Namespace | dict,
    **kwargs
):
    if not isinstance(config, Namespace):

        config = Namespace(**config)

    ret_data, ret_stat = {}, {}

    compute_binned_loss = (
        config.num_t_bins is not None and config.num_t_bins > 0 and config.lad)

    input_ids, labels, condition_label_ids = [], [], []
    for item in batch_data:
        context = item["context"]
        response = item["response"]
        condition_label_id = item["category_id"]
        input_ids.append(context + response)
        labels.append(
            [-100] * len(context[1:])
            + response
            + [tokenizer.eos_token_id]
        )
        condition_label_ids.append(condition_label_id)

    max_seq_len = max([len(seq) for seq in input_ids])
    input_ids = [
        seq + [tokenizer.eos_token_id] * (max_seq_len - len(seq))
        for seq in input_ids
    ]
    labels = [seq + [-100] * (max_seq_len - len(seq)) for seq in labels]

    input_ids = torch.LongTensor(input_ids).to(model.device)
    labels = torch.LongTensor(labels).to(model.device)
    if condition_label_ids[0] is not None:
        condition_label_ids = torch.LongTensor(
            condition_label_ids).to(model.device)
    else:
        condition_label_ids = None

    collect_nstep = (config.compute_nstep_loss is not None and
                     config.compute_nstep_loss > 0)

    if config.lad:

        model_outputs = model(
            input_ids=input_ids,
            class_labels=condition_label_ids,
            labels=labels,
            shifted_labels=True,

            compute_loss_for_labels=False,
            return_dict=True,
            output_hidden_states=False,
            use_cache=collect_nstep,
            output_cached_final_hidden_states=collect_nstep,
        )
        logits = model_outputs["logits"]

        if config.use_data_split_guidance:

            model_outputs_wo_guidance = model(
                input_ids=input_ids,
                class_labels=None,
                labels=labels,
                shifted_labels=True,

                compute_loss_for_labels=False,
                return_dict=True,
                output_hidden_states=False,
                use_cache=collect_nstep,
                output_cached_final_hidden_states=collect_nstep,
            )

            logits_wo_guidance = model_outputs_wo_guidance["logits"]

            loss_wo_guidance = torch.nn.functional.cross_entropy(
                logits_wo_guidance.view(-1, logits_wo_guidance.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            ppl_wo_guidance = loss_wo_guidance.exp()

            ret_stat["loss_wo_guidance"] = loss_wo_guidance.item()
            ret_stat["ppl_wo_guidance"] = ppl_wo_guidance.item()

            expected_accuracy, greedy_accuracy = compute_accuracy_from_logits(
                logits=logits_wo_guidance,
                labels=labels,
                temperature=1.0,
                ignore_index=-100,
            )
            ret_stat[f"acc_expected_wo_guidance"] = expected_accuracy
            ret_stat[f"acc_greedy_wo_guidance"] = greedy_accuracy

        if collect_nstep:

            cached_final_hidden_states = (
                model_outputs.cached_final_hidden_states)
            past_key_values = model_outputs.past_key_values

            flow_inference_output = model.nstep_inference(
                input_ids=input_ids,
                class_labels=condition_label_ids,
                ode_steps=config.compute_nstep_loss,
                initial_flow_representation_embeds=None,
                precompute_non_flow_path=True,
                cached_final_hidden_states=cached_final_hidden_states,
                past_key_values=past_key_values
            )
            flow_embeds_trajectory = flow_inference_output.flow_trajectory
            timestep_trajectory = flow_inference_output.timestep_trajectory
            flow_embeds = flow_embeds_trajectory[-1].to(model.dtype)
            timesteps = timestep_trajectory[-1]

            nstep_model_outputs = model(
                input_ids=input_ids,
                class_labels=condition_label_ids,
                timesteps=timesteps,
                flow_representation_embeds=flow_embeds,
                return_dict=True,
                cached_final_hidden_states=cached_final_hidden_states,
                past_key_values=past_key_values,
            )
            nstep_logits = nstep_model_outputs["logits"]

            nstep_loss = torch.nn.functional.cross_entropy(
                nstep_logits.view(-1, nstep_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            nstep_ppl = nstep_loss.exp()
            ret_stat[
                f"loss/{config.compute_nstep_loss}_nstep"] = nstep_loss.item()
            ret_stat[
                f"ppl/{config.compute_nstep_loss}_nstep"] = nstep_ppl.item()

            expected_accuracy, greedy_accuracy = compute_accuracy_from_logits(
                logits=nstep_logits,
                labels=labels,
                temperature=1.0,
                ignore_index=-100,
            )
            ret_stat[f"acc_expected/{config.compute_nstep_loss}_nstep"] = (
                expected_accuracy)
            ret_stat[f"acc_greedy/{config.compute_nstep_loss}_nstep"] = (
                greedy_accuracy)
    else:

        model_outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_hidden_states=False
        )
        logits = model_outputs["logits"]

    if not compute_binned_loss:

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
    else:
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none"
        ).view(labels.size())

        valid_mask = labels != -100

        loss = losses.sum() / valid_mask.sum()

        timesteps = model_outputs["timesteps"]
        binned_stat = calculate_binned_losses(
            losses, valid_mask, timesteps, num_bins=config.num_t_bins
        )
        ret_stat.update(binned_stat)

    ppl = loss.exp()
    ret_stat["loss"] = loss.item()
    ret_stat["ppl"] = ppl.item()

    expected_accuracy, greedy_accuracy = compute_accuracy_from_logits(
        logits=logits,
        labels=labels,
        temperature=1.0,
        ignore_index=-100,
    )
    ret_stat[f"acc_expected"] = expected_accuracy
    ret_stat[f"acc_greedy"] = greedy_accuracy

    if config.lad and config.compute_diff_at_training:

        model_outputs_wo_flow = model(
            input_ids=input_ids,
            return_dict=True,
            output_hidden_states=False
        )
        logits_wo_flow = model_outputs_wo_flow["logits"]

        loss_wo_flow = torch.nn.functional.cross_entropy(
            logits_wo_flow.view(-1, logits_wo_flow.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
        ppl_wo_flow = loss_wo_flow.exp()

        loss_diff = loss - loss_wo_flow
        ppl_diff = ppl - ppl_wo_flow

        ret_stat["loss_wo_flow"] = loss_wo_flow.item()
        ret_stat["loss_diff"] = loss_diff.item()
        ret_stat["ppl_diff"] = ppl_diff.item()
        ret_stat["ppl_wo_flow"] = ppl_wo_flow.item()

        expected_accuracy, greedy_accuracy = compute_accuracy_from_logits(
            logits=logits_wo_flow,
            labels=labels,
            temperature=1.0,
            ignore_index=-100,
        )
        ret_stat[f"acc_expected_wo_flow"] = expected_accuracy
        ret_stat[f"acc_greedy_wo_flow"] = greedy_accuracy

    return ret_data, ret_stat


def get_custom_arguments(parser: ArgumentParser):

    parser.add_argument("--lad", action="store_true")
    parser.add_argument(
        "--base_params_to_freeze",
        type=str, default="all"
    )
    parser.add_argument(
        "--noise_schedule",
        type=str, default="identity"
    )
    parser.add_argument(
        "--minimum_training_noise",
        type=float, default=None
    )
    parser.add_argument(
        "--minimum_training_noise_units",
        type=str, default="time"
    )
    parser.add_argument(
        "--flow_representation_space",
        type=str, default="mapping"
    )
    parser.add_argument(
        "--flow_representation_dim",
        type=int, default=256
    )
    parser.add_argument(
        "--flow_representation_num_layers",
        type=int, default=1
    )
    parser.add_argument(
        "--flow_representation_normalize",
        type=str2bool, default=True
    )
    parser.add_argument(
        "--flow_representation_rescaling",
        type=str, default='mult'
    )
    parser.add_argument(
        "--noise_rescaling",
        type=float, default=256.0
    )
    parser.add_argument(
        "--flow_to_lm_translation_depth",
        type=int, default=2
    )
    parser.add_argument(
        "--flow_to_lm_hidden_size",
        type=int, default=256
    )
    parser.add_argument(
        "--flow_to_lm_timestep_rescaling",
        type=float, default=1.0
    )
    parser.add_argument(
        "--flow_to_lm_rescale_in_float32",
        type=str2bool, default=True
    )
    parser.add_argument(
        "--preserve_behavior_at_flow_start",
        type=str2bool, default=True
    )
    parser.add_argument(
        "--modulate_hidden_states",
        type=str2bool, default=False
    )
    parser.add_argument(
        "--full_dit_modulation",
        type=str2bool, default=True
    )
    parser.add_argument(
        "--timestep_modulation_num_layers",
        type=int, default=2
    )
    parser.add_argument(
        "--timestep_modulation_freq_embedding_size",
        type=int, default=256
    )
    parser.add_argument(
        "--timestep_modulation_hidden_size",
        type=int, default=2048
    )
    parser.add_argument(
        "--use_data_split_guidance",
        type=str2bool, default=False
    )

    parser.add_argument(
        "--guidance_categories",
        type=str, nargs="*"
    )
    parser.add_argument(
        "--guidance_modulation_training_dropout",
        type=float, default=0.2
    )
    parser.add_argument(
        "--nstep_guidance_parameter",
        type=float, default=1.0
    )
    parser.add_argument(
        "--freeze_modulation_at_flow_start",
        type=str2bool, default=True
    )
    parser.add_argument(
        "--separate_flow_params",
        type=str2bool, default=True
    )
    parser.add_argument(
        "--separate_flow_params_with_lora",
        type=str2bool, default=False
    )
    parser.add_argument(
        "--flow_lora_rank",
        type=int, default=32
    )
    parser.add_argument(
        "--flow_lora_alpha",
        type=float, default=None
    )

    parser.add_argument(
        "--compute_diff_at_training",
        type=str2bool, default=False
    )

    parser.add_argument(
        "--compute_nstep_loss",
        type=int, default=8
    )

    parser.add_argument(
        "--nstep_final_timestep",
        type=float, default=1.0
    )

    parser.add_argument(
        "--nstep_x1_estimation",
        type=str, default='sample'
    )

    parser.add_argument(
        "--nstep_normalize_x1_predictions",
        type=str2bool, default=False
    )

    parser.add_argument(
        "--nstep_clamp_predictions",
        type=str2bool, default=False
    )

    parser.add_argument(
        "--nstep_temperature_schedule",
        type=str_or_float, default=0.0
    )

    parser.add_argument(
        "--num_t_bins",
        type=int, default=None
    )

    parser.add_argument(
        "--exclude_train_data_category",
        type=str, nargs="*"
    )

    parser.add_argument(
        "--ode_solver",
        type=str, default='midpoint'
    )

    parser.add_argument(
        "--ode_solver_atol",
        type=float, default=1e-5
    )

    parser.add_argument(
        "--ode_solver_rtol",
        type=float, default=1e-5
    )

    parser.add_argument(
        "--reinit_flow_params",
        type=str2bool, default=False
    )
    return parser


def main(
    config: Namespace,
    local_rank: int
):
    data_processor_classes = []
    for filepath in config.train_filepaths:
        if "smoltalk" in filepath.lower():
            data_processor_classes.append(SmolTalkDataProcessor)
        elif "mmlu" in filepath.lower():
            data_processor_classes.append(MMLUDataProcessor)
        elif "gsm8k" in filepath.lower():
            data_processor_classes.append(GSM8kDataProcessor)
    valid_data_processor_classes = []
    for filepath in config.valid_filepaths:
        if "smoltalk" in filepath.lower():
            valid_data_processor_classes.append(SmolTalkDataProcessor)
        elif "mmlu" in filepath.lower():
            valid_data_processor_classes.append(MMLUDataProcessor)
        elif "gsm8k" in filepath.lower():
            valid_data_processor_classes.append(GSM8kDataProcessor)

    training_pipeline = TrainingPipeline(
        config=config,
        world_size=config.world_size,
        local_rank=local_rank,
        global_rank=config.global_rank,
    )

    training_pipeline.run(
        build_model_fn=build_model,
        build_tokenizer_fn=build_tokenizer,
        load_data_from_filepath_fn=load_conversation_data_from_hf,
        data_processor_classes=data_processor_classes,
        valid_data_processor_classes=valid_data_processor_classes,
        train_forward_step_fn=train_forward_step,
        valid_forward_step_fn=valid_forward_step,
    )


if __name__ == "__main__":
    config = get_arguments(
        get_custom_arguments,
        default_config_files=["cfgs/training.cfg"]
    )
    main(config, config.local_rank)
