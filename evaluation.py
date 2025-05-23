

from argparse import (
    ArgumentParser,
    Namespace
)
from collections import defaultdict

import torch
import numpy as np

from data import load_conversation_data_from_hf
from pipeline import (
    InferencePipeline,
    get_inference_arguments,
    str2bool,
)
from tasks import (
    GSM8kDataProcessor,
    MMLUDataProcessor,
    SmolTalkDataProcessor,
)
from training import (
    build_model,
    get_custom_arguments
)
from utils import (
    calculate_binned_losses, compute_accuracy_from_logits)


def forward_step(
    model: object,
    tokenizer: object,
    batch_data: list,
    config: Namespace,
    **kwargs
):
    ret_data = {}
    ret_stat = {}
    ret_timed_stat = defaultdict(dict)

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
            output_hidden_states=True,
            use_cache=collect_nstep,
            output_cached_final_hidden_states=collect_nstep,
        )
        logits = model_outputs["logits"]
        hidden_states = model_outputs["hidden_states"]
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none"
        ).view(labels.size())
        valid_mask = labels != -100

        loss = losses.sum() / valid_mask.sum()

        if config.num_t_bins is not None:
            timesteps = model_outputs["timesteps"]
            binned_stat = calculate_binned_losses(
                losses, valid_mask, timesteps, num_bins=config.num_t_bins
            )
            ret_stat.update(binned_stat)

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

    else:

        model_outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_hidden_states=True
        )
        logits = model_outputs["logits"]
        hidden_states = model_outputs["hidden_states"]

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )

    with torch.no_grad():
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

    if config.lad and collect_nstep:

        cached_final_hidden_states = (
            model_outputs.cached_final_hidden_states)
        past_key_values = model_outputs.past_key_values

        if config.provide_targets_at_t is not None:
            start_timestep = config.provide_targets_at_t*torch.ones(
                size=input_ids.shape,
                device=model.device,
            )
            assert (config.provide_targets_at_t >= 0 and
                    config.provide_targets_at_t <= 1)
            _, init_flow_reprs = model.sample_timestep_and_flow_embeds(
                labels=labels,
                timesteps=start_timestep,
            )
        else:
            start_timestep = 0
            init_flow_reprs = None

        flow_inference_output = model.nstep_inference(
            input_ids=input_ids,
            class_labels=condition_label_ids,
            start_timestep=start_timestep,
            ode_steps=config.compute_nstep_loss,
            initial_flow_representation_embeds=init_flow_reprs,
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
        nstep_ppl = nstep_loss.exp().item()
        nstep_loss = nstep_loss.item()
        ret_stat[f"loss/{config.compute_nstep_loss}_nstep"] = nstep_loss
        ret_stat[f"ppl/{config.compute_nstep_loss}_nstep"] = nstep_ppl
        ret_stat[f"loss/final_step"] = nstep_loss
        ret_stat[f"ppl/final_step"] = nstep_ppl

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
        ret_stat[f"acc_expected/final_step"] = expected_accuracy
        ret_stat[f"acc_greedy/final_step"] = greedy_accuracy

        if config.nstep_evaluate_intermediate_steps:
            all_losses = []
            all_ppls = []
            all_timesteps = []
            all_expected_accs = []
            all_greedy_accs = []
            for trajectory_idx in range(len(flow_embeds_trajectory)):
                flow_embeds = flow_embeds_trajectory[trajectory_idx].to(
                    model.dtype)
                timesteps = timestep_trajectory[trajectory_idx]

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
                nstep_ppl = nstep_loss.exp().item()
                nstep_loss = nstep_loss.item()
                time_float = timesteps.item()
                expected_accuracy, greedy_accuracy = (
                    compute_accuracy_from_logits(
                        logits=nstep_logits,
                        labels=labels,
                        temperature=1.0,
                        ignore_index=-100,
                    ))

                time_int_x100 = max(1, int(time_float*100))

                ret_timed_stat[trajectory_idx][
                    f"loss/intermediate_nstep"] = nstep_loss
                ret_timed_stat[trajectory_idx][
                    f"ppl/intermediate_nstep"] = nstep_ppl
                ret_timed_stat[trajectory_idx][
                    f"acc_expected/intermediate_nstep"] = expected_accuracy
                ret_timed_stat[trajectory_idx][
                    f"acc_greedy/intermediate_nstep"] = greedy_accuracy

                ret_timed_stat[trajectory_idx][f"time"] = time_float
                ret_timed_stat[trajectory_idx][f"time_int"] = time_int_x100

                all_losses.append(nstep_loss)
                all_ppls.append(nstep_ppl)
                all_timesteps.append(time_float)
                all_expected_accs.append(expected_accuracy)
                all_greedy_accs.append(greedy_accuracy)

            best_loss_index = np.argmin(all_losses)
            best_loss = all_losses[best_loss_index]
            best_ppl = all_ppls[best_loss_index]
            best_timestep = all_timesteps[best_loss_index]
            ret_stat[f"loss/best_step"] = best_loss
            ret_stat[f"ppl/best_step"] = best_ppl
            ret_stat[f"time/loss/best"] = best_timestep
            ret_stat[f"index/loss/best_out_of_{config.compute_nstep_loss}"] = (
                best_loss_index)

            best_expected_acc_index = np.argmax(all_expected_accs)
            best_expected_acc = all_expected_accs[best_expected_acc_index]
            best_timestep = all_timesteps[best_expected_acc_index]

            ret_stat[f"acc_expected/best_step"] = best_expected_acc
            ret_stat[f"time/acc_expected/best"] = best_timestep
            ret_stat["index/acc_expected/best_out_of_"
                     f"{config.compute_nstep_loss}"] = best_expected_acc_index

            best_greedy_acc_index = np.argmax(all_greedy_accs)
            best_greedy_acc = all_greedy_accs[best_greedy_acc_index]
            best_timestep = all_timesteps[best_greedy_acc_index]

            ret_stat[f"acc_greedy/best_step"] = best_greedy_acc
            ret_stat[f"time/acc_greedy/best"] = best_timestep
            ret_stat["index/acc_greedy/best_out_of_"
                     f"{config.compute_nstep_loss}"] = best_greedy_acc_index

        if config.use_data_split_guidance:
            flow_inference_output = model.nstep_inference(
                input_ids=input_ids,
                class_labels=None,
                start_timestep=start_timestep,
                ode_steps=config.compute_nstep_loss,
                initial_flow_representation_embeds=init_flow_reprs,
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
                class_labels=None,
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
            nstep_ppl = nstep_loss.exp().item()
            nstep_loss = nstep_loss.item()
            ret_stat[f"loss_wo_guidance/{config.compute_nstep_loss}_nstep"] = nstep_loss
            ret_stat[f"ppl_wo_guidance/{config.compute_nstep_loss}_nstep"] = nstep_ppl
            ret_stat[f"loss_wo_guidance/final_step"] = nstep_loss
            ret_stat[f"ppl_wo_guidance/final_step"] = nstep_ppl

            expected_accuracy, greedy_accuracy = compute_accuracy_from_logits(
                logits=nstep_logits,
                labels=labels,
                temperature=1.0,
                ignore_index=-100,
            )
            ret_stat[f"acc_expected_wo_guidance/{config.compute_nstep_loss}_nstep"] = (
                expected_accuracy)
            ret_stat[f"acc_greedy_wo_guidance/{config.compute_nstep_loss}_nstep"] = (
                greedy_accuracy)
            ret_stat[f"acc_expected_wo_guidance/final_step"] = expected_accuracy
            ret_stat[f"acc_greedy_wo_guidance/final_step"] = greedy_accuracy

            if config.nstep_evaluate_intermediate_steps:
                all_losses = []
                all_ppls = []
                all_timesteps = []
                all_expected_accs = []
                all_greedy_accs = []
                for trajectory_idx in range(len(flow_embeds_trajectory)):
                    flow_embeds = flow_embeds_trajectory[trajectory_idx].to(
                        model.dtype)
                    timesteps = timestep_trajectory[trajectory_idx]

                    nstep_model_outputs = model(
                        input_ids=input_ids,
                        class_labels=None,
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
                    nstep_ppl = nstep_loss.exp().item()
                    nstep_loss = nstep_loss.item()
                    time_float = timesteps.item()
                    expected_accuracy, greedy_accuracy = (
                        compute_accuracy_from_logits(
                            logits=nstep_logits,
                            labels=labels,
                            temperature=1.0,
                            ignore_index=-100,
                        ))

                    time_int_x100 = max(1, int(time_float*100))

                    ret_timed_stat[trajectory_idx][
                        f"loss_wo_guidance/intermediate_nstep"] = nstep_loss
                    ret_timed_stat[trajectory_idx][
                        f"ppl_wo_guidance/intermediate_nstep"] = nstep_ppl
                    ret_timed_stat[trajectory_idx][
                        f"acc_expected_wo_guidance/intermediate_nstep"] = expected_accuracy
                    ret_timed_stat[trajectory_idx][
                        f"acc_greedy_wo_guidance/intermediate_nstep"] = greedy_accuracy

                    ret_timed_stat[trajectory_idx][f"time"] = time_float
                    ret_timed_stat[trajectory_idx][f"time_int"] = time_int_x100

                    all_losses.append(nstep_loss)
                    all_ppls.append(nstep_ppl)
                    all_timesteps.append(time_float)
                    all_expected_accs.append(expected_accuracy)
                    all_greedy_accs.append(greedy_accuracy)

                best_loss_index = np.argmin(all_losses)
                best_loss = all_losses[best_loss_index]
                best_ppl = all_ppls[best_loss_index]
                best_timestep = all_timesteps[best_loss_index]
                ret_stat[f"loss_wo_guidance/best_step"] = best_loss
                ret_stat[f"ppl_wo_guidance/best_step"] = best_ppl
                ret_stat[f"time/loss_wo_guidance/best"] = best_timestep
                ret_stat[f"index/loss_wo_guidance/best_out_of_{config.compute_nstep_loss}"] = (
                    best_loss_index)

                best_expected_acc_index = np.argmax(all_expected_accs)
                best_expected_acc = all_expected_accs[best_expected_acc_index]
                best_timestep = all_timesteps[best_expected_acc_index]

                ret_stat[f"acc_expected_wo_guidance/best_step"] = best_expected_acc
                ret_stat[f"time/acc_expected_wo_guidance/best"] = best_timestep
                ret_stat["index/acc_expected_wo_guidance/best_out_of_"
                         f"{config.compute_nstep_loss}"] = best_expected_acc_index

                best_greedy_acc_index = np.argmax(all_greedy_accs)
                best_greedy_acc = all_greedy_accs[best_greedy_acc_index]
                best_timestep = all_timesteps[best_greedy_acc_index]

                ret_stat[f"acc_greedy_wo_guidance/best_step"] = best_greedy_acc
                ret_stat[f"time/acc_greedy_wo_guidance/best"] = best_timestep
                ret_stat["index/acc_greedy_wo_guidance/best_out_of_"
                         f"{config.compute_nstep_loss}"] = best_greedy_acc_index

    if config.lad:

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
            hsd.abs().mean() for hsd in hidden_states_diff
        ]) / len(hidden_states_diff)
        loss_diff = loss - loss_wo_flow
        ppl_diff = ppl - ppl_wo_flow

        ret_stat["logits_diff"] = logits_diff.item()
        ret_stat["h_diff"] = hidden_states_diff.item()
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

    return ret_data, ret_stat, ret_timed_stat


def get_eval_custom_arguments(parser: ArgumentParser):
    get_custom_arguments(parser)

    parser.add_argument(
        "--provide_targets_at_t",
        type=float, default=None
    )

    parser.add_argument(
        "--nstep_evaluate_intermediate_steps",
        type=str2bool, default=False
    )

    return parser


def main(
    config: Namespace,
    local_rank: int
):
    data_processor_classes = []
    for filepath in config.filepaths:
        if "smoltalk" in filepath.lower():
            data_processor_classes.append(SmolTalkDataProcessor)
        elif "gsm8k" in filepath.lower():
            data_processor_classes.append(GSM8kDataProcessor)
        elif "mmlu" in filepath.lower():
            data_processor_classes.append(MMLUDataProcessor)

    config.num_data_categories = data_processor_classes[0].num_categories
    if config.use_data_split_guidance:
        assert len(data_processor_classes) == 1

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
    )


if __name__ == "__main__":
    config = get_inference_arguments(
        get_eval_custom_arguments,
        default_config_files=["cfgs/eval.cfg"]
    )
    main(config, config.local_rank)
