import abc
from typing import Optional
import torch
from torch import nn
import math
import gc


def empty_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def add_pad_token(tokenizer, model, reserved_token_idx=0):
    new_created_token = tokenizer.add_special_tokens(
        {"pad_token": f"<|reserved_special_token_{reserved_token_idx}|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    return new_created_token


def estimate_nstep_cross_entropy(
        num_samples, model, input_ids, labels, ignore_index=-100, **kwargs):
    raise NotImplementedError


def compute_accuracy_from_logits(
        logits, labels, temperature=1.0, ignore_index=-100):
    labels = labels.view(-1)
    logits = logits.view(-1, logits.size(-1))

    relevant_indices = ~(labels == ignore_index)

    labels = labels[relevant_indices]
    logits = logits[relevant_indices]

    probabilities = torch.softmax(logits.float()/temperature, dim=-1)
    correct_probabilities = torch.gather(
        probabilities, dim=-1, index=labels.unsqueeze(-1))
    expected_accuracy = torch.mean(correct_probabilities).item()

    greedy_preds = torch.argmax(logits, dim=-1)
    greedy_accuracy = (greedy_preds == labels).float().mean().item()

    return expected_accuracy, greedy_accuracy


def calculate_binned_losses(loss, valid_mask, timesteps, num_bins=10):

    assert num_bins > 0

    bin_edges = torch.linspace(0, 1, num_bins + 1, device=loss.device)

    eps = 1e-6
    bin_edges[-1] = bin_edges[-1] + eps
    timestep_bins = torch.bucketize(timesteps, bin_edges, right=True) - 1

    bin_losses = torch.zeros(num_bins, device=loss.device)
    bin_counts = torch.zeros(num_bins, device=loss.device, dtype=torch.long)

    for bin_idx in range(num_bins):
        bin_mask = (timestep_bins == bin_idx) & valid_mask
        if bin_mask.any():
            bin_losses[bin_idx] = loss[bin_mask].sum()
            bin_counts[bin_idx] = bin_mask.sum()

    bin_losses = torch.where(
        bin_counts > 0,
        bin_losses / bin_counts.float(),
        torch.tensor(float('nan'), device=loss.device)
    )

    ret_stat = {}
    for i in range(len(bin_losses)):
        start = i / len(bin_losses)
        end = (i + 1) / len(bin_losses)
        key_range = f"{start:.1f}-{end:.1f}"
        loss_val = bin_losses[i]
        ppl = loss_val.exp()
        count = bin_counts[i]
        ret_stat[f"binned_ppl/t{key_range}"] = ppl.item()
        ret_stat[f"binned_count/t{key_range}"] = count.item()

    return ret_stat
