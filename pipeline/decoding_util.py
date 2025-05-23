

from collections import defaultdict
from typing import (
    Dict,
    List,
    Optional,
)
import copy
from peft import PeftModelForCausalLM
import torch
from transformers import (
    AutoTokenizer,
    DynamicCache,
    StoppingCriteria
)


class EOSCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        eos_flags = (input_ids == self.eos_token_id).sum(1) > 0
        eos_flag = (eos_flags.sum() == input_ids.size(0))
        return eos_flag


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None
        self.all_seqs = None

    def all_seq_set(self):
        if self.all_seqs is None:
            self.all_seqs = set()
            for seq in self:
                self.all_seqs.add(tuple(seq[:-1]))
            return self.all_seqs
        else:
            return self.all_seqs

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def autoregressive_decode(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: List[List[int]],
    class_labels: Optional[List[List[int]]] = None,
    min_new_tokens: int = 0,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    suppress_token_ids: List[int] = [],
    repetition_penalty: float = 1.0,
    repetition_max_n_gram: int = 6,
    num_return_sequences: int = 1,
    valid_token_trie: Trie = None,
    use_legacy_past_key_values: bool = False,
    other_output_keys: List[str] = [],
    lad: bool = False,
    ode_steps: int = 8,


    final_timestep: Optional[float] = None,
    return_timestep_sequence: bool = True,
    record_internal_ode_steps: bool = False,
):

    base_model = model
    if hasattr(model, 'module'):
        if isinstance(model.module, PeftModelForCausalLM):
            base_model = model.module.base_model.model

    if lad:

        assert repetition_penalty == 1
        assert top_k == 0
        assert do_sample
        assert top_p == 1.0

    batch_size = len(input_ids)
    seq_lens = [len(seq) for seq in input_ids]
    max_seq_len = max(seq_lens)
    num_virtual_tokens = 0

    position_ids = [
        list(range(num_virtual_tokens, seq_len + num_virtual_tokens))
        for seq_len in seq_lens
    ]

    input_ids = [
        [tokenizer.pad_token_id]*(max_seq_len-len(seq)) + seq
        for seq in input_ids
    ]
    position_ids = [
        [0]*(max_seq_len-len(seq)) + seq
        for seq in position_ids
    ]

    input_ids = torch.LongTensor(input_ids).to(model.device)
    input_embeds = None
    position_ids = torch.LongTensor(position_ids).to(model.device)
    past_key_values = None if use_legacy_past_key_values else DynamicCache()
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    repetition_tries = [Trie() for _ in range(batch_size)]
    if class_labels is not None:
        if class_labels[0] is not None:
            class_labels = torch.LongTensor(class_labels).to(model.device)
        else:
            class_labels = None

    pre_input_ids = input_ids[:, :-1] if input_ids is not None else None
    pre_input_embeds = input_embeds[:, :-
                                    1] if input_embeds is not None else None
    pre_position_ids = position_ids[:, :-1]
    pre_attention_mask = attention_mask[:, :-1]
    pre_past_key_values = past_key_values

    if lad and ode_steps > 0:

        pre_outputs = base_model.model.forward(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=pre_past_key_values,
            output_hidden_states=False,
            output_cached_final_hidden_states=True,
            use_cache=True
        )
    else:
        pre_outputs = base_model.model.forward(
            input_ids=pre_input_ids,
            inputs_embeds=pre_input_embeds,
            position_ids=pre_position_ids,
            attention_mask=pre_attention_mask,
            past_key_values=pre_past_key_values,
            output_hidden_states=False,
            use_cache=True
        )

    all_output_id_seqs = []
    all_other_output_key2seqs = defaultdict(list)
    for return_seq_idx in range(num_return_sequences):

        output_id_seqs = [[] for _ in range(batch_size)]
        other_output_key2seqs = defaultdict(
            lambda: [[] for _ in range(batch_size)])
        early_stoppings = [0 for _ in range(batch_size)]
        last_hidden_states = []
        other_output_key2seqs = defaultdict(
            lambda: [[] for _ in range(batch_size)])

        step_input_ids = torch.clone(
            input_ids[:, -1:]) if input_ids is not None else None
        step_input_embeds = None
        step_position_ids = torch.clone(position_ids[:, -1:])
        step_attention_mask = torch.clone(attention_mask)
        if num_return_sequences == return_seq_idx + 1:
            step_past_key_values = pre_outputs["past_key_values"]
        else:

            step_past_key_values = copy.deepcopy(
                pre_outputs["past_key_values"])
        if lad and record_internal_ode_steps:
            num_lad_steps = []
        for step in range(max_new_tokens):

            if lad:
                if step == 0:
                    cached_final_hidden_states = pre_outputs.cached_final_hidden_states
                else:
                    cached_final_hidden_states = None
                if ode_steps > 0:
                    flow_inference_output = base_model.nstep_inference(
                        input_ids=step_input_ids,
                        class_labels=class_labels,
                        position_ids=step_position_ids,
                        attention_mask=step_attention_mask,
                        past_key_values=step_past_key_values,
                        cached_final_hidden_states=cached_final_hidden_states,
                        ode_steps=ode_steps,
                        final_timestep=final_timestep,
                        precompute_non_flow_path=True,
                        only_last_token_prediction=True,
                        record_evaluated_steps=record_internal_ode_steps,
                    )
                    flow_embeds_trajectory = flow_inference_output.flow_trajectory
                    timestep_trajectory = flow_inference_output.timestep_trajectory
                    cached_final_hidden_states = flow_inference_output.cached_final_hidden_states
                    flow_embeds = flow_embeds_trajectory[-1].to(model.dtype)
                    timesteps = timestep_trajectory[-1]
                    if record_internal_ode_steps:
                        num_lad_steps.append(
                            len(flow_inference_output.tracked_steps))
                else:
                    cached_final_hidden_states = None
                    flow_embeds = None
                    timesteps = None
                step_outputs = base_model.forward(
                    input_ids=step_input_ids,
                    class_labels=class_labels,
                    timesteps=timesteps,
                    flow_representation_embeds=flow_embeds,
                    position_ids=step_position_ids,
                    attention_mask=step_attention_mask,
                    past_key_values=step_past_key_values,
                    cached_final_hidden_states=cached_final_hidden_states,
                    output_hidden_states=False,
                    use_cache=True,
                    return_dict=True,
                    disable_recursive_nstep_call=True,
                )
            else:
                step_outputs = base_model.forward(
                    input_ids=step_input_ids,
                    inputs_embeds=step_input_embeds,
                    position_ids=step_position_ids,
                    attention_mask=step_attention_mask,
                    past_key_values=step_past_key_values,
                    output_hidden_states=False,
                    use_cache=True
                )

            step_suppress_token_ids = suppress_token_ids
            if step < min_new_tokens:
                step_suppress_token_ids = step_suppress_token_ids + \
                    [tokenizer.eos_token_id]

            step_good_token_ids = []
            if valid_token_trie is not None:
                for batch_idx in range(batch_size):
                    step_good_token_ids.append(
                        valid_token_trie.get(output_id_seqs[batch_idx]))

            repetition_penalty_token_id2penalty = []
            if repetition_penalty > 1:
                for prefix, trie in zip(output_id_seqs, repetition_tries):
                    penalty_token_id2penalty = {}
                    for n in range(repetition_max_n_gram, 0, -1):
                        n_gram_prefix = prefix[-n:]
                        suffices = trie.get(n_gram_prefix)
                        penalty = repetition_penalty ** n
                        if len(suffices) > 0:
                            penalty_token_id2penalty = {
                                token_id: penalty for token_id in suffices}
                            break
                    repetition_penalty_token_id2penalty.append(
                        penalty_token_id2penalty)

            step_logits = step_outputs["logits"][:, -1, :]
            step_symbol = step_decode(
                model=base_model,
                step=step,
                step_input_ids=step_input_ids,
                step_position_ids=step_position_ids,
                step_attention_mask=step_attention_mask,
                step_outputs=step_outputs,
                last_hidden_states=last_hidden_states,
                logits=step_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                bad_token_ids=step_suppress_token_ids,
                good_token_ids=step_good_token_ids,
                penalty_token_id2penalty=repetition_penalty_token_id2penalty,
            )

            for batch_idx in range(batch_size):
                if early_stoppings[batch_idx] == 0:
                    output_id_seqs[batch_idx].append(
                        step_symbol[batch_idx].item())
                    for key in other_output_keys:
                        value = step_outputs[key][:, -1]
                        other_output_key2seqs[key][batch_idx].append(
                            value[batch_idx])
                if step_symbol[batch_idx] == tokenizer.eos_token_id:
                    early_stoppings[batch_idx] = 1
            if sum(early_stoppings) == batch_size:
                break

            if repetition_penalty > 1:
                for batch_idx in range(batch_size):
                    if early_stoppings[batch_idx] == 0:
                        for n_gram in range(1, repetition_max_n_gram+1):
                            repetition_tries[batch_idx].add(
                                output_id_seqs[batch_idx][-n_gram:])

            step_position_ids = step_position_ids + 1
            step_input_ids = step_symbol
            step_input_embeds = None
            next_step_attention_mask = torch.ones(
                batch_size, 1).long().to(model.device)
            step_attention_mask = torch.cat(
                [step_attention_mask, next_step_attention_mask], dim=-1)
            step_past_key_values = step_outputs["past_key_values"]

        all_output_id_seqs.extend(output_id_seqs)
        for key, value in other_output_key2seqs.items():
            all_other_output_key2seqs[key].extend(value)

    ret_data = {
        "output_id_seqs": all_output_id_seqs,
    }
    for key, value in all_other_output_key2seqs.items():
        ret_data[key] = value

    if lad and return_timestep_sequence:
        ret_data['timestep_sequence'] = timestep_trajectory
    if lad and record_internal_ode_steps:
        ret_data['lad_steps'] = num_lad_steps
    return ret_data


def step_decode(
    model: torch.nn.Module = None,
    step: int = 0,
    step_input_ids: torch.LongTensor = None,
    step_position_ids: torch.LongTensor = None,
    step_attention_mask: torch.BoolTensor = None,
    step_outputs: Dict = None,
    last_hidden_states: List[torch.FloatTensor] = None,
    logits: torch.FloatTensor = None,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    good_token_ids: List[List[int]] = [],
    bad_token_ids: List[int] = [],
    penalty_token_id2penalty: Dict = {}
):

    logits = temperature_processor(logits, temperature)
    logits = good_token_ids_processor(logits, good_token_ids)
    logits = bad_token_ids_processor(logits, bad_token_ids)
    logits = penalty_processor(logits, penalty_token_id2penalty)

    if not do_sample:
        symbol = logits.topk(1)[1]
    else:
        logits = top_k_processor(logits, top_k)
        logits = top_p_processor(logits, top_p)
        probs = logits.softmax(dim=-1)
        symbol = torch.multinomial(probs, num_samples=1)

    return symbol


def temperature_processor(logits, temp):
    if temp == 0.0:
        temp = 1e-8
    assert temp > 0.0
    logits = logits/temp
    return logits


def good_token_ids_processor(logits, good_token_ids):
    if len(good_token_ids) > 0:

        assert type(
            good_token_ids[0]) is list, "good_token_ids should be a list of list"
        for batch_idx, item_good_token_ids in enumerate(good_token_ids):
            item_good_token_ids = torch.LongTensor(
                item_good_token_ids).to(logits.device)
            item_good_token_logits = logits[batch_idx].gather(
                0, item_good_token_ids)
            logits[batch_idx] = 0.0
            logits[batch_idx].scatter_(
                0, item_good_token_ids, item_good_token_logits)
    return logits


def bad_token_ids_processor(logits, bad_token_ids):
    if len(bad_token_ids) > 0:

        if type(bad_token_ids[0]) is list:
            for batch_idx, item_bad_token_ids in enumerate(bad_token_ids):
                item_bad_token_ids = torch.LongTensor(
                    item_bad_token_ids).to(logits.device)
                logits[batch_idx, item_bad_token_ids] = -float("inf")

        else:
            bad_token_ids = torch.LongTensor(bad_token_ids).to(logits.device)
            logits[:, bad_token_ids] = -float("inf")
    return logits


def penalty_processor(logits, penalty_token_id2penalty):
    batch_size = logits.size(0)
    if len(penalty_token_id2penalty) > 0:
        assert len(penalty_token_id2penalty) == batch_size
        for batch_idx in range(batch_size):
            for token_id, penalty in penalty_token_id2penalty[batch_idx].items():
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= penalty
                else:
                    logits[batch_idx, token_id] *= penalty
    return logits


def top_k_processor(logits, top_k):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = -float("inf")
    return logits


def top_p_processor(logits, top_p):
    batch_size = logits.size(0)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 0] = 0

        for batch_idx in range(batch_size):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float("inf")
    return logits
