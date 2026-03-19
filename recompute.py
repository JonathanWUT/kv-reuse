import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from rouge import Rouge

def rouge_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

@torch.no_grad()
def generate(model, inputs_ids, max_new_tokens, kv_cache):
    for i in range(max_new_tokens):
        seen_lens = kv_cache.get_seq_length()
        if i == 0:
            # prefill stage
            position_ids = torch.arange(seen_lens, seen_lens + inputs_ids.size(1), device=inputs_ids.device).unsqueeze(
                0
            )
            out = model(
                inputs_ids,
                past_key_values=kv_cache,
                position_ids=position_ids,
                use_cache=True,
            )
        else:
            # decoding stage
            position_ids = torch.tensor([[seen_lens - 1]], device=inputs_ids.device)

            out = model(
                inputs_ids[:, -1:],
                past_key_values=kv_cache,
                position_ids=position_ids,
                use_cache=True,
            )

        logits = out.logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        inputs_ids = torch.cat([inputs_ids, next_token], dim=-1)

    return inputs_ids

def chunk_text_by_paragraphs(text: str):
    chunks = text.split("\n\n")
    return chunks


def cache_concat(cache_list):
    if len(cache_list) == 1:
        return cache_list[0]

    dst = cache_list[0]
    # layers[i].keys
    #   shape = [bsz, num_head, seqlen, headdim]
    for cache in cache_list[1:]:
        for layer_idx in range(len(dst.layers)):
            dst.layers[layer_idx].keys = torch.cat([
                dst.layers[layer_idx].keys,
                cache.layers[layer_idx].keys,
            ], dim=-2)

            dst.layers[layer_idx].values = torch.cat([
                dst.layers[layer_idx].values,
                cache.layers[layer_idx].values,
            ], dim=-2)
        del cache
    return dst

def tensor_split(tensor, idx, dim):
    mask = torch.zeros([tensor.shape[dim]], dtype=torch.bool)
    mask[idx] = True
    copy_slice = [slice(None)] * tensor.dim()

    copy_slice[dim] = mask
    selected_data = tensor[tuple(copy_slice)]
    copy_slice[dim] = ~mask
    unselected_data = tensor[tuple(copy_slice)]
    return selected_data, unselected_data

def base_forward_hook(module, input, output):
    # print(module)
    # print(input)
    # print(output)
    # nothing
    return output

def simple_concat_forward_hook(module, input, output):
    # no recompute
    return output

def random_recompute_forward_hook(module, input, output):
    # handle recompute
    return output

def monkeypatch(model, cache, hook):
    for name, module in model.named_modules():
        # print(name)
        if name == "lm_head":
            module.register_forward_hook(hook)
            module.cache = cache

    return model

def precompute(model, tokenizer, context):
    precomputed_cache = []
    history_input_ids = []
    for chunk in chunk_text_by_paragraphs(context):
        cache = DynamicCache()
        inputs_ids = tokenizer.encode(chunk, return_tensors="pt").to(model.device)
        history_input_ids.append(inputs_ids)
        generate(model, inputs_ids, max_new_tokens=1, kv_cache=cache)
        precomputed_cache.append(cache)

    return precomputed_cache, history_input_ids




def base_pipeline(model, tokenizer, context, question = ""):
    max_new_tokens = 64
    prompt = context + question
    inputs_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    cache = DynamicCache()
    model = monkeypatch(model, cache, base_forward_hook)
    outputs = generate(model, inputs_ids, max_new_tokens=max_new_tokens, kv_cache=cache)
    return outputs

def simple_concat_pipeline(model, tokenizer, context, question = ""):
    precomputed_cache, _ = precompute(model, tokenizer, context)
    cache = cache_concat(precomputed_cache)

    max_new_tokens = 64
    inputs_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)

    model = monkeypatch(model, cache, simple_concat_forward_hook)
    outputs = generate(model, inputs_ids, max_new_tokens=max_new_tokens, kv_cache=cache)
    return outputs

def random_recompute_pipeline(model, tokenizer, context, question = ""):
    precomputed_cache, history_input_ids = precompute(model, tokenizer, context)

    cache = cache_concat(precomputed_cache)
    full_input_ids = torch.cat(history_input_ids, dim=1)
    origin_len = cache.layers[0].keys.shape[-2]

    model = monkeypatch(model, cache, simple_concat_forward_hook)

    # ############## recompute stage start ############
    # 1. random select and remove in cache
    selected_idx = range(0, full_input_ids.shape[1], 4)
    ids_to_recompute, _ = tensor_split(full_input_ids, selected_idx, 1)

    for layer_idx in range(len(cache.layers)):
        _, cache.layers[layer_idx].keys = tensor_split(cache.layers[layer_idx].keys, selected_idx, -2)
        _, cache.layers[layer_idx].values = tensor_split(cache.layers[layer_idx].values, selected_idx, -2)


    # 2. recompute selected kv cache with correct position id and mask
    #   ignore here for now
    # NOTE: 注意, 这里没有考虑正确的position id, 也没有考虑causal mask
    generate(model, ids_to_recompute, max_new_tokens=1, kv_cache=cache)
    # ############## recompute stage end ############

    max_new_tokens = 64
    inputs_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)

    recompute_len = cache.layers[0].keys.shape[-2]

    assert origin_len == recompute_len

    outputs = generate(model, inputs_ids, max_new_tokens=max_new_tokens, kv_cache=cache)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="/home/ring/Documents/workspace/modules/tinyllama-110M", help="Path to the model or model name"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="./gov_report.jsonl", help="Path to the input JSONL file"
    )
    args = parser.parse_args()

    data = []
    with open(args.input, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # context = data[0]["context"]
    # question = "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    # prompt = context + " " + question
    # answer = data[0]["answer"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    context = "Once upon a time. \n\n Once upon a time. "
    question = "There"
    ref_outputs = base_pipeline(model, tokenizer, context, question)
    ref_generated_text = tokenizer.decode(ref_outputs[0])
    print(ref_generated_text)

    cc_outputs = simple_concat_pipeline(model, tokenizer, context, question)
    cc_generated_text = tokenizer.decode(cc_outputs[0])
    print(cc_generated_text)
    print(rouge_score(cc_generated_text, ref_generated_text))

    r_outputs = random_recompute_pipeline(model, tokenizer, context, question)
    r_generated_text = tokenizer.decode(r_outputs[0])
    print(r_generated_text)
    print(rouge_score(r_generated_text, ref_generated_text))

if __name__ == "__main__":
    main()

