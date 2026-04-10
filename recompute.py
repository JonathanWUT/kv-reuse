import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
import re
import string
import time
from types import MethodType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward
from rouge import Rouge


# 这里使用手工拼接的 Qwen 风格 chat 模板，避免不同 tokenizer 版本带来不一致。
QWEN3_USER_HEADER = "<|im_start|>user\n"
QWEN3_ASSISTANT_PREFILL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# 下面这些常量用于模拟 LMCache 风格的 chunk、blend 与重算策略。
LMCACHE_CHUNK_SIZE = 256
LMCACHE_BLEND_RECOMPUTE_RATIO = 0.15
# LMCache 原始实现只按 ratio 取 top-k，没有额外的最小 token 下限。
LMCACHE_BLEND_MIN_TOKENS = 0
LMCACHE_BLEND_BOUNDARY_WINDOW = 16
LMCACHE_BLEND_CHECK_LAYER = 1
LMCACHE_BLEND_RUN_MERGE_GAP = 0
LMCACHE_PARAGRAPH_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class DecodeConfig:
    # 统一管理不同数据集的解码配置，避免在 pipeline 中散落硬编码参数。
    max_new_tokens: int
    do_sample: bool
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 1234


@dataclass(frozen=True)
class ChunkPlan:
    # 记录每个 chunk 在完整输入中的范围，以及对应的前缀哈希。
    start_idx: int
    end_idx: int
    prefix_hash: str


@dataclass(frozen=True)
class PipelineMetrics:
    # ttft_ms 只统计在线首 token 延迟。
    # excluded_prep_ms 单独记录被排除在 TTFT 外的准备时间，便于日志核对。
    ttft_ms: float
    excluded_prep_ms: float = 0.0
    selection_ms: float = 0.0
    recompute_ms: float = 0.0
    selected_tokens: int = 0
    selected_runs: int = 0


class _StopKeyCapture(RuntimeError):
    # 用于在抓到目标层 K 之后提前中断整段前向，避免无意义地继续跑后续层。
    pass


def get_model_context_window(model, tokenizer):
    # 取模型配置和 tokenizer 配置中更可靠的较小值，作为可用上下文窗口。
    cfg_max = getattr(model.config, "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)

    candidates = []
    if isinstance(cfg_max, int) and cfg_max > 0:
        candidates.append(cfg_max)
    if isinstance(tok_max, int) and 0 < tok_max < 10**6:
        candidates.append(tok_max)

    if not candidates:
        raise ValueError("无法从模型配置或 tokenizer 中获取有效的上下文窗口大小")

    return min(candidates)


def encode_ids(tokenizer, text, device, max_length=None):
    # 统一封装编码逻辑，确保所有调用都不自动加特殊 token。
    kwargs = {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    return tokenizer.encode(text, **kwargs).to(device)


def split_token_ids(token_ids, chunk_size):
    # 按固定 token 数分块，供分段 prefill 和 chunk 预计算复用。
    if len(token_ids) == 0:
        return []
    return [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def hash_token_chunk(token_ids, prefix_hash=""):
    # 使用前缀链式哈希模拟 LMCache 中“当前 chunk 依赖前缀”的索引方式。
    payload = ",".join(str(token_id) for token_id in token_ids)
    digest = hashlib.sha256()
    digest.update(prefix_hash.encode("utf-8"))
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def build_chunk_plans(token_ids, chunk_size, save_unfull_chunk=True, skip_prefix_tokens=0):
    # 基于 token 序列生成 chunk 计划，后续预计算和重算都依赖这里的切分结果。
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if skip_prefix_tokens % chunk_size != 0:
        raise ValueError("skip_prefix_tokens 必须是 chunk_size 的整数倍")

    end = len(token_ids) if save_unfull_chunk else (len(token_ids) - len(token_ids) % chunk_size)
    chunk_plans = []
    prefix_hash = ""
    for start_idx in range(0, end, chunk_size):
        end_idx = min(start_idx + chunk_size, len(token_ids))
        chunk_ids = token_ids[start_idx:end_idx]
        prefix_hash = hash_token_chunk(chunk_ids, prefix_hash)
        if start_idx < skip_prefix_tokens:
            continue
        chunk_plans.append(
            ChunkPlan(
                start_idx=start_idx,
                end_idx=end_idx,
                prefix_hash=prefix_hash,
            )
        )
    return chunk_plans


def find_separator_spans(token_ids, separator_ids):
    # 在 token 序列中查找段落分隔符的位置，用于启发式重算回退逻辑。
    if not separator_ids or len(token_ids) < len(separator_ids):
        return []

    separator_spans = []
    index = 0
    separator_len = len(separator_ids)
    while index <= len(token_ids) - separator_len:
        if token_ids[index : index + separator_len] == separator_ids:
            separator_spans.append((index, index + separator_len))
            index += separator_len
        else:
            index += 1
    return separator_spans


def select_heuristic_recompute_positions(
    full_input_ids,
    chunk_size,
    separator_ids,
    recompute_ratio=LMCACHE_BLEND_RECOMPUTE_RATIO,
    min_tokens=LMCACHE_BLEND_MIN_TOKENS,
    boundary_window=LMCACHE_BLEND_BOUNDARY_WINDOW,
):
    # 这是 K 偏差选点失败时的保底策略：优先选择 chunk 边界和段落分隔符附近的 token。
    total_len = full_input_ids.shape[1]
    if total_len == 0:
        return []

    target_tokens = max(1, int(total_len * recompute_ratio))
    if total_len >= min_tokens:
        target_tokens = max(target_tokens, min_tokens)
    target_tokens = min(total_len, target_tokens)

    token_list = full_input_ids[0].tolist()
    candidate_positions = []

    for boundary in range(chunk_size, total_len, chunk_size):
        start_idx = max(0, boundary - boundary_window)
        end_idx = min(total_len, boundary + boundary_window)
        candidate_positions.extend(range(start_idx, end_idx))

    for start_idx, end_idx in find_separator_spans(token_list, separator_ids):
        span_start = max(0, start_idx - boundary_window)
        span_end = min(total_len, end_idx + boundary_window)
        candidate_positions.extend(range(span_start, span_end))

    selected_positions = []
    seen_positions = set()
    for position in candidate_positions:
        if position in seen_positions:
            continue
        seen_positions.add(position)
        selected_positions.append(position)
        if len(selected_positions) >= target_tokens:
            return selected_positions

    stride = max(1, total_len // target_tokens)
    for position in range(0, total_len, stride):
        if position in seen_positions:
            continue
        seen_positions.add(position)
        selected_positions.append(position)
        if len(selected_positions) >= target_tokens:
            break

    return selected_positions


@torch.no_grad()
def collect_layer_key_states(model, input_ids, layer_idx, position_ids):
    # 通过临时包裹目标层 attention.forward，抓取某一层当前输入对应的 K。
    # 这里不改模型权重，只在一次前向期间拦截中间张量。
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("当前模型结构不支持按层提取 K")
    if layer_idx < 0 or layer_idx >= len(model.model.layers):
        raise ValueError("check_layer 超出模型层数")

    attention_layer = model.model.layers[layer_idx].self_attn
    captured = {}
    original_forward = attention_layer.forward

    def wrapped_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        # 这里基本复现 Qwen2Attention.forward 的关键路径，只是在 RoPE 后保存 key_states。
        # 一旦拿到目标层的 K，就立刻中断后续层的前向，降低选点阶段的额外开销。
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        captured["keys"] = key_states.detach()
        raise _StopKeyCapture()

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attention_layer.forward = MethodType(wrapped_forward, attention_layer)
    try:
        model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )
    except _StopKeyCapture:
        pass
    finally:
        attention_layer.forward = original_forward

    if "keys" not in captured:
        raise RuntimeError("未捕获到目标层的 K")

    return captured["keys"]


@torch.no_grad()
def select_lmcache_recompute_positions(
    model,
    cache,
    full_input_ids,
    chunk_size,
    separator_ids,
    recompute_ratio=LMCACHE_BLEND_RECOMPUTE_RATIO,
    min_tokens=LMCACHE_BLEND_MIN_TOKENS,
    boundary_window=LMCACHE_BLEND_BOUNDARY_WINDOW,
    check_layer=LMCACHE_BLEND_CHECK_LAYER,
):
    # 优先按检查层的 K 偏差做 top-k 选点，这更接近 LMCache 的正式 blend 逻辑。
    # 如果模型结构不兼容或抓取失败，再回退到启发式策略。
    total_len = full_input_ids.shape[1]
    if total_len == 0:
        return []

    target_tokens = max(1, int(total_len * recompute_ratio))
    if total_len >= min_tokens:
        target_tokens = max(target_tokens, min_tokens)
    target_tokens = min(total_len, target_tokens)

    if len(cache.layers) == 0:
        return []

    try:
        position_ids = torch.arange(
            total_len,
            device=full_input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        fresh_keys = collect_layer_key_states(
            model=model,
            input_ids=full_input_ids,
            layer_idx=check_layer,
            position_ids=position_ids,
        )
        cached_keys = cache.layers[check_layer].keys

        if fresh_keys.shape != cached_keys.shape:
            raise ValueError(
                f"K 形状不一致: fresh={tuple(fresh_keys.shape)}, cached={tuple(cached_keys.shape)}"
            )

        diff_k = (fresh_keys.to(torch.float32) - cached_keys.to(torch.float32)).pow(2).sum(dim=(0, 1, 3))
        top_indices = torch.topk(diff_k, k=target_tokens).indices
        top_indices, _ = torch.sort(top_indices)
        return top_indices.tolist()
    except Exception as exc:
        print(f"[warn] K 偏差选点失败，回退到启发式选点: {exc}")
        return select_heuristic_recompute_positions(
            full_input_ids=full_input_ids,
            chunk_size=chunk_size,
            separator_ids=separator_ids,
            recompute_ratio=recompute_ratio,
            min_tokens=min_tokens,
            boundary_window=boundary_window,
        )


@torch.no_grad()
def select_lmcache_recompute_token_head_pairs(
    model,
    cache,
    full_input_ids,
    chunk_size,
    separator_ids,
    recompute_ratio=LMCACHE_BLEND_RECOMPUTE_RATIO,
    min_tokens=LMCACHE_BLEND_MIN_TOKENS,
    boundary_window=LMCACHE_BLEND_BOUNDARY_WINDOW,
    check_layer=LMCACHE_BLEND_CHECK_LAYER,
):
    # 新加一条线：
    # 不再先把每个 token 上所有 head 的偏差聚合后再取 top-k，
    # 而是在所有 (head, token) 对上统一做 top-k。
    # 返回:
    #   {token_pos: [head_idx1, head_idx2, ...], ...}
    total_len = full_input_ids.shape[1]
    if total_len == 0:
        return {}

    target_tokens = max(1, int(total_len * recompute_ratio))
    if total_len >= min_tokens:
        target_tokens = max(target_tokens, min_tokens)

    if len(cache.layers) == 0:
        return {}

    try:
        position_ids = torch.arange(
            total_len,
            device=full_input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        fresh_keys = collect_layer_key_states(
            model=model,
            input_ids=full_input_ids,
            layer_idx=check_layer,
            position_ids=position_ids,
        )
        cached_keys = cache.layers[check_layer].keys

        if fresh_keys.shape != cached_keys.shape:
            raise ValueError(
                f"K 形状不一致: fresh={tuple(fresh_keys.shape)}, cached={tuple(cached_keys.shape)}"
            )

        # [bsz, num_heads, seq, head_dim] -> [num_heads, seq]
        diff_k_by_head = (
            (fresh_keys.to(torch.float32) - cached_keys.to(torch.float32))
            .pow(2)
            .sum(dim=(0, 3))
        )

        num_heads, seq_len = diff_k_by_head.shape
        flat_scores = diff_k_by_head.reshape(-1)

        # 在所有 (head, token) 对上统一取 top-k
        target_pairs = min(flat_scores.numel(), target_tokens)
        top_flat_indices = torch.topk(flat_scores, k=target_pairs).indices

        selected_token_heads = {}
        for flat_idx in top_flat_indices.tolist():
            head_idx = flat_idx // seq_len
            token_idx = flat_idx % seq_len
            selected_token_heads.setdefault(token_idx, []).append(head_idx)

        for token_idx in selected_token_heads:
            selected_token_heads[token_idx] = sorted(set(selected_token_heads[token_idx]))

        return dict(sorted(selected_token_heads.items()))
    except Exception as exc:
        print(f"[warn] token-head K 偏差选点失败，回退到启发式选点: {exc}")
        fallback_positions = select_heuristic_recompute_positions(
            full_input_ids=full_input_ids,
            chunk_size=chunk_size,
            separator_ids=separator_ids,
            recompute_ratio=recompute_ratio,
            min_tokens=min_tokens,
            boundary_window=boundary_window,
        )
        num_heads = cache.layers[check_layer].keys.shape[1]
        return {pos: list(range(num_heads)) for pos in fallback_positions}

def rouge_score(prediction, ground_truth):
    # gov_report 使用 ROUGE-L F1 作为主指标。
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def get_decode_config(dataset):
    # 针对不同数据集使用不同解码策略：gov_report 偏摘要，2wikimqa 偏短答案。
    if dataset == "gov_report":
        return DecodeConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            seed=1234,
        )

    return DecodeConfig(
        max_new_tokens=64,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=1234,
    )


def build_model_load_kwargs():
    # 根据当前可见 GPU 数自动构造加载参数。
    # 单卡走 auto，多卡时给出显存预算，尽量避免模型全落在 0 号卡上。
    load_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }

    if not torch.cuda.is_available():
        load_kwargs["device_map"] = "cpu"
        return load_kwargs

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 1:
        load_kwargs["device_map"] = "auto"
        return load_kwargs

    max_memory = {}
    for device_idx in range(visible_gpu_count):
        total_gib = torch.cuda.get_device_properties(device_idx).total_memory // (1024**3)
        usable_gib = max(16, int(total_gib * 0.82))
        max_memory[device_idx] = f"{usable_gib}GiB"

    load_kwargs["device_map"] = "balanced_low_0"
    load_kwargs["max_memory"] = max_memory
    return load_kwargs


def set_generation_seed(seed):
    # 固定随机种子，保证相同配置下输出尽量可复现。
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda_memory():
    # 显式释放 Python 引用和 CUDA 缓存，降低长跑评测时的显存累积风险。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def sync_cuda_if_needed():
    # 计时前后同步 CUDA，避免异步执行导致 TTFT 失真。
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prefill_cache(model, input_ids, kv_cache, position_start=None, position_ids=None):
    # 仅做 prefill，不采样新 token；位置既可以自动推断，也可以显式指定。
    if input_ids.numel() == 0 or input_ids.size(1) == 0:
        return None

    if position_ids is None:
        seen_lens = kv_cache.get_seq_length()
        start = seen_lens if position_start is None else position_start
        end = start + input_ids.size(1)
        position_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
    else:
        position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

    return model(
        input_ids,
        past_key_values=kv_cache,
        position_ids=position_ids,
        use_cache=True,
    )


def sample_next_token(next_token_logits, decode_config):
    # 统一处理 greedy / top-k / top-p 采样逻辑。
    if (not decode_config.do_sample) or decode_config.temperature <= 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    logits = next_token_logits / decode_config.temperature

    if decode_config.top_k > 0:
        top_k = min(decode_config.top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0 < decode_config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > decode_config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(model, inputs_ids, decode_config, kv_cache, stop_token_ids=None, prefill_chunk_size=None):
    # 完整生成流程：先 prefill，再逐 token 解码。
    # 这个函数不负责 TTFT 统计，供普通生成路径复用。
    stop_token_ids = set(stop_token_ids or [])
    if inputs_ids.numel() == 0 or inputs_ids.size(1) == 0:
        raise ValueError("generate 需要至少一个输入 token")

    generated_ids = inputs_ids

    out = None
    if prefill_chunk_size is not None and generated_ids.size(1) > prefill_chunk_size:
        for chunk_ids in split_token_ids(generated_ids[0].tolist(), prefill_chunk_size):
            chunk_tensor = torch.tensor([chunk_ids], device=generated_ids.device, dtype=generated_ids.dtype)
            out = prefill_cache(model, chunk_tensor, kv_cache)
    else:
        out = prefill_cache(model, generated_ids, kv_cache)

    next_token_logits = out.logits[:, -1, :]

    for step in range(decode_config.max_new_tokens):
        next_token = sample_next_token(next_token_logits, decode_config)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if stop_token_ids and next_token.item() in stop_token_ids:
            break

        if step == decode_config.max_new_tokens - 1:
            break

        seen_lens = kv_cache.get_seq_length()
        position_ids = torch.tensor([[seen_lens]], device=generated_ids.device)

        out = model(
            next_token,
            past_key_values=kv_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        next_token_logits = out.logits[:, -1, :]

    return generated_ids


@torch.no_grad()
def generate_with_ttft(
    model,
    inputs_ids,
    decode_config,
    kv_cache,
    stop_token_ids=None,
    prefill_chunk_size=None,
    ttft_start_time=None,
):
    # 和 generate 基本一致，但会单独统计首 token 延迟。
    # 计时起点由调用方决定，从而可以把预计算阶段排除在 TTFT 外。
    stop_token_ids = set(stop_token_ids or [])
    if inputs_ids.numel() == 0 or inputs_ids.size(1) == 0:
        raise ValueError("generate 需要至少一个输入 token")

    generated_ids = inputs_ids

    if ttft_start_time is None:
        sync_cuda_if_needed()
        ttft_start = time.perf_counter()
    else:
        ttft_start = ttft_start_time

    out = None
    if prefill_chunk_size is not None and generated_ids.size(1) > prefill_chunk_size:
        for chunk_ids in split_token_ids(generated_ids[0].tolist(), prefill_chunk_size):
            chunk_tensor = torch.tensor([chunk_ids], device=generated_ids.device, dtype=generated_ids.dtype)
            out = prefill_cache(model, chunk_tensor, kv_cache)
    else:
        out = prefill_cache(model, generated_ids, kv_cache)

    next_token_logits = out.logits[:, -1, :]
    next_token = sample_next_token(next_token_logits, decode_config)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    sync_cuda_if_needed()
    ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

    if stop_token_ids and next_token.item() in stop_token_ids:
        return generated_ids, ttft_ms

    for step in range(1, decode_config.max_new_tokens):
        seen_lens = kv_cache.get_seq_length()
        position_ids = torch.tensor([[seen_lens]], device=generated_ids.device)

        out = model(
            next_token,
            past_key_values=kv_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        next_token_logits = out.logits[:, -1, :]
        next_token = sample_next_token(next_token_logits, decode_config)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    return generated_ids, ttft_ms

def chunk_text_by_paragraphs(text: str):
    # 保留一个按段落切分的简单辅助函数，便于调试或后续实验。
    chunks = text.split("\n\n")
    return chunks


def first_n_words(text: str, n: int = 40) -> str:
    # 日志里只打印输入前若干词，避免长上下文直接刷屏。
    words = text.split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + " ..."


def normalize_answer_text(text):
    # 对答案做归一化，供 2wikimqa 的 EM/F1 计算使用。
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def exact_match_any(prediction, answers):
    # 只要命中任一标准答案，就视为 EM 命中。
    norm_pred = normalize_answer_text(prediction)
    for ans in answers:
        if norm_pred == normalize_answer_text(ans):
            return 1.0
    return 0.0


def contains_match_any(prediction, answers):
    # 某些短答案任务允许“预测中包含标准答案”视作命中。
    norm_pred = normalize_answer_text(prediction)
    for ans in answers:
        norm_ans = normalize_answer_text(ans)
        if norm_ans and norm_ans in norm_pred:
            return 1.0
    return 0.0


def token_f1_score(prediction, ground_truth):
    # 按 token overlap 计算 F1，用于 2wikimqa。
    pred_tokens = normalize_answer_text(prediction).split()
    gt_tokens = normalize_answer_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in gt_tokens:
        count = common.get(token, 0)
        if count > 0:
            overlap += 1
            common[token] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score_any(prediction, answers):
    # 多答案场景下取最佳 F1。
    if not answers:
        return 0.0
    return max(token_f1_score(prediction, ans) for ans in answers)


def build_2wikimqa_prompt(context, question):
    # 2wikimqa 使用短答案模板，要求模型只输出答案本身。
    instruction = "Answer the question based on the given passages. Output only the short answer."
    formatted_context = (
        f"{QWEN3_USER_HEADER}"
        f"{instruction}\n"
        "The following are given passages.\n"
        f"{context}"
    )
    formatted_question = (
        f"\n\nQuestion: {question}\n"
        "Answer:"
        f"{QWEN3_ASSISTANT_PREFILL}"
    )
    return formatted_context, formatted_question


def build_gov_report_prompt(context):
    # gov_report 使用摘要模板，要求模型输出单页事实性摘要。
    formatted_context = (
        f"{QWEN3_USER_HEADER}"
        "You are given a report by a government agency.\n"
        "Read the report carefully.\n"
        "Report:\n"
        f"{context}"
    )
    formatted_question = (
        "\n\nWrite a factual one-page summary of the report. "
        "Cover the report's purpose, methods, key findings, and recommendations when available. "
        "Output only the summary in plain prose."
        f"{QWEN3_ASSISTANT_PREFILL}"
    )
    return formatted_context, formatted_question


def cache_concat(cache_list):
    # 将多个分块 cache 沿序列维拼接，形成一段连续上下文的 KV。
    if len(cache_list) == 0:
        raise ValueError("cache_list 不能为空")
    if len(cache_list) == 1:
        return cache_list[0]

    dst = cache_list[0]
    # layers[i].keys
    #   shape = [bsz, num_head, seqlen, headdim]
    num_layers = len(dst.layers)
    for layer_idx in range(num_layers):
        dst.layers[layer_idx].keys = torch.cat(
            [cache.layers[layer_idx].keys for cache in cache_list], dim=-2
        )
        dst.layers[layer_idx].values = torch.cat(
            [cache.layers[layer_idx].values for cache in cache_list], dim=-2
        )
    return dst


def cache_append(dst_cache, src_cache):
    # 以增量方式拼接 cache，避免先存一大堆分块 cache 再统一拼接造成峰值显存过高。
    if dst_cache is None:
        return src_cache
    return cache_concat([dst_cache, src_cache])


def build_recompute_runs(selected_positions, merge_gap=LMCACHE_BLEND_RUN_MERGE_GAP):
    # 将离得较近的重算位置合并成连续区间，减少大量零碎小前向带来的延迟。
    if not selected_positions:
        return []

    selected_runs = []
    run_start = selected_positions[0]
    run_prev = run_start
    for pos in selected_positions[1:]:
        if pos <= run_prev + 1 + merge_gap:
            run_prev = pos
        else:
            selected_runs.append((run_start, run_prev))
            run_start = pos
            run_prev = pos
    selected_runs.append((run_start, run_prev))
    return selected_runs


@torch.no_grad()
def recompute_selected_kv_with_position_and_mask(model, cache, full_input_ids, selected_positions):
    # 只重算被选中的位置，并把新的 KV 原位覆盖回总 cache。
    # 这样可以模拟 LMCache blend 中“局部重算”而不是整段重算的效果。
    if len(selected_positions) == 0:
        return cache, 0

    total_len = full_input_ids.shape[1]
    selected_positions = sorted(
        set(int(p) for p in selected_positions if 0 <= int(p) < total_len)
    )
    if len(selected_positions) == 0:
        return cache, 0

    cache_len = cache.layers[0].keys.shape[-2]
    if cache_len != total_len:
        raise ValueError("cache 长度与 full_input_ids 长度不一致")

    device = full_input_ids.device

    # 将离散 token 合并成较少的连续区间，减少小粒度重算带来的巨大调度开销。
    selected_runs = build_recompute_runs(selected_positions)

    for run_start, run_end in selected_runs:
        run_positions = torch.arange(run_start, run_end + 1, device=device, dtype=torch.long)
        tokens = full_input_ids[:, run_positions]

        # 构造到 run_start 的前缀 cache 视图，不做删除和重排；
        # 仅用该前缀重算当前 run，然后将新增 KV 原位覆盖到原 cache 的目标位置。
        prefix_cache = DynamicCache()
        if run_start > 0:
            for layer_idx in range(len(cache.layers)):
                prefix_cache.update(
                    cache.layers[layer_idx].keys[:, :, :run_start, :],
                    cache.layers[layer_idx].values[:, :, :run_start, :],
                    layer_idx,
                )

        position_ids = run_positions.unsqueeze(0)

        model(
            tokens,
            past_key_values=prefix_cache,
            position_ids=position_ids,
            use_cache=True,
        )

        # run KV 位于 prefix_cache 末尾，直接覆盖到原 cache 对应位置。
        run_len = run_positions.numel()
        for layer_idx in range(len(cache.layers)):
            run_keys = prefix_cache.layers[layer_idx].keys[:, :, -run_len:, :]
            run_values = prefix_cache.layers[layer_idx].values[:, :, -run_len:, :]
            layer_device = cache.layers[layer_idx].keys.device
            run_index = run_positions.to(device=layer_device, dtype=torch.long)
            cache.layers[layer_idx].keys.index_copy_(-2, run_index, run_keys)
            cache.layers[layer_idx].values.index_copy_(-2, run_index, run_values)

        return cache, len(selected_runs)


@torch.no_grad()
def recompute_selected_kv_heads_with_position_and_mask(model, cache, full_input_ids, selected_token_heads):
    # 新加一条线：
    # 只重算被选中的 token；写回时，只覆盖这些 token 上被选中的 heads。
    if not selected_token_heads:
        return cache, 0

    total_len = full_input_ids.shape[1]
    selected_token_heads = {
        int(pos): sorted(set(int(h) for h in heads))
        for pos, heads in selected_token_heads.items()
        if 0 <= int(pos) < total_len and len(heads) > 0
    }
    if not selected_token_heads:
        return cache, 0

    cache_len = cache.layers[0].keys.shape[-2]
    if cache_len != total_len:
        raise ValueError("cache 长度与 full_input_ids 长度不一致")

    device = full_input_ids.device
    selected_positions = sorted(selected_token_heads.keys())
    selected_runs = build_recompute_runs(selected_positions)

    for run_start, run_end in selected_runs:
        run_positions = torch.arange(run_start, run_end + 1, device=device, dtype=torch.long)
        tokens = full_input_ids[:, run_positions]

        prefix_cache = DynamicCache()
        if run_start > 0:
            for layer_idx in range(len(cache.layers)):
                prefix_cache.update(
                    cache.layers[layer_idx].keys[:, :, :run_start, :],
                    cache.layers[layer_idx].values[:, :, :run_start, :],
                    layer_idx,
                )

        position_ids = run_positions.unsqueeze(0)

        model(
            tokens,
            past_key_values=prefix_cache,
            position_ids=position_ids,
            use_cache=True,
        )

        run_len = run_positions.numel()
        for layer_idx in range(len(cache.layers)):
            run_keys = prefix_cache.layers[layer_idx].keys[:, :, -run_len:, :]
            run_values = prefix_cache.layers[layer_idx].values[:, :, -run_len:, :]

            layer_device = cache.layers[layer_idx].keys.device

            for abs_pos in range(run_start, run_end + 1):
                heads = selected_token_heads.get(abs_pos)
                if not heads:
                    continue

                local_pos = abs_pos - run_start
                head_index = torch.tensor(heads, device=layer_device, dtype=torch.long)

                cache.layers[layer_idx].keys[:, head_index, abs_pos:abs_pos + 1, :] = (
                    run_keys[:, head_index, local_pos:local_pos + 1, :]
                )
                cache.layers[layer_idx].values[:, head_index, abs_pos:abs_pos + 1, :] = (
                    run_values[:, head_index, local_pos:local_pos + 1, :]
                )

    return cache, len(selected_runs)

def tensor_split(tensor, idx, dim):
    # 调试辅助函数：按给定索引把 tensor 切成“选中”和“未选中”两部分。
    mask = torch.zeros([tensor.shape[dim]], dtype=torch.bool)
    mask[idx] = True
    copy_slice = [slice(None)] * tensor.dim()

    copy_slice[dim] = mask
    selected_data = tensor[tuple(copy_slice)]
    copy_slice[dim] = ~mask
    unselected_data = tensor[tuple(copy_slice)]
    return selected_data, unselected_data

def base_forward_hook(module, input, output):
    # 目前这些 hook 是占位接口，保留给后续更细粒度的实验使用。
    # print(module)
    # print(input)
    # print(output)
    # nothing
    return output

def simple_concat_forward_hook(module, input, output):
    # simple_concat 路径目前不需要额外 hook 逻辑。
    # no recompute
    return output

def random_recompute_forward_hook(module, input, output):
    # blend / recompute 路径目前也未接入额外 hook 逻辑。
    # handle recompute
    return output

def monkeypatch(model, cache, hook):
    # 仅给 lm_head 注册 hook，主要用于早期实验；当前主流程并未依赖这里。
    for name, module in model.named_modules():
        # print(name)
        if name == "lm_head":
            module.register_forward_hook(hook)
            module.cache = cache

    return model

def precompute(model, tokenizer, context, context_budget_tokens, max_chunk_tokens):
    # 将长 context 分 chunk 预计算成 KV，并按全文绝对位置写入 cache。
    # 这样后续拼接后的 KV 位置语义才与完整上下文一致。
    if context_budget_tokens is not None:
        full_context_ids = tokenizer.encode(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=context_budget_tokens,
        )
    else:
        full_context_ids = tokenizer.encode(context, add_special_tokens=False)

    chunk_plans = build_chunk_plans(full_context_ids, max_chunk_tokens)
    merged_cache = None
    for chunk_plan in chunk_plans:
        cache = DynamicCache()
        chunk_ids = full_context_ids[chunk_plan.start_idx : chunk_plan.end_idx]
        inputs_ids = torch.tensor([chunk_ids], device=model.device, dtype=torch.long)
        chunk_position_ids = torch.arange(
            chunk_plan.start_idx,
            chunk_plan.end_idx,
            device=model.device,
            dtype=torch.long,
        ).unsqueeze(0)
        prefill_cache(
            model,
            inputs_ids,
            cache,
            position_ids=chunk_position_ids,
        )
        merged_cache = cache_append(merged_cache, cache)

    if merged_cache is None:
        merged_cache = DynamicCache()

    full_input_ids = torch.tensor([full_context_ids], device=model.device, dtype=torch.long)
    return merged_cache, full_input_ids, chunk_plans




def base_pipeline(model, tokenizer, context, question="", decode_config=None, stop_token_ids=None):
    # 基线方案：直接对 context + question 做完整前向与生成。
    decode_config = decode_config or get_decode_config("gov_report")
    model_ctx = get_model_context_window(model, tokenizer)

    # 先确保 question 保留在窗口内，再用剩余预算截断 context。
    # 否则长 context 会把 question 挤掉，模型就会退化成续写上下文尾部。
    max_question_tokens = max(1, model_ctx - decode_config.max_new_tokens)
    question_ids = encode_ids(tokenizer, question, model.device, max_length=max_question_tokens)
    question_tokens = question_ids.size(1)

    context_budget = max(1, model_ctx - decode_config.max_new_tokens - question_tokens)
    context_ids = encode_ids(tokenizer, context, model.device, max_length=context_budget)
    inputs_ids = torch.cat([context_ids, question_ids], dim=1)

    set_generation_seed(decode_config.seed)
    cache = DynamicCache()
    prefill_chunk_size = max(1, min(128, model_ctx - 1))
    sync_cuda_if_needed()
    online_ttft_start = time.perf_counter()
    outputs, ttft_ms = generate_with_ttft(
        model,
        inputs_ids,
        decode_config=decode_config,
        kv_cache=cache,
        stop_token_ids=stop_token_ids,
        prefill_chunk_size=prefill_chunk_size,
        ttft_start_time=online_ttft_start,
    )
    return outputs, inputs_ids.shape[1], PipelineMetrics(ttft_ms=ttft_ms)

def simple_concat_pipeline(model, tokenizer, context, question="", decode_config=None, stop_token_ids=None):
    # simple_concat：先把 context 预计算成 KV，再只对 question 做在线生成。
    # TTFT 只统计在线阶段，不包含 chunk/哈希/预计算时间。
    decode_config = decode_config or get_decode_config("gov_report")
    model_ctx = get_model_context_window(model, tokenizer)

    # 将 question 单独处理，而不是与 context 混合截断
    max_question_tokens = max(1, model_ctx - decode_config.max_new_tokens)
    question_ids = encode_ids(tokenizer, question, model.device, max_length=max_question_tokens)
    question_tokens = question_ids.size(1)

    # context 可以使用剩余的预算
    context_budget = max(1, model_ctx - decode_config.max_new_tokens - question_tokens)
    cache_chunk_size = max(1, min(LMCACHE_CHUNK_SIZE, context_budget))
    prep_start = time.perf_counter()
    cache, _, _ = precompute(
        model,
        tokenizer,
        context,
        context_budget_tokens=context_budget,
        max_chunk_tokens=cache_chunk_size,
    )
    sync_cuda_if_needed()
    excluded_prep_ms = (time.perf_counter() - prep_start) * 1000.0

    set_generation_seed(decode_config.seed)
    sync_cuda_if_needed()
    online_ttft_start = time.perf_counter()
    outputs, ttft_ms = generate_with_ttft(
        model,
        question_ids,
        decode_config=decode_config,
        kv_cache=cache,
        stop_token_ids=stop_token_ids,
        ttft_start_time=online_ttft_start,
    )
    return outputs, question_ids.shape[1], PipelineMetrics(ttft_ms=ttft_ms, excluded_prep_ms=excluded_prep_ms)


def random_recompute_pipeline(model, tokenizer, context, question="", decode_config=None, stop_token_ids=None):
    # blend_recompute：在预计算 KV 基础上，按 K 偏差选点并局部重算，再进入生成。
    # TTFT 同样排除预计算，但会保留在线重算时间。
    decode_config = decode_config or get_decode_config("gov_report")
    model_ctx = get_model_context_window(model, tokenizer)

    # 将 question 单独处理，而不是与 context 混合截断
    max_question_tokens = max(1, model_ctx - decode_config.max_new_tokens)
    question_ids = encode_ids(tokenizer, question, model.device, max_length=max_question_tokens)
    question_tokens = question_ids.size(1)

    context_budget = max(1, model_ctx - decode_config.max_new_tokens - question_tokens)
    cache_chunk_size = max(1, min(LMCACHE_CHUNK_SIZE, context_budget))

    prep_start = time.perf_counter()
    cache, full_input_ids, _ = precompute(
        model,
        tokenizer,
        context,
        context_budget_tokens=context_budget,
        max_chunk_tokens=cache_chunk_size,
    )
    sync_cuda_if_needed()
    excluded_prep_ms = (time.perf_counter() - prep_start) * 1000.0
    origin_len = cache.layers[0].keys.shape[-2]

    sync_cuda_if_needed()
    online_ttft_start = time.perf_counter()
    
    # ############## recompute stage start ############
    # 这里的在线阶段包含两部分：
    # 1）按检查层 K 偏差选出待重算位置；
    # 2）只对这些位置对应的连续区间做局部重算。
    separator_ids = tokenizer.encode(LMCACHE_PARAGRAPH_SEPARATOR, add_special_tokens=False)
    selection_start = time.perf_counter()
    selected_idx = select_lmcache_recompute_positions(
        model=model,
        cache=cache,
        full_input_ids=full_input_ids,
        chunk_size=cache_chunk_size,
        separator_ids=separator_ids,
    )
    sync_cuda_if_needed()
    selection_ms = (time.perf_counter() - selection_start) * 1000.0

    recompute_start = time.perf_counter()
    cache, selected_run_count = recompute_selected_kv_with_position_and_mask(
        model=model,
        cache=cache,
        full_input_ids=full_input_ids,
        selected_positions=selected_idx,
    )
    sync_cuda_if_needed()
    recompute_ms = (time.perf_counter() - recompute_start) * 1000.0
    # ############## recompute stage end ############

    recompute_len = cache.layers[0].keys.shape[-2]

    assert origin_len == recompute_len

    set_generation_seed(decode_config.seed)
    outputs, ttft_ms = generate_with_ttft(
        model,
        question_ids,
        decode_config=decode_config,
        kv_cache=cache,
        stop_token_ids=stop_token_ids,
        ttft_start_time=online_ttft_start,
    )
    return outputs, question_ids.shape[1], PipelineMetrics(
        ttft_ms=ttft_ms,
        excluded_prep_ms=excluded_prep_ms,
        selection_ms=selection_ms,
        recompute_ms=recompute_ms,
        selected_tokens=len(selected_idx),
        selected_runs=selected_run_count,
    )



def random_recompute_headwise_pipeline(model, tokenizer, context, question="", decode_config=None, stop_token_ids=None):
    # 新加 1 条线：
    # 按 (head, token) 对统一选 top-k；
    # 每个 token 只重算被选中的 heads。
    decode_config = decode_config or get_decode_config("gov_report")
    model_ctx = get_model_context_window(model, tokenizer)

    max_question_tokens = max(1, model_ctx - decode_config.max_new_tokens)
    question_ids = encode_ids(tokenizer, question, model.device, max_length=max_question_tokens)
    question_tokens = question_ids.size(1)

    context_budget = max(1, model_ctx - decode_config.max_new_tokens - question_tokens)
    cache_chunk_size = max(1, min(LMCACHE_CHUNK_SIZE, context_budget))

    prep_start = time.perf_counter()
    cache, full_input_ids, _ = precompute(
        model,
        tokenizer,
        context,
        context_budget_tokens=context_budget,
        max_chunk_tokens=cache_chunk_size,
    )
    sync_cuda_if_needed()
    excluded_prep_ms = (time.perf_counter() - prep_start) * 1000.0
    origin_len = cache.layers[0].keys.shape[-2]

    sync_cuda_if_needed()
    online_ttft_start = time.perf_counter()

    separator_ids = tokenizer.encode(LMCACHE_PARAGRAPH_SEPARATOR, add_special_tokens=False)

    selection_start = time.perf_counter()
    selected_token_heads = select_lmcache_recompute_token_head_pairs(
        model=model,
        cache=cache,
        full_input_ids=full_input_ids,
        chunk_size=cache_chunk_size,
        separator_ids=separator_ids,
    )
    sync_cuda_if_needed()
    selection_ms = (time.perf_counter() - selection_start) * 1000.0

    recompute_start = time.perf_counter()
    cache, selected_run_count = recompute_selected_kv_heads_with_position_and_mask(
        model=model,
        cache=cache,
        full_input_ids=full_input_ids,
        selected_token_heads=selected_token_heads,
    )
    sync_cuda_if_needed()
    recompute_ms = (time.perf_counter() - recompute_start) * 1000.0

    recompute_len = cache.layers[0].keys.shape[-2]
    assert origin_len == recompute_len

    set_generation_seed(decode_config.seed)
    outputs, ttft_ms = generate_with_ttft(
        model,
        question_ids,
        decode_config=decode_config,
        kv_cache=cache,
        stop_token_ids=stop_token_ids,
        ttft_start_time=online_ttft_start,
    )
    return outputs, question_ids.shape[1], PipelineMetrics(
        ttft_ms=ttft_ms,
        excluded_prep_ms=excluded_prep_ms,
        selection_ms=selection_ms,
        recompute_ms=recompute_ms,
        selected_tokens=sum(len(heads) for heads in selected_token_heads.values()),
        selected_runs=selected_run_count,
    )

def decode_generated_text(tokenizer, outputs, prompt_len):
    # 去掉模板残留、think 标签和常见前缀，让评测文本更干净。
    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<|im_end|>", " ")
    text = text.replace("<|endoftext|>", " ")
    text = re.sub(r"^\s*(Answer|Assistant|Summary)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def run_example_eval(model, tokenizer):
    # 小样例回归测试，便于快速观察三条 pipeline 的输出差异。
    decode_config = DecodeConfig(max_new_tokens=96, do_sample=False, seed=1234)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    # 更充实的测试例子：多段落技术文本
    context = (
        "The transformer architecture introduces self-attention mechanisms to process sequences. "
        "Self-attention computes relationships between all positions in parallel, enabling efficient computation. "
        "\n\n"
        "Key innovations include multi-head attention, positional encoding, and feed-forward networks. "
        "These components work together to capture complex dependencies in sequential data. "
        "\n\n"
        "Transformers have become the foundation for modern large language models and achieved state-of-the-art results across NLP tasks."
    )
    question = "What are the main components of the transformer architecture?"

    print("\n" + "=" * 70)
    print("Recompute Pipeline 示例测试")
    print("=" * 70)
    print(f"\nContext: {context[:60]}...")
    print(f"Question: {question}")
    print("\n" + "-" * 70)

    # 基线：完整前向传播
    print("\n[1] 基线方案 (Base Pipeline - Full Forward Pass)")
    ref_outputs, ref_prompt_len, ref_metrics = base_pipeline(
        model,
        tokenizer,
        context,
        question,
        decode_config=decode_config,
        stop_token_ids=stop_token_ids,
    )
    ref_generated_text = decode_generated_text(tokenizer, ref_outputs, ref_prompt_len)
    print(f"生成文本:\n  {ref_generated_text}")
    print(f"TTFT (base): {ref_metrics.ttft_ms:.2f} ms")

    # 简单拼接（有重定位）：多chunk KV拼接 + RoPE位置重定位
    print("\n[2] 简单拼接方案 - 有重定位 (Simple Concat - With Reposition)")
    cc_outputs, cc_prompt_len, cc_metrics = simple_concat_pipeline(
        model,
        tokenizer,
        context,
        question,
        decode_config=decode_config,
        stop_token_ids=stop_token_ids,
    )
    cc_generated_text = decode_generated_text(tokenizer, cc_outputs, cc_prompt_len)
    print(f"生成文本:\n  {cc_generated_text}")
    print(f"TTFT (simple_concat, 不含预计算): {cc_metrics.ttft_ms:.2f} ms")
    cc_score = rouge_score(cc_generated_text, ref_generated_text)
    print(f"ROUGE-L F1 分数 (vs 基线): {cc_score:.4f}")

    # 随机重计算：随机选择部分token重新计算
    print("\n[3] 随机重计算方案 (Random Recompute)")
    r_outputs, r_prompt_len, r_metrics = random_recompute_pipeline(
        model,
        tokenizer,
        context,
        question,
        decode_config=decode_config,
        stop_token_ids=stop_token_ids,
    )
    r_generated_text = decode_generated_text(tokenizer, r_outputs, r_prompt_len)
    print(f"生成文本:\n  {r_generated_text}")
    print(f"TTFT (blend_recompute, 不含预计算): {r_metrics.ttft_ms:.2f} ms")
    print(
        f"  选点 {r_metrics.selected_tokens} tokens / {r_metrics.selected_runs} runs, "
        f"selection={r_metrics.selection_ms:.2f} ms, recompute={r_metrics.recompute_ms:.2f} ms"
    )
    r_score = rouge_score(r_generated_text, ref_generated_text)
    print(f"ROUGE-L F1 分数 (vs 基线): {r_score:.4f}")

    print("\n" + "=" * 70)
    print("示例总结:")
    print(f"  简单拼接(有重定位) ROUGE 相似度: {cc_score:.4f}")
    print(f"  随机重计算 ROUGE 相似度: {r_score:.4f}")
    print("=" * 70 + "\n")


def evaluate_on_2wikimqa(model, tokenizer, input_path, max_samples=-1):
    # 2wikimqa 评测入口，同时输出质量指标与 TTFT。
    decode_config = get_decode_config("2wikimqa")
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    eval_samples = data if max_samples == -1 else data[:max_samples]
    if len(eval_samples) == 0:
        raise ValueError("输入文件中没有可用样本")

    base_f1_scores = []
    cc_f1_scores = []
    r_f1_scores = []
    rh_f1_scores = []
    base_ttft_scores = []
    cc_ttft_scores = []
    r_ttft_scores = []
    rh_ttft_scores = []

    print("\n" + "=" * 70)
    print("Recompute Pipeline 在 2wikimqa.jsonl 上测评")
    print("=" * 70)
    print(f"样本数: {len(eval_samples)}")

    for idx, sample in enumerate(eval_samples, start=1):
        context = sample.get("context", "")
        question_raw = sample.get("input", "")
        answers = sample.get("answers", [])
        if not isinstance(answers, list):
            answers = [str(answers)]

        formatted_context, formatted_question = build_2wikimqa_prompt(context, question_raw)

        print("\n" + "-" * 70)
        print(f"样本 {idx}/{len(eval_samples)}")
        print(f"question: {question_raw}")

        full_model_input_text = formatted_context + formatted_question
        print("模型输入前 40 词:")
        print(first_n_words(full_model_input_text, 40))

        ref_outputs, ref_prompt_len, ref_metrics = base_pipeline(
            model,
            tokenizer,
            formatted_context,
            formatted_question,
            decode_config=decode_config,
            stop_token_ids=stop_token_ids,
        )
        ref_generated_text = decode_generated_text(tokenizer, ref_outputs, ref_prompt_len)
        print("基线输出:")
        print(ref_generated_text)
        print(f"TTFT (base): {ref_metrics.ttft_ms:.2f} ms")
        base_ttft_scores.append(ref_metrics.ttft_ms)
        del ref_outputs
        release_cuda_memory()

        cc_outputs, cc_prompt_len, cc_metrics = simple_concat_pipeline(
            model,
            tokenizer,
            formatted_context,
            formatted_question,
            decode_config=decode_config,
            stop_token_ids=stop_token_ids,
        )
        cc_generated_text = decode_generated_text(tokenizer, cc_outputs, cc_prompt_len)
        print("simple_concat 输出:")
        print(cc_generated_text)
        print(f"TTFT (simple_concat, 不含预计算): {cc_metrics.ttft_ms:.2f} ms")
        cc_ttft_scores.append(cc_metrics.ttft_ms)
        del cc_outputs
        release_cuda_memory()

        r_outputs, r_prompt_len, r_metrics = random_recompute_pipeline(
            model,
            tokenizer,
            formatted_context,
            formatted_question,
            decode_config=decode_config,
            stop_token_ids=stop_token_ids,
        )
        r_generated_text = decode_generated_text(tokenizer, r_outputs, r_prompt_len)
        print("random_recompute 输出:")
        print(r_generated_text)
        print(f"TTFT (random_recompute, 不含预计算): {r_metrics.ttft_ms:.2f} ms")
        print(
            f"  选点 {r_metrics.selected_tokens} tokens / {r_metrics.selected_runs} runs, "
            f"selection={r_metrics.selection_ms:.2f} ms, recompute={r_metrics.recompute_ms:.2f} ms"
        )
        r_ttft_scores.append(r_metrics.ttft_ms)
        del r_outputs
        release_cuda_memory()

        rh_outputs, rh_prompt_len, rh_metrics = random_recompute_headwise_pipeline(
            model,
            tokenizer,
            formatted_context,
            formatted_question,
            decode_config=decode_config,
            stop_token_ids=stop_token_ids,
        )
        rh_generated_text = decode_generated_text(tokenizer, rh_outputs, rh_prompt_len)
        print("random_recompute_headwise 输出:")
        print(rh_generated_text)
        print(f"TTFT (random_recompute_headwise, 不含预计算): {rh_metrics.ttft_ms:.2f} ms")
        print(
            f"  选中 {rh_metrics.selected_tokens} 个 token-head 对 / {rh_metrics.selected_runs} runs, "
            f"selection={rh_metrics.selection_ms:.2f} ms, recompute={rh_metrics.recompute_ms:.2f} ms"
        )
        rh_ttft_scores.append(rh_metrics.ttft_ms)
        del rh_outputs
        release_cuda_memory()

        print("标准答案:")
        print(answers)

        base_f1 = f1_score_any(ref_generated_text, answers)
        cc_f1 = f1_score_any(cc_generated_text, answers)
        r_f1 = f1_score_any(r_generated_text, answers)
        rh_f1 = f1_score_any(rh_generated_text, answers)

        base_f1_scores.append(base_f1)
        cc_f1_scores.append(cc_f1)
        r_f1_scores.append(r_f1)
        rh_f1_scores.append(rh_f1)

        print(f"F1 (base): {base_f1:.4f}")
        print(f"F1 (simple_concat): {cc_f1:.4f}")
        print(f"F1 (random_recompute): {r_f1:.4f}")
        print(f"F1 (random_recompute_headwise): {rh_f1:.4f}")

        release_cuda_memory()

    def avg(scores):
        return sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 70)
    print("2wikimqa 测评汇总")
    print(f"平均 F1 (base): {avg(base_f1_scores):.4f}")
    print(f"平均 F1 (simple_concat): {avg(cc_f1_scores):.4f}")
    print(f"平均 F1 (random_recompute): {avg(r_f1_scores):.4f}")
    print(f"平均 F1 (random_recompute_headwise): {avg(rh_f1_scores):.4f}")
    print(f"平均 TTFT (base): {avg(base_ttft_scores):.2f} ms")
    print(f"平均 TTFT (simple_concat, 不含预计算): {avg(cc_ttft_scores):.2f} ms")
    print(f"平均 TTFT (random_recompute, 不含预计算): {avg(r_ttft_scores):.2f} ms")
    print(f"平均 TTFT (random_recompute_headwise, 不含预计算): {avg(rh_ttft_scores):.2f} ms")
    print("=" * 70 + "\n")


def parse_pipeline_names(pipelines_arg):
    # 解析命令行里的 pipeline 选择，过滤非法名字。
    valid_names = {"base", "simple_concat", "random_recompute", "random_recompute_headwise"}
    selected = []
    for name in pipelines_arg.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in valid_names:
            raise ValueError(f"不支持的 pipeline: {name}")
        selected.append(name)

    if not selected:
        raise ValueError("至少需要选择一个 pipeline")

    return selected


def main():
    # 主入口：加载模型、识别数据集，并分发到对应评测流程。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Local model path or Hugging Face repo id"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="./gov_report.jsonl", help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate. Use -1 for all samples.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "gov_report", "2wikimqa"],
        help="Dataset type for evaluation",
    )
    parser.add_argument(
        "--run-example",
        action="store_true",
        help="Run the small demo example before dataset evaluation",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="base,simple_concat,random_recompute,random_recompute_headwise",
        help="Comma-separated pipeline names. Supported: base,simple_concat,random_recompute,random_recompute_headwise",
    )
    args = parser.parse_args()

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    model_load_kwargs = build_model_load_kwargs()
    print(f"模型加载参数: {model_load_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    selected_pipelines = parse_pipeline_names(args.pipelines)

    dataset = args.dataset
    if dataset == "auto":
        filename = os.path.basename(args.input).lower()
        dataset = "2wikimqa" if "2wikimqa" in filename else "gov_report"

    if dataset == "2wikimqa":
        if data:
            first_input = data[0].get("input", "")
            if not isinstance(first_input, str) or not first_input.strip():
                raise ValueError(
                    "当前输入文件看起来不是 2wikimqa（样本 input 为空）。"
                    "请使用 -i ./2wikimqa.jsonl，或将 --dataset 设置为 gov_report。"
                )
        evaluate_on_2wikimqa(
            model=model,
            tokenizer=tokenizer,
            input_path=args.input,
            max_samples=args.max_samples,
        )
        return

    if args.run_example:
        run_example_eval(model, tokenizer)

    eval_samples = data if args.max_samples == -1 else data[: args.max_samples]
    if len(eval_samples) == 0:
        raise ValueError("输入文件中没有可用样本")

    base_vs_gt_scores = []
    cc_vs_gt_scores = []
    r_vs_gt_scores = []
    rh_vs_gt_scores = []
    base_ttft_scores = []
    cc_ttft_scores = []
    r_ttft_scores = []
    rh_ttft_scores = []
    decode_config = get_decode_config("gov_report")

    print("\n" + "=" * 70)
    print("Recompute Pipeline 在 gov_report.jsonl 上测评")
    print("=" * 70)
    print(f"样本数: {len(eval_samples)}")

    for idx, sample in enumerate(eval_samples, start=1):
        context = sample.get("context", "")
        answers = sample.get("answers", [])
        ground_truth = answers[0] if isinstance(answers, list) and len(answers) > 0 else ""
        formatted_context, formatted_question = build_gov_report_prompt(context)

        print("\n" + "-" * 70)
        print(f"样本 {idx}/{len(eval_samples)}")
        print(f"context 前缀: {context[:80].replace(chr(10), ' ')}...")

        # 打印模型输入前40词（template + context + instruction）
        full_model_input_text = formatted_context + formatted_question
        print("模型输入前 40 词:")
        print(first_n_words(full_model_input_text, 40))

        # 基线：完整前向传播
        ref_generated_text = None
        cc_generated_text = None
        r_generated_text = None
        rh_generated_text = None

        if "base" in selected_pipelines:
            ref_outputs, ref_prompt_len, ref_metrics = base_pipeline(
                model,
                tokenizer,
                formatted_context,
                formatted_question,
                decode_config=decode_config,
                stop_token_ids=stop_token_ids,
            )
            ref_generated_text = decode_generated_text(tokenizer, ref_outputs, ref_prompt_len)
            print("基线输出:")
            print(ref_generated_text)
            print(f"TTFT (base): {ref_metrics.ttft_ms:.2f} ms")
            base_ttft_scores.append(ref_metrics.ttft_ms)
            del ref_outputs
            release_cuda_memory()

        if "simple_concat" in selected_pipelines:
            cc_outputs, cc_prompt_len, cc_metrics = simple_concat_pipeline(
                model,
                tokenizer,
                formatted_context,
                formatted_question,
                decode_config=decode_config,
                stop_token_ids=stop_token_ids,
            )
            cc_generated_text = decode_generated_text(tokenizer, cc_outputs, cc_prompt_len)
            print("simple_concat 输出:")
            print(cc_generated_text)
            print(f"TTFT (simple_concat, 不含预计算): {cc_metrics.ttft_ms:.2f} ms")
            cc_ttft_scores.append(cc_metrics.ttft_ms)
            del cc_outputs
            release_cuda_memory()

        if "random_recompute" in selected_pipelines:
            r_outputs, r_prompt_len, r_metrics = random_recompute_pipeline(
                model,
                tokenizer,
                formatted_context,
                formatted_question,
                decode_config=decode_config,
                stop_token_ids=stop_token_ids,
            )
            r_generated_text = decode_generated_text(tokenizer, r_outputs, r_prompt_len)
            print("blend_recompute 输出:")
            print(r_generated_text)
            print(f"TTFT (blend_recompute, 不含预计算): {r_metrics.ttft_ms:.2f} ms")
            print(
                f"  选点 {r_metrics.selected_tokens} tokens / {r_metrics.selected_runs} runs, "
                f"selection={r_metrics.selection_ms:.2f} ms, recompute={r_metrics.recompute_ms:.2f} ms"
            )
            r_ttft_scores.append(r_metrics.ttft_ms)
            del r_outputs
            release_cuda_memory()

        if "random_recompute_headwise" in selected_pipelines:
            rh_outputs, rh_prompt_len, rh_metrics = random_recompute_headwise_pipeline(
                model,
                tokenizer,
                formatted_context,
                formatted_question,
                decode_config=decode_config,
                stop_token_ids=stop_token_ids,
            )
            rh_generated_text = decode_generated_text(tokenizer, rh_outputs, rh_prompt_len)
            print("blend_recompute_headwise 输出:")
            print(rh_generated_text)
            print(f"TTFT (blend_recompute_headwise, 不含预计算): {rh_metrics.ttft_ms:.2f} ms")
            print(
                f"  选中 {rh_metrics.selected_tokens} 个 token-head 对 / {rh_metrics.selected_runs} runs, "
                f"selection={rh_metrics.selection_ms:.2f} ms, recompute={rh_metrics.recompute_ms:.2f} ms"
            )
            rh_ttft_scores.append(rh_metrics.ttft_ms)
            del rh_outputs
            release_cuda_memory()

        print("标准答案:")
        print(ground_truth if ground_truth else "(空)")

        if ground_truth:
            if ref_generated_text is not None:
                base_vs_gt_scores.append(rouge_score(ref_generated_text, ground_truth))
            if cc_generated_text is not None:
                cc_vs_gt_scores.append(rouge_score(cc_generated_text, ground_truth))
            if r_generated_text is not None:
                r_vs_gt_scores.append(rouge_score(r_generated_text, ground_truth))
            if rh_generated_text is not None:
                rh_vs_gt_scores.append(rouge_score(rh_generated_text, ground_truth))

        if ground_truth and ref_generated_text is not None:
            print(f"ROUGE-L F1 (base vs ground_truth): {base_vs_gt_scores[-1]:.4f}")
        if ground_truth and cc_generated_text is not None:
            print(f"ROUGE-L F1 (simple_concat_with_reposition vs ground_truth): {cc_vs_gt_scores[-1]:.4f}")
        if ground_truth and r_generated_text is not None:
            print(f"ROUGE-L F1 (blend_recompute vs ground_truth): {r_vs_gt_scores[-1]:.4f}")
        if ground_truth and rh_generated_text is not None:
            print(f"ROUGE-L F1 (blend_recompute_headwise vs ground_truth): {rh_vs_gt_scores[-1]:.4f}")

        release_cuda_memory()

    def avg(scores):
        return sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 70)
    print("测评汇总")
    if base_vs_gt_scores:
        print(f"平均 ROUGE-L F1 (base vs ground_truth): {avg(base_vs_gt_scores):.4f}")
        print(f"平均 TTFT (base): {avg(base_ttft_scores):.2f} ms")
    if cc_vs_gt_scores:
        print(f"平均 ROUGE-L F1 (simple_concat_with_reposition vs ground_truth): {avg(cc_vs_gt_scores):.4f}")
        print(f"平均 TTFT (simple_concat, 不含预计算): {avg(cc_ttft_scores):.2f} ms")
    if r_vs_gt_scores:
        print(f"平均 ROUGE-L F1 (blend_recompute vs ground_truth): {avg(r_vs_gt_scores):.4f}")
        print(f"平均 TTFT (blend_recompute, 不含预计算): {avg(r_ttft_scores):.2f} ms")
    if rh_vs_gt_scores:
        print(f"平均 ROUGE-L F1 (blend_recompute_headwise vs ground_truth): {avg(rh_vs_gt_scores):.4f}")
        print(f"平均 TTFT (blend_recompute_headwise, 不含预计算): {avg(rh_ttft_scores):.2f} ms")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

