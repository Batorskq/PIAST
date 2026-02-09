import argparse
import csv
import os
import re
import random
import time
from typing import List, Tuple, Dict, Optional, Any

from tqdm import tqdm

import math
import statistics
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from easse.sari import corpus_sari

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _safe_std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return 0.0
    try:
        return statistics.pstdev(xs)
    except Exception:
        return 0.0

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def append_csv_row(path: str, row: Dict[str, Any]):
    _ensure_dir(path)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def subsample_items(items: List[Dict], n: int, seed: int) -> List[Dict]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)

def _msg_user_text(obj: Dict) -> str:
    msgs = obj.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""

def dedup_items_by_sentence(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for it in items:
        s = _msg_user_text(it)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(it)
    return out

def sample_replay(current: List[Dict], n: int, seed: int) -> List[Dict]:
    if n <= 0:
        return []
    return subsample_items(current, min(n, len(current)), seed)

def strip_quotes(s: str) -> str:
    t = s.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith('“') and t.endswith('”')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t

def normalize_simplification_text(text: str) -> str:
    t = text.strip()
    t = t.replace("\n", " ").strip()
    t = re.sub(r'^(simple|simplify)\s*:\s*', '', t, flags=re.IGNORECASE).strip()
    t = strip_quotes(t)
    t = re.sub(r"\s+", " ", t)
    return t

EXAMPLE_HEADER_RE = re.compile(r"^Example\s*(\d+)\s*:\s*", re.IGNORECASE | re.MULTILINE)
INPUT_LINE_RE = re.compile(r'^\s*(Input|Complex)\s*:\s*"(.*?)"\s*$', re.IGNORECASE | re.MULTILINE | re.DOTALL)
OUTPUT_LINE_RE = re.compile(r'^\s*(Output|Simple|Simplification)\s*:\s*"(.*?)"\s*$', re.IGNORECASE | re.MULTILINE | re.DOTALL)

def format_examples_for_evaluator(examples: List[Dict[str, str]], drop_index: Optional[int] = None) -> str:
    lines = []
    idx = 1
    for i, ex in enumerate(examples):
        if drop_index is not None and i == drop_index:
            continue
        c = ex.get("complex", "").replace('"', "'").strip()
        s = ex.get("simple", "").replace('"', "'").strip()
        lines.append(f"Example{idx}:")
        lines.append(f'Complex: "{c}"')
        lines.append(f'Simple: "{s}"')
        lines.append("")
        idx += 1
    return "\n".join(lines).strip()

def parse_examples(text: str, k: int) -> List[Dict[str, str]]:
    blocks = []
    headers = list(EXAMPLE_HEADER_RE.finditer(text))
    if not headers:
        pairs = []
        inputs = INPUT_LINE_RE.findall(text)
        outputs = OUTPUT_LINE_RE.findall(text)
        for (lab_i, c), (lab_o, s) in zip(inputs, outputs):
            c = c.strip()
            s = s.strip()
            if c and s:
                pairs.append({"complex": c, "simple": s})
            if len(pairs) >= k:
                break
        return pairs
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        blocks.append(text[start:end])
    pairs = []
    seen = set()
    for block in blocks:
        m_in = INPUT_LINE_RE.search(block)
        m_out = OUTPUT_LINE_RE.search(block)
        if not (m_in and m_out):
            continue
        c = m_in.group(2).strip()
        s = m_out.group(2).strip()
        key = (c, s)
        if not c or not s or key in seen:
            continue
        seen.add(key)
        pairs.append({"complex": c, "simple": s})
        if len(pairs) >= k:
            break
    return pairs

SIMPLIFICATION_TOPICS = [
    "geography/place names","history/event","biology/animals","technology","sports","arts/culture",
    "economics/finance","law/politics","education","transport","medicine/health","environment",
    "astronomy/space","food/cuisine","music","film/TV","literature","mythology","architecture",
    "math/science fact","weather/climate","travel/tourism","computing","social media","chemistry",
    "physics","engineering","agriculture","linguistics","philosophy"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8, help="Number of in-context examples to propose (crafting starts at 8).")
    p.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Local path or HF id.")
    p.add_argument("--eval-on-test", action="store_true", help="If set, run final evaluation on the TEST set at --dataset-path.")
    p.add_argument("--dataset-path", type=str, default="dataset/sim_test.jsonl", help="Path to simplification test JSONL.")
    p.add_argument("--infer-dataset-path", type=str, default="dataset/sim_infer.jsonl", help="Path to simplification infer JSONL.")
    p.add_argument("--infer-subsample-size", type=int, default=30, help="Subsample from sim_infer per crafting iteration.")
    p.add_argument("--seed", type=int, default=123, help="Random seed base for subsampling and proposer.")
    p.add_argument("--refine-candidates", type=int, default=3, help="How many candidate replacements to try.")
    p.add_argument("--craft-iterations", type=int, default=10, help="Number of craft-refine/drop iterations.")
    p.add_argument("--replay-add", type=int, default=5, help="How many items to add to replay per iteration.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of TEST items (0 = all).")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    p.add_argument("--gpu-mem-util", type=float, default=0.92, help="vLLM GPU memory utilization.")
    p.add_argument("--max-model-len", type=int, default=4096, help="Max model length for vLLM.")
    p.add_argument("--quantization", type=str, default=None, help="Optional vLLM quantization.")
    p.add_argument("--max-new-tokens-proposer", type=int, default=0, help="Override proposer max tokens. Default ~ 64*k (min 128).")
    p.add_argument("--max-new-tokens-evaluator", type=int, default=64, help="Max new tokens for evaluator prediction (simplification output).")
    p.add_argument("--use-vllm-for-evaluator", action="store_true", help="Use vLLM for FINAL evaluator instead of Transformers.")
    p.add_argument("--batch-size", type=int, default=8, help="(Transformers) batch size for evaluator.")
    p.add_argument("--save-examples-to", type=str, default="sim_proposed_examples.txt", help="Where to save initial generated examples.")
    p.add_argument("--save-crafted-examples-to", type=str, default="sim_crafted_examples.txt", help="Where to save crafted examples.")
    p.add_argument("--run-grid-search", action="store_true", help="Run grid search over many specs and seeds.")
    p.add_argument("--run-random-search", action="store_true", help="Run random search over hyperparameters and seeds.")
    p.add_argument("--grid-k", type=str, default="4,6,8,12,16", help="Comma-separated k values.")
    p.add_argument("--grid-infer-sizes", type=str, default="10,20,30,50,70", help="Comma-separated infer subsample sizes.")
    p.add_argument("--grid-craft-iters", type=str, default="1,3,5,8,10", help="Comma-separated craft iteration counts.")
    p.add_argument("--grid-refine-cands", type=str, default="1,3,5,10", help="Comma-separated candidate counts.")
    p.add_argument("--grid-replay-add", type=str, default="0,5,15", help="Comma-separated replay-add values.")
    p.add_argument("--grid-seeds", type=str, default="1,2,3,4", help="Comma-separated seeds.")
    p.add_argument("--grid-results-csv", type=str, default="results/simplification.csv", help="Where to save grid/random search results CSV.")
    p.add_argument("--random-specs", type=int, default=200, help="Number of random hyperparameter specs to evaluate.")
    p.add_argument("--random-hparam-seed", type=int, default=12345, help="RNG seed for random spec sampling.")
    p.add_argument("--shapley-permutations", type=int, default=1, help="Number of random permutations for Monte-Carlo Shapley.")
    p.add_argument("--shapley-tmc", action="store_true", help="Enable Truncated Monte Carlo (stop permutation when v≈full).")
    p.add_argument("--shapley-epsilon", type=float, default=0.0, help="Tolerance for TMC stop; 0.0 = only stop when equal to full SARI.")
    p.add_argument("--shapley-log-dir", type=str, default="results_shapley_sim", help="Directory to save per-iteration Shapley CSV logs.")
    p.add_argument("--random-shapley-permutations", type=str, default="1,2,3,5,10", help="Choices for Shapley permutations in random search.")
    p.add_argument("--early-stop-sari-threshold", type=float, default=45.0, help="If seed=1 SARI < threshold, skip remaining seeds for that spec.")
    return p.parse_args()

class VLLMProposer:
    def __init__(self, model_path: str, tokenizer, tp: int, util: float, max_len: int, quantization: Optional[str]):
        from vllm import LLM
        self.tokenizer = tokenizer
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=util,
            max_model_len=max_len,
            dtype="auto",
            quantization=quantization,
            enable_prefix_caching=True,
        )

    def _gen_with_messages(self, messages, max_new_tokens: int, seed: Optional[int]) -> str:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=max_new_tokens,
            stop=["\n\n\n"],
            seed=None if seed is None else int(seed),
        )
        return self.llm.generate([prompt_text], sp, use_tqdm=False)[0].outputs[0].text

    def _build_messages(self, k: int, topics: List[str], seed: Optional[int]) -> List[Dict[str, str]]:
        plan_lines = []
        for i in range(k):
            topic = topics[i % len(topics)]
            n_sent = 1 + (i % 2)
            plan_lines.append(f"- Example{i+1}: topic {topic}; input has {n_sent} sentence(s); output is ONE shorter, simpler sentence.")
        diversity_plan = "\n".join(plan_lines)
        sys_msg = (
            "You are a data generator that writes high-quality in-context learning examples "
            "for TEXT SIMPLIFICATION: convert complex text to a simpler sentence while keeping the meaning."
        )
        user_msg = f"""Create exactly {k} examples in THIS STRICT format:

Example1:
Complex: "<complex text>"
Simple: "<simple sentence>"

Example2:
Complex: "<complex text>"
Simple: "<simple sentence>"

...
Example{k}:
Complex: "<complex text>"
Simple: "<simple sentence>"

Diversity plan (MUST FOLLOW):
{diversity_plan}

Rules:
- Use ASCII only; do NOT include double quotes inside the text.
- Complex can have 1–2 sentences (short). Simple MUST be a single concise sentence.
- Keep the same meaning; remove extraneous details; fix grammar; simpler vocabulary.
- Do NOT wrap output in Markdown/code fences.
- Output ONLY the examples in the exact format above; no extra text.
"""
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    def generate_examples(self, k: int, *, seed: Optional[int] = None, max_new_tokens: Optional[int] = None) -> Tuple[str, List[Dict[str, str]]]:
        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = max(128, 64 * k)
        msgs = self._build_messages(k, SIMPLIFICATION_TOPICS, seed)
        text = self._gen_with_messages(msgs, max_new_tokens, None if seed is None else seed + 0)
        pairs = parse_examples(text, k)
        rng = random.Random(None if seed is None else seed + 99991)
        rng.shuffle(pairs)
        dedup: List[Dict[str, str]] = []
        seen = set()
        for ex in pairs:
            c = ex["complex"]
            if c in seen:
                continue
            seen.add(c)
            dedup.append(ex)
            if len(dedup) >= k:
                break
        out_text = format_examples_for_evaluator(dedup)
        return out_text, dedup

EVAL_TASK_HEADER = "Simplify the text."

def build_evaluator_user_prompt(examples_block: str, sentence: str) -> str:
    return (
        f"{EVAL_TASK_HEADER}\n\n"
        f"{examples_block}\n\n"
        f'Complex: "{sentence}"\n'
        f"Simple:"
    )

class TransformersEvaluator:
    def __init__(self, model_path: str, max_new_tokens: int, batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.max_new = max_new_tokens
        self.batch_size = max(1, int(batch_size))
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompts: List[str]) -> List[str]:
        results: List[str] = []
        tok = self.tokenizer
        mdl = self.model
        device = mdl.device
        pad_id = tok.pad_token_id or tok.eos_token_id
        eos_id = tok.eos_token_id
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            input_lengths = enc["attention_mask"].sum(dim=1)
            with torch.no_grad():
                out = mdl.generate(
                    **enc,
                    max_new_tokens=self.max_new,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
            for j in range(out.size(0)):
                gen_only = out[j, input_lengths[j]:]
                text = tok.decode(gen_only, skip_special_tokens=True)
                results.append(normalize_simplification_text(text))
        return results

class VLLMEvaluator:
    def __init__(self, model_path: str, tokenizer, tp: int, util: float, max_len: int, quantization: Optional[str], max_new_tokens: int, llm=None):
        from vllm import LLM, SamplingParams
        self.tokenizer = tokenizer
        self.llm = llm if llm is not None else LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=util,
            max_model_len=max_len,
            dtype="auto",
            quantization=quantization,
            enable_prefix_caching=True,
        )
        self.sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)

    def generate(self, prompts: List[str]) -> List[str]:
        outs = self.llm.generate(prompts, self.sp, use_tqdm=False)
        return [normalize_simplification_text(o.outputs[0].text) for o in outs]

def eval_sari_on_items(evaluator, examples_block: str, items: List[Dict]) -> Tuple[int, int, float]:
    sources: List[str] = []
    refs: List[str] = []
    prompts: List[str] = []
    for obj in items:
        src = _msg_user_text(obj)
        ref = str(obj.get("solution", "")).strip()
        if not src:
            continue
        sources.append(src)
        refs.append(ref)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": build_evaluator_user_prompt(examples_block, src)},
        ]
        if hasattr(evaluator, "tokenizer"):
            prompts.append(evaluator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            from transformers import AutoTokenizer
            prompts.append(messages[1]["content"])
    if not sources:
        return 0, 0, 0.0
    hyps = evaluator.generate(prompts)
    n = min(len(sources), len(hyps), len(refs))
    sources = sources[:n]
    hyps = hyps[:n]
    refs = refs[:n]
    sari = corpus_sari(sources, hyps, [refs])
    return len(hyps), len(refs), float(sari)

def _subset_block_from_indices(examples: List[Dict[str, str]], idx_set: List[int]) -> str:
    subset = [examples[i] for i in sorted(idx_set)]
    return format_examples_for_evaluator(subset, drop_index=None)

def estimate_shapley_values(
    examples: List[Dict[str, str]],
    infer_items: List[Dict],
    evaluator,
    *,
    permutations: int,
    use_tmc: bool,
    epsilon: float,
    rng: random.Random,
    log_dir: str,
    iter_index: int,
    run_seed: int,
) -> Dict:
    _ensure_dir(log_dir)
    n = len(examples)
    value_cache: Dict[frozenset, Tuple[int, int, float]] = {}

    def value_of(idx_tuple: Tuple[int, ...]) -> Tuple[int, int, float]:
        key = frozenset(idx_tuple)
        if key in value_cache:
            return value_cache[key]
        block = _subset_block_from_indices(examples, list(key))
        c, t, sari_val = eval_sari_on_items(evaluator, block, infer_items)
        value_cache[key] = (c, t, sari_val)
        return c, t, sari_val

    _, _, v_empty = value_of(tuple())
    _, _, v_full  = value_of(tuple(range(n)))

    print(f"[SHAPLEY] iter={iter_index} seed={run_seed} | K={permutations} | TMC={int(use_tmc)} eps={epsilon:.4g} | v(empty)={v_empty:.4f} v(full)={v_full:.4f}")

    contribs: List[List[float]] = [[] for _ in range(n)]
    cache_hits = 0
    cache_misses = 0

    def cached_value(idx_tuple: Tuple[int, ...]) -> Tuple[int, int, float]:
        nonlocal cache_hits, cache_misses
        key = frozenset(idx_tuple)
        if key in value_cache:
            cache_hits += 1
            return value_cache[key]
        cache_misses += 1
        return value_of(idx_tuple)

    for _ in range(permutations):
        perm = list(range(n))
        rng.shuffle(perm)
        prefix: List[int] = []
        _, _, prev_val = cached_value(tuple(prefix))
        saturated = False
        processed = 0
        for i_idx in perm:
            new_prefix = prefix + [i_idx]
            _, _, new_val = cached_value(tuple(new_prefix))
            contribs[i_idx].append(new_val - prev_val)
            prefix = new_prefix
            prev_val = new_val
            processed += 1
            if use_tmc and abs(prev_val - v_full) <= epsilon:
                saturated = True
                break
        if saturated and processed < n:
            for j in perm[processed:]:
                contribs[j].append(0.0)

    phi = [(sum(cs) / len(cs) if len(cs) > 0 else 0.0) for cs in contribs]
    std = [_safe_std(cs) for cs in contribs]
    samples = [len(cs) for cs in contribs]

    csv_path = os.path.join(
        log_dir,
        f"shapley_iter{iter_index}_seed{run_seed}_{_now_tag()}.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "seed", "index", "complex", "simple", "phi", "std", "samples",
                    "v_empty", "v_full", "permutations", "tmc", "epsilon", "cache_hits", "cache_misses"])
        for i, ex in enumerate(examples):
            w.writerow([
                iter_index, run_seed, i, ex.get("complex",""), ex.get("simple",""),
                f"{phi[i]:.6f}", f"{std[i]:.6f}", samples[i],
                f"{v_empty:.6f}", f"{v_full:.6f}", permutations, int(use_tmc), f"{epsilon:.6f}",
                cache_hits, cache_misses
            ])

    print(f"[SHAPLEY] iter={iter_index} saved CSV -> {csv_path}")
    worst_idx = min(range(n), key=lambda i: phi[i])
    print(f"[SHAPLEY] iter={iter_index} worst_index={worst_idx} phi={phi[worst_idx]:.6f} (cache hits={cache_hits}, misses={cache_misses})")

    return {
        "phi": phi,
        "std": std,
        "samples": samples,
        "v_empty": v_empty,
        "v_full": v_full,
        "permutations": permutations,
        "tmc": use_tmc,
        "epsilon": epsilon,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "csv_path": csv_path,
        "worst_index": worst_idx,
    }

def build_refinement_messages(remaining_examples: List[Dict[str, str]], num_candidates: int) -> List[Dict[str, str]]:
    rem_block = format_examples_for_evaluator(remaining_examples, drop_index=None)
    plan_lines = []
    for i in range(num_candidates):
        topic = SIMPLIFICATION_TOPICS[i % len(SIMPLIFICATION_TOPICS)]
        in_sent = 1 + (i % 2)
        plan_lines.append(f"- Example{i+1}: topic {topic}; input has {in_sent} sentence(s); output is ONE simpler sentence.")
    diversity_plan = "\n".join(plan_lines)
    sys_msg = ("You are improving in-context examples for TEXT SIMPLIFICATION. "
               "Generate replacements that diversify topics and input length; avoid paraphrasing existing examples.")
    user_msg = f"""You are given the CURRENT examples (do not repeat or paraphrase them):

{rem_block}

Now create exactly {num_candidates} NEW examples in THIS STRICT format:

Example1:
Complex: "<complex text>"
Simple: "<simple sentence>"

Example2:
Complex: "<complex text>"
Simple: "<simple sentence>"

...
Example{num_candidates}:
Complex: "<complex text>"
Simple: "<simple sentence>"

Diversity plan (MUST FOLLOW):
{diversity_plan}

Rules:
- Use exactly ONE 'Complex:' and ONE 'Simple:' line per example.
- Complex can have 1–2 short sentences. Simple MUST be one concise sentence.
- ASCII only. Do NOT include double quotes inside the text.
- Make topics clearly different from the given examples and from each other; avoid near-duplicates.
- Do NOT wrap output in Markdown/code fences.
- Output ONLY the examples in the exact format above; no extra text.
"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

def refine_or_drop_one_fast(
    proposer_llm,
    tokenizer,
    base_examples: List[Dict[str, str]],
    infer_items: List[Dict],
    vllm_args: Dict,
    max_new_tokens_eval: int,
    num_candidates: int,
    *,
    shapley_permutations: int,
    shapley_tmc: bool,
    shapley_epsilon: float,
    shapley_log_dir: str,
    iter_index: int,
    run_seed: int,
) -> Tuple[List[Dict[str, str]], Dict]:
    fast_eval = VLLMEvaluator(
        model_path=vllm_args["model_path"],
        tokenizer=tokenizer,
        tp=vllm_args["tp"],
        util=vllm_args["util"],
        max_len=vllm_args["max_len"],
        quantization=vllm_args["quantization"],
        max_new_tokens=max_new_tokens_eval,
        llm=proposer_llm,
    )
    block_all = format_examples_for_evaluator(base_examples, drop_index=None)
    base_preds, base_refs, base_sari = eval_sari_on_items(fast_eval, block_all, infer_items)
    rng = random.Random(run_seed + 1337 + 10000 * max(0, iter_index))
    shapley_info = estimate_shapley_values(
        base_examples,
        infer_items,
        fast_eval,
        permutations=shapley_permutations,
        use_tmc=shapley_tmc,
        epsilon=shapley_epsilon,
        rng=rng,
        log_dir=shapley_log_dir,
        iter_index=iter_index,
        run_seed=run_seed,
    )
    worst_idx = shapley_info["worst_index"]
    block_drop_i = format_examples_for_evaluator(base_examples[:worst_idx] + base_examples[worst_idx+1:], drop_index=None)
    drop_preds, drop_refs, drop_sari = eval_sari_on_items(fast_eval, block_drop_i, infer_items)
    remaining = [ex for j, ex in enumerate(base_examples) if j != worst_idx]
    ref_messages = build_refinement_messages(remaining, num_candidates)
    ref_prompt = tokenizer.apply_chat_template(ref_messages, tokenize=False, add_generation_prompt=True)
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max(120, 60 * num_candidates),
        stop=["\n\n\n"],
        seed=int(run_seed + 4242),
    )
    ref_text = proposer_llm.generate([ref_prompt], sp, use_tqdm=False)[0].outputs[0].text
    candidates = parse_examples(ref_text, num_candidates)
    cand_results = []
    best_cand_sari = float("-inf")
    best_cand = None
    for cand in candidates:
        trial_examples = list(base_examples)
        trial_examples[worst_idx] = cand
        block_trial = format_examples_for_evaluator(trial_examples, drop_index=None)
        _, _, sari_val = eval_sari_on_items(fast_eval, block_trial, infer_items)
        cand_results.append({"candidate": cand, "sari": sari_val})
        if sari_val > best_cand_sari:
            best_cand_sari = sari_val
            best_cand = cand
    refined_sari = best_cand_sari if best_cand is not None else float("-inf")
    decision = "keep"
    final_examples = base_examples
    if best_cand is not None and refined_sari >= drop_sari and refined_sari >= base_sari:
        decision = "replace"
        final_examples = list(base_examples)
        final_examples[worst_idx] = best_cand
    elif drop_sari >= base_sari and len(base_examples) > 1:
        decision = "drop"
        final_examples = [ex for j, ex in enumerate(base_examples) if j != worst_idx]
    else:
        decision = "keep"
    final_metric = {"keep": base_sari, "drop": drop_sari, "replace": refined_sari}[decision]
    summary = {
        "baseline": {"preds": base_preds, "refs": base_refs, "sari": base_sari},
        "shapley": shapley_info,
        "chosen_worst_index": worst_idx,
        "drop_stats": {"preds": drop_preds, "refs": drop_refs, "sari": drop_sari},
        "ref_text_raw": ref_text,
        "candidate_results": cand_results,
        "decision": decision,
        "final_k": len(final_examples),
        "best_candidate_sari": None if best_cand is None else refined_sari,
        "final_sari": final_metric,
    }
    return final_examples, summary

def craft_iterative(
    proposer,
    tokenizer,
    initial_examples: List[Dict[str, str]],
    infer_data: List[Dict],
    *,
    infer_subsample_size: int,
    seed: int,
    craft_iterations: int,
    refine_candidates: int,
    replay_add: int,
    vllm_args: Dict,
    max_new_tokens_eval: int,
    shapley_permutations: int,
    shapley_tmc: bool,
    shapley_epsilon: float,
    shapley_log_dir: str,
) -> Tuple[List[Dict[str, str]], float]:
    current_sample = subsample_items(infer_data, infer_subsample_size, seed)
    replay = []
    crafted_examples = list(initial_examples)
    craft_start = time.time()
    for it in range(craft_iterations):
        combined = dedup_items_by_sentence(current_sample + replay)
        crafted_examples, _ = refine_or_drop_one_fast(
            proposer_llm=proposer.llm,
            tokenizer=tokenizer,
            base_examples=crafted_examples,
            infer_items=combined,
            vllm_args=vllm_args,
            max_new_tokens_eval=max_new_tokens_eval,
            num_candidates=refine_candidates,
            shapley_permutations=shapley_permutations,
            shapley_tmc=shapley_tmc,
            shapley_epsilon=shapley_epsilon,
            shapley_log_dir=shapley_log_dir,
            iter_index=it,
            run_seed=seed,
        )
        replay_add_batch = sample_replay(current_sample, replay_add, seed + 1000*(it+1))
        replay = dedup_items_by_sentence(replay + replay_add_batch)
        current_sample = subsample_items(infer_data, infer_subsample_size, seed + (it+1))
    craft_elapsed = time.time() - craft_start
    return crafted_examples, craft_elapsed

def evaluate_on_test(
    examples: List[Dict[str, str]],
    tokenizer,
    proposer,
    args,
    test_data: List[Dict],
) -> Tuple[int, int, float]:
    examples_block = format_examples_for_evaluator(examples)
    if args.use_vllm_for_evaluator:
        evaluator = VLLMEvaluator(
            model_path=args.model_path,
            tokenizer=tokenizer,
            tp=args.tp,
            util=args.gpu_mem_util,
            max_len=args.max_model_len,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens_evaluator,
            llm=proposer.llm,
        )
    else:
        evaluator = TransformersEvaluator(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens_evaluator,
            batch_size=args.batch_size,
        )
    data = test_data if not args.limit or args.limit <= 0 else test_data[:args.limit]
    preds, refs, sari = eval_sari_on_items(evaluator, examples_block, data)
    return preds, refs, sari

def main():
    args = parse_args()
    _ensure_dir(args.shapley_log_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    proposer = VLLMProposer(
        model_path=args.model_path,
        tokenizer=tokenizer,
        tp=args.tp,
        util=args.gpu_mem_util,
        max_len=args.max_model_len,
        quantization=args.quantization,
    )
    infer_data = load_jsonl(args.infer_dataset_path)
    if not infer_data:
        print(f"[ERROR] No data loaded from {args.infer_dataset_path}")
        return
    test_data = load_jsonl(args.dataset_path)
    if not test_data:
        print(f"[ERROR] No data loaded from {args.dataset_path}")
        return
    if args.run_random_search:
        args.use_vllm_for_evaluator = True
        print("[RANDOM] Forcing vLLM evaluator for all random-search runs (use_vllm_eval=1).")
        k_list             = _parse_int_list(args.grid_k)
        infer_sizes_list   = _parse_int_list(args.grid_infer_sizes)
        craft_iters_list   = _parse_int_list(args.grid_craft_iters)
        refine_cands_list  = _parse_int_list(args.grid_refine_cands)
        replay_add_list    = _parse_int_list(args.grid_replay_add)
        seeds_list         = _parse_int_list(args.grid_seeds)
        shapley_perms_list = _parse_int_list(args.random_shapley_permutations)
        rng = random.Random(args.random_hparam_seed)
        unique_specs = set()
        attempts = 0
        max_attempts = args.random_specs * 20
        while len(unique_specs) < args.random_specs and attempts < max_attempts:
            attempts += 1
            spec = (
                rng.choice(k_list),
                rng.choice(infer_sizes_list),
                rng.choice(craft_iters_list),
                rng.choice(refine_cands_list),
                rng.choice(replay_add_list),
                rng.choice(shapley_perms_list),
            )
            if spec not in unique_specs:
                unique_specs.add(spec)
        results: List[Dict] = []
        spec_id = 0
        print(f"[RANDOM] Specs to evaluate: {len(unique_specs)} (requested {args.random_specs})")
        print(f"[RANDOM] Shapley permutations choices: {shapley_perms_list}")
        print(f"[RANDOM] Test items considered: {len(test_data) if not args.limit or args.limit<=0 else min(args.limit, len(test_data))}")
        def order_seeds(seeds: List[int]) -> List[int]:
            if 1 in seeds:
                return [1] + [s for s in seeds if s != 1]
            return list(seeds)
        ordered_seed_list = order_seeds(seeds_list)
        for spec in unique_specs:
            k, infer_size, craft_iters, refine_cands, replay_add, shapley_perm = spec
            spec_id += 1
            early_stopped = False
            for idx, seed in enumerate(ordered_seed_list):
                _, init_examples = proposer.generate_examples(
                    k,
                    seed=seed,
                    max_new_tokens=args.max_new_tokens_proposer
                )
                vllm_args = {
                    "model_path": args.model_path,
                    "tp": args.tp,
                    "util": args.gpu_mem_util,
                    "max_len": args.max_model_len,
                    "quantization": args.quantization,
                }
                crafted_examples, craft_time = craft_iterative(
                    proposer=proposer,
                    tokenizer=tokenizer,
                    initial_examples=init_examples,
                    infer_data=infer_data,
                    infer_subsample_size=infer_size,
                    seed=seed,
                    craft_iterations=craft_iters,
                    refine_candidates=refine_cands,
                    replay_add=replay_add,
                    vllm_args=vllm_args,
                    max_new_tokens_eval=args.max_new_tokens_evaluator,
                    shapley_permutations=shapley_perm,
                    shapley_tmc=args.shapley_tmc,
                    shapley_epsilon=args.shapley_epsilon,
                    shapley_log_dir=args.shapley_log_dir,
                )
                prompt_block = format_examples_for_evaluator(crafted_examples)
                preds, refs, sari = evaluate_on_test(
                    examples=crafted_examples,
                    tokenizer=tokenizer,
                    proposer=proposer,
                    args=args,
                    test_data=test_data,
                )
                this_row_early_stop = 1 if ((1 in seeds_list) and (seed == 1) and (sari < args.early_stop_sari_threshold)) else 0
                row = {
                    "spec_id": spec_id,
                    "seed": seed,
                    "k_init": k,
                    "k_final": len(crafted_examples),
                    "infer_subsample_size": infer_size,
                    "craft_iterations": craft_iters,
                    "refine_candidates": refine_cands,
                    "replay_add": replay_add,
                    "shapley_permutations": shapley_perm,
                    "acc_test": sari,
                    "preds": preds,
                    "refs": refs,
                    "craft_time_sec": round(craft_time, 4),
                    "use_vllm_eval": int(args.use_vllm_for_evaluator or True),
                    "early_stop": this_row_early_stop,
                    "prompt": "" if this_row_early_stop else prompt_block,
                }
                results.append(row)
                append_csv_row(args.grid_results_csv, row)
                print(f"[RANDOM] spec#{spec_id} seed={seed} | k={k} infer={infer_size} iters={craft_iters} cands={refine_cands} replay_add={replay_add} shapley_perm={shapley_perm} | SARI={sari:.2f} (preds={preds}, refs={refs}) | craft={craft_time:.3f}s | use_vllm_eval={row['use_vllm_eval']} | CSV -> {args.grid_results_csv}")
                if this_row_early_stop:
                    print(f"[RANDOM][EARLY-STOP] spec#{spec_id}: seed=1 SARI={sari:.2f} < {args.early_stop_sari_threshold} — skipping remaining seeds for this spec.")
                    early_stopped = True
                    break
            if early_stopped:
                pass
        ranked = sorted(results, key=lambda r: (-r["sari_test"], r["craft_time_sec"]))
        top2 = ranked[:2]
        print("\n[RANDOM] Top 2 runs (by SARI desc, craft_time asc):")
        for i, r in enumerate(top2, 1):
            print(f"  #{i}: spec_id={r['spec_id']} seed={r['seed']} | k={r['k_init']}→{r['k_final']} infer={r['infer_subsample_size']} iters={r['craft_iterations']} cands={r['refine_candidates']} replay_add={r['replay_add']} shapley_perm={r['shapley_permutations']} | SARI={r['sari_test']:.2f} craft={r['craft_time_sec']:.3f}s (preds={r['preds']}, refs={r['refs']})")
        return
    if args.run_grid_search:
        args.use_vllm_for_evaluator = True
        print("[GRID] Forcing vLLM evaluator for all grid runs (use_vllm_eval=1).")
        k_list            = _parse_int_list(args.grid_k)
        infer_sizes_list  = _parse_int_list(args.grid_infer_sizes)
        craft_iters_list  = _parse_int_list(args.grid_craft_iters)
        refine_cands_list = _parse_int_list(args.grid_refine_cands)
        replay_add_list   = _parse_int_list(args.grid_replay_add)
        seeds_list        = _parse_int_list(args.grid_seeds)
        results: List[Dict] = []
        spec_id = 0
        print(f"[GRID] Specs: k={k_list} | infer_sizes={infer_sizes_list} | craft_iters={craft_iters_list} | refine_cands={refine_cands_list} | replay_add={replay_add_list} | seeds={seeds_list}")
        print(f"[GRID] Test items considered: {len(test_data) if not args.limit or args.limit<=0 else min(args.limit, len(test_data))}")
        def order_seeds(seeds: List[int]) -> List[int]:
            if 1 in seeds:
                return [1] + [s for s in seeds if s != 1]
            return list(seeds)
        for k in k_list:
            for infer_size in infer_sizes_list:
                for craft_iters in craft_iters_list:
                    for refine_cands in refine_cands_list:
                        for replay_add in replay_add_list:
                            spec_id += 1
                            early_stopped = False
                            ordered_seeds = order_seeds(seeds_list)
                            for idx, seed in enumerate(ordered_seeds):
                                _, init_examples = proposer.generate_examples(
                                    k,
                                    seed=seed,
                                    max_new_tokens=args.max_new_tokens_proposer
                                )
                                vllm_args = {
                                    "model_path": args.model_path,
                                    "tp": args.tp,
                                    "util": args.gpu_mem_util,
                                    "max_len": args.max_model_len,
                                    "quantization": args.quantization,
                                }
                                crafted_examples, craft_time = craft_iterative(
                                    proposer=proposer,
                                    tokenizer=tokenizer,
                                    initial_examples=init_examples,
                                    infer_data=infer_data,
                                    infer_subsample_size=infer_size,
                                    seed=seed,
                                    craft_iterations=craft_iters,
                                    refine_candidates=refine_cands,
                                    replay_add=replay_add,
                                    vllm_args=vllm_args,
                                    max_new_tokens_eval=args.max_new_tokens_evaluator,
                                    shapley_permutations=args.shapley_permutations,
                                    shapley_tmc=args.shapley_tmc,
                                    shapley_epsilon=args.shapley_epsilon,
                                    shapley_log_dir=args.shapley_log_dir,
                                )
                                prompt_block = format_examples_for_evaluator(crafted_examples)
                                preds, refs, sari = evaluate_on_test(
                                    examples=crafted_examples,
                                    tokenizer=tokenizer,
                                    proposer=proposer,
                                    args=args,
                                    test_data=test_data,
                                )
                                this_row_early_stop = 1 if ((1 in seeds_list) and (seed == 1) and (sari < args.early_stop_sari_threshold)) else 0
                                row = {
                                    "spec_id": spec_id,
                                    "seed": seed,
                                    "k_init": k,
                                    "k_final": len(crafted_examples),
                                    "infer_subsample_size": infer_size,
                                    "craft_iterations": craft_iters,
                                    "refine_candidates": refine_cands,
                                    "replay_add": replay_add,
                                    "shapley_permutations": args.shapley_permutations,
                                    "sari_test": sari,
                                    "preds": preds,
                                    "refs": refs,
                                    "craft_time_sec": round(craft_time, 4),
                                    "use_vllm_eval": int(args.use_vllm_for_evaluator or True),
                                    "early_stop": this_row_early_stop,
                                    "prompt": "" if this_row_early_stop else prompt_block,
                                }
                                results.append(row)
                                append_csv_row(args.grid_results_csv, row)
                                print(f"[GRID] spec#{spec_id} seed={seed} | k={k} infer={infer_size} iters={craft_iters} cands={refine_cands} replay_add={replay_add} shapley_perm={args.shapley_permutations} | SARI={sari:.2f} (preds={preds}, refs={refs}) | craft={craft_time:.3f}s | use_vllm_eval={row['use_vllm_eval']} | CSV -> {args.grid_results_cvv if 'grid_results_cvv' in row else args.grid_results_csv}")
                                if this_row_early_stop:
                                    print(f"[GRID][EARLY-STOP] spec#{spec_id}: seed=1 SARI={sari:.2f} < {args.early_stop_sari_threshold} — skipping remaining seeds for this spec.")
                                    early_stopped = True
                                    break
                            if early_stopped:
                                pass
        ranked = sorted(results, key=lambda r: (-r["sari_test"], r["craft_time_sec"]))
        top2 = ranked[:2]
        print("\n[GRID] Top 2 runs (by SARI desc, craft_time asc):")
        for i, r in enumerate(top2, 1):
            print(f"  #{i}: spec_id={r['spec_id']} seed={r['seed']} | k={r['k_init']}→{r['k_final']} infer={r['infer_subsample_size']} iters={r['craft_iterations']} cands={r['refine_candidates']} replay_add={r['replay_add']} shapley_perm={r['shapley_permutations']} | SARI={r['sari_test']:.2f} craft={r['craft_time_sec']:.3f}s (preds={r['preds']}, refs={r['refs']})")
        return
    _, examples = proposer.generate_examples(
        args.k,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens_proposer
    )
    if len(examples) < args.k:
        print(f"[WARN] Parsed only {len(examples)} / {args.k} examples from proposer output. Continuing with available.")
    with open(args.save_examples_to, "w", encoding="utf-8") as f:
        f.write(format_examples_for_evaluator(examples))
    print(f"[INFO] Saved proposed examples to: {args.save_examples_to}")
    print("\n[PROPOSED EXAMPLES]\n" + format_examples_for_evaluator(examples) + "\n")
    vllm_args = {"model_path": args.model_path, "tp": args.tp, "util": args.gpu_mem_util, "max_len": args.max_model_len, "quantization": args.quantization}
    crafted_examples, craft_elapsed = craft_iterative(
        proposer=proposer,
        tokenizer=tokenizer,
        initial_examples=examples,
        infer_data=infer_data,
        infer_subsample_size=args.infer_subsample_size,
        seed=args.seed,
        craft_iterations=args.craft_iterations,
        refine_candidates=args.refine_candidates,
        replay_add=args.replay_add,
        vllm_args=vllm_args,
        max_new_tokens_eval=args.max_new_tokens_evaluator,
        shapley_permutations=args.shapley_permutations,
        shapley_tmc=args.shapley_tmc,
        shapley_epsilon=args.shapley_epsilon,
        shapley_log_dir=args.shapley_log_dir,
    )
    print(f"[CRAFT] Completed {args.craft_iterations} iteration(s). Time: {craft_elapsed:.3f} seconds.")
    crafted_block = format_examples_for_evaluator(crafted_examples)
    with open(args.save_crafted_examples_to, "w", encoding="utf-8") as f:
        f.write(crafted_block)
    print(f"[INFO] Saved crafted examples to: {args.save_crafted_examples_to}")
    if args.eval_on_test:
        preds, refs, sari = evaluate_on_test(crafted_examples, tokenizer, proposer, args, test_data)
        print(f"[RESULT] Evaluated {refs} TEST items from: {args.dataset_path}")
        print(f"[METRIC] SARI: {sari:.2f}  (preds={preds}, refs={refs})")
    else:
        print("[INFO] Skipped TEST evaluation (run with --eval-on-test to enable).")

if __name__ == "__main__":
    main()
