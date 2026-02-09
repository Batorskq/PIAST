import argparse
import os
import re
import random
import time
import csv
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from fastprompt_utils import *
from fastprompt_utils import (
    _translate_cuda_visible_devices, _sentence_count_plan, append_csv_row,
    format_examples_for_evaluator, load_jsonl, subsample_items,
    dedup_items_by_sentence, sample_replay, _ensure_dir, _safe_std, _now_tag, _parse_int_list
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EARLY_STOP_THRESHOLD = 0.83

REFINEMENT_TOPICS = [
    "geopolitics/diplomacy","elections/policy","conflict/security","public health",
    "disasters/climate","courts/law","trade/supply chains","markets/stocks",
    "earnings/guidance","mergers & acquisitions","inflation/rates","employment/labor",
    "startups/funding","AI/software","cybersecurity/data breaches","consumer tech/gadgets",
    "telecom/internet","space/aviation","automotive/EV","energy/oil & gas",
    "match result/recap","transfers/trades","injuries/lineups","records/milestones"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=12, help="Number of in-context examples to propose.")
    p.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Local path or HF id to qwen-2.5-7b-instruct.")
    p.add_argument("--eval-on-test", action="store_true", help="If set, run final evaluation on the TEST set at --dataset-path.")
    p.add_argument("--dataset-path", type=str, default="dataset/news_test.jsonl", help="Path to news_test.jsonl.")
    p.add_argument("--infer-dataset-path", type=str, default="dataset/news_infer.jsonl", help="Path to news_infer.jsonl.")
    p.add_argument("--infer-subsample-size", type=int, default=30, help="Subsample from news_infer per crafting iteration.")
    p.add_argument("--seed", type=int, default=123, help="Random seed base for subsampling and proposer.")
    p.add_argument("--refine-candidates", type=int, default=3, help="How many candidate replacements to try.")
    p.add_argument("--craft-iterations", type=int, default=10, help="Number of craft-refine/drop iterations.")
    p.add_argument("--replay-add", type=int, default=5, help="How many items to add to replay per iteration.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of TEST items (0 = all).")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    p.add_argument("--gpu-mem-util", type=float, default=0.92, help="vLLM GPU memory utilization.")
    p.add_argument("--max-model-len", type=int, default=2048, help="Max model length for vLLM.")
    p.add_argument("--quantization", type=str, default=None, help="Optional vLLM quantization.")
    p.add_argument("--max-new-tokens-proposer", type=int, default=0, help="Override proposer max tokens. Default = 48*k (or 64 min).")
    p.add_argument("--max-new-tokens-evaluator", type=int, default=4, help="Max new tokens for evaluator prediction.")
    p.add_argument("--use-vllm-for-evaluator", action="store_true", help="Use vLLM for FINAL evaluator instead of Transformers.")
    p.add_argument("--batch-size", type=int, default=1, help="(Transformers) batch size for evaluator.")
    p.add_argument("--save-examples-to", type=str, default="news_proposed_examples.txt", help="Where to save initial generated examples.")
    p.add_argument("--save-crafted-examples-to", type=str, default="news_crafted_examples.txt", help="Where to save crafted examples.")
    p.add_argument("--run-grid-search", action="store_true", help="Run grid search over many specs and seeds.")
    p.add_argument("--run-random-search", action="store_true", help="Run random search over hyperparameters and seeds.")
    p.add_argument("--grid-k", type=str, default="8,12,18,24", help="Comma-separated k values.")
    p.add_argument("--grid-infer-sizes", type=str, default="20,30,50,70", help="Comma-separated infer subsample sizes.")
    p.add_argument("--grid-craft-iters", type=str, default="1,3,5,8,10", help="Comma-separated craft iteration counts.")
    p.add_argument("--grid-refine-cands", type=str, default="1,3,5,10", help="Comma-separated candidate counts.")
    p.add_argument("--grid-replay-add", type=str, default="0,5,15", help="Comma-separated replay-add values.")
    p.add_argument("--grid-seeds", type=str, default="1,2,3,4", help="Comma-separated seeds.")
    p.add_argument("--grid-results-csv", type=str, default="results/news_initial.csv", help="Where to save grid/random search results CSV.")
    p.add_argument("--random-specs", type=int, default=1000, help="Number of random hyperparameter specs to evaluate.")
    p.add_argument("--random-hparam-seed", type=int, default=12345, help="RNG seed for random spec sampling.")
    p.add_argument("--shapley-permutations", type=int, default=1, help="Number of random permutations for Monte-Carlo Shapley.")
    p.add_argument("--shapley-tmc", action="store_true", help="Enable Truncated Monte Carlo (stop permutation when acc≈full).")
    p.add_argument("--shapley-epsilon", type=float, default=0.0, help="Tolerance for TMC stop; 0.0 = only stop when equal to full acc.")
    p.add_argument("--shapley-log-dir", type=str, default="results_shapley_news", help="Directory to save per-iteration Shapley CSV logs.")
    p.add_argument("--random-shapley-permutations", type=str, default="1,2,3,5,10",
                   help="Comma-separated values for Shapley permutations to sample in random search.")
    return p.parse_args()

NEWS_LABELS = ["World", "Sports", "Business", "Tech"]

def _balanced_label_allocation(k: int, labels: List[str]) -> Dict[str, int]:
    base = k // len(labels)
    rem = k % len(labels)
    alloc = {lab: base for lab in labels}
    for i in range(rem):
        alloc[labels[i]] += 1
    return alloc

def _label_style_rules(label: str) -> str:
    if label == "World":
        return (
            "- Label MUST be exactly: World.\n"
            "- Avoid sports/business/consumer-tech framing.\n"
        )
    if label == "Sports":
        return (
            "- Label MUST be exactly: Sports.\n"
            "- Avoid macroeconomics or international diplomacy framing.\n"
        )
    if label == "Business":
        return (
            "- Label MUST be exactly: Business.\n"
            "- Avoid consumer-tech product specs unless framed as market/business impact.\n"
        )
    return (
        "- Label MUST be exactly: Tech.\n"
        "- Avoid market-centric or global politics framing unless primarily about technology.\n"
    )

def _build_label_specific_messages(
    num_examples: int,
    label: str,
    counts: List[int],
    topics: List[str],
) -> List[Dict[str, str]]:
    assert label in NEWS_LABELS
    plan_lines = []
    for i in range(num_examples):
        n = counts[i]
        plan_lines.append(f"- Example{i+1}: write exactly {n} sentence{'s' if n>1 else ''}; topic: {topics[i % len(topics)]}.")
    diversity_plan = "\n".join(plan_lines)

    sys_msg = (
        "You are a data generator that writes high-quality in-context learning examples "
        "for 4-way news topic classification."
    )
    style_rules = _label_style_rules(label)

    user_msg = f"""Create exactly {num_examples} training examples in THIS STRICT format only:

Example1:
Sentence: "<text>"
Label: {label}

Example2:
Sentence: "<text>"
Label: {label}

...
Example{num_examples}:
Sentence: "<text>"
Label: {label}

Diversity plan (MUST FOLLOW):
{diversity_plan}

Rules:
- Each example's "Sentence" must contain exactly the number of sentences specified above (1–3).
- Keep sentences concise: typically 5–22 words each. Include at least one very short (≤7 words) and one longer (16–22 words) across the set.
- Use only ASCII characters. Do NOT include double quotes inside the text.
- Use exactly ONE 'Sentence:' line per example; if multiple sentences are needed, put them inside the same quotes separated by a space.
- Make the topic unmistakable for the given label by using clear cues (e.g., 'quarterly earnings' for Business, 'semiconductor launch' for Tech, 'quarterfinal win' for Sports, 'summit talks' for World).
{style_rules}
- Do NOT wrap output in Markdown/code fences.
- Output ONLY the examples in the exact format above; no extra text.
"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

EXAMPLE_HEADER_RE = re.compile(r"^Example\s*(\d+)\s*:\s*", re.IGNORECASE | re.MULTILINE)
SENTENCE_LINE_RE = re.compile(r'^\s*Sentence\s*:\s*"(.*?)"\s*$', re.IGNORECASE | re.MULTILINE | re.DOTALL)
LABEL_RE = re.compile(r'^\s*Label\s*:\s*(World|Sports|Business|Tech)\b', re.IGNORECASE | re.MULTILINE)

def parse_examples(text: str, k: int) -> List[Dict[str, str]]:
    blocks = []
    headers = list(EXAMPLE_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        blocks.append((int(m.group(1)), text[start:end]))

    tuples = []
    for idx, block in blocks:
        sents = [m.group(1).strip() for m in SENTENCE_LINE_RE.finditer(block)]
        if not sents:
            continue
        sentence = " ".join(sents)
        m_lab = LABEL_RE.search(block)
        if not m_lab:
            continue
        lab = m_lab.group(1).capitalize().strip()
        if lab not in NEWS_LABELS:
            continue
        tuples.append((idx, sentence, lab))

    tuples.sort(key=lambda x: x[0])
    dedup = []
    seen = set()
    for _, sent, lab in tuples:
        key = (sent, lab)
        if key in seen:
            continue
        seen.add(key)
        dedup.append({"sentence": sent, "label": lab})
        if len(dedup) >= k:
            break
    return dedup

class VLLMProposer:
    def __init__(self, model_path: str, tokenizer, tp: int, util: float, max_len: int, quantization: Optional[str]):
        from vllm import LLM, SamplingParams   
        self.tokenizer = tokenizer
        self.SamplingParams = SamplingParams
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

    def generate_examples_balanced(self, k: int, *, seed: Optional[int] = None, max_new_tokens: Optional[int] = None) -> Tuple[str, List[Dict[str, str]]]:
        alloc = _balanced_label_allocation(k, NEWS_LABELS)
        sent_counts = _sentence_count_plan(k)

        idx = 0
        per_label_counts: Dict[str, List[int]] = {}
        for lab in NEWS_LABELS:
            n_lab = alloc[lab]
            per_label_counts[lab] = []
            for _ in range(n_lab):
                per_label_counts[lab].append(sent_counts[idx % len(sent_counts)])
                idx += 1

        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = max(64, 48 * max(alloc.values()) if alloc else 64)

        all_parsed: List[Dict[str, str]] = []
        label_seed_bases = {lab: (0 if seed is None else seed * 97 + i * 7) for i, lab in enumerate(NEWS_LABELS)}
        for lab in NEWS_LABELS:
            n_lab = alloc[lab]
            if n_lab <= 0:
                continue
            msgs = _build_label_specific_messages(n_lab, lab, per_label_counts[lab], REFINEMENT_TOPICS)
            txt = self._gen_with_messages(msgs, max_new_tokens, None if seed is None else label_seed_bases[lab] + 0)
            parsed = [ex for ex in parse_examples(txt, n_lab) if ex["label"] == lab]
            all_parsed.extend(parsed)

        produced_counts = {lab: 0 for lab in NEWS_LABELS}
        for ex in all_parsed:
            produced_counts[ex["label"]] += 1
        for lab in NEWS_LABELS:
            need = alloc[lab] - produced_counts[lab]
            if need > 0:
                msgs = _build_label_specific_messages(need, lab, per_label_counts[lab][:need], REFINEMENT_TOPICS)
                txt = self._gen_with_messages(msgs, max_new_tokens, None if seed is None else label_seed_bases[lab] + 1)
                all_parsed.extend([ex for ex in parse_examples(txt, need) if ex["label"] == lab])

        rng = random.Random(None if seed is None else seed + 99991)
        seen = set()
        dedup = []
        for ex in all_parsed:
            s = ex["sentence"]
            if s in seen:
                continue
            seen.add(s)
            dedup.append(ex)
            if len(dedup) >= k:
                break
        rng.shuffle(dedup)

        out_text = format_examples_for_evaluator(dedup)
        return out_text, dedup


EVAL_TASK_HEADER = (
    "Please perform News Classification task. Given the news item, assign a label from "
    "['World', 'Sports', 'Business', 'Tech']. Return label only without any other text."
)

def build_evaluator_user_prompt(examples_block: str, sentence: str) -> str:
    return f"{EVAL_TASK_HEADER}\n\n{examples_block}\n\nSentence: \"{sentence}\"\nLabel:"

def normalize_label(text: str) -> Optional[str]:
    t = text.strip().lower()

    if "world" in t: return "World"
    if "sport" in t: return "Sports" 
    if "business" in t or "econom" in t or "market" in t: return "Business"
    if "tech" in t or "technology" in t or "software" in t or "cyber" in t: return "Tech"

    t0 = re.sub(r"[^0-9a-z]+", "", t)
    if "1" in t or t0 == "1": return "World"
    if "2" in t or t0 == "2": return "Sports"
    if "3" in t or t0 == "3": return "Business"
    if "4" in t or t0 == "4": return "Tech"

    return None

class TransformersEvaluator:
    def __init__(self, model_path: str, max_new_tokens: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.max_new = max_new_tokens
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict_label(self, examples_block: str, sentence: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": build_evaluator_user_prompt(examples_block, sentence)},
        ]
        enc = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        enc = enc.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                enc,
                max_new_tokens=self.max_new,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )
        gen_only = out[0, enc.shape[1]:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        return normalize_label(text)

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

    def predict_label(self, examples_block: str, sentence: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": build_evaluator_user_prompt(examples_block, sentence)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self.llm.generate([prompt_text], self.sp, use_tqdm=False)[0].outputs[0].text
        return normalize_label(out)

def eval_accuracy_on_items(evaluator, examples_block: str, items: List[Dict]) -> Tuple[int, int, float]:
    sentences: List[str] = []
    gold: List[str] = []
    for obj in items:
        solution = str(obj.get("solution", "")).strip()
        solution_norm = solution.capitalize()
        if solution_norm not in NEWS_LABELS:
            continue
        msgs = obj.get("messages", [])
        user_turn = next((m for m in msgs if m.get("role") == "user"), None)
        if not user_turn:
            continue
        sentence = str(user_turn.get("content", "")).strip()
        if not sentence:
            continue
        sentences.append(sentence)
        gold.append(solution_norm)

    total = len(sentences)
    if total == 0:
        return 0, 0, 0.0

    correct = 0

    if isinstance(evaluator, VLLMEvaluator):
        bs = 128
        for i in range(0, total, bs):
            batch_sents = sentences[i:i+bs]
            prompts = []
            for s in batch_sents:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": build_evaluator_user_prompt(examples_block, s)},
                ]
                prompts.append(evaluator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            outs = evaluator.llm.generate(prompts, evaluator.sp, use_tqdm=False)
            preds = [normalize_label(o.outputs[0].text) for o in outs]
            for p, g in zip(preds, gold[i:i+bs]):
                if p == g:
                    correct += 1
        acc = correct / total
        return correct, total, acc

    if isinstance(evaluator, TransformersEvaluator):
        bs = 128
        tok = evaluator.tokenizer
        mdl = evaluator.model
        device = mdl.device
        pad_id = tok.pad_token_id or tok.eos_token_id
        eos_id = tok.eos_token_id

        for i in range(0, total, bs):
            batch_sents = sentences[i:i+bs]
            prompts = []
            for s in batch_sents:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": build_evaluator_user_prompt(examples_block, s)},
                ]
                prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            input_lengths = enc["attention_mask"].sum(dim=1)

            with torch.no_grad():
                out = mdl.generate(
                    **enc,
                    max_new_tokens=evaluator.max_new,
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
                p = normalize_label(text)
                if p == gold[i + j]:
                    correct += 1

        acc = correct / total
        return correct, total, acc

    for s, g in zip(sentences, gold):
        p = evaluator.predict_label(examples_block, s)
        if p == g:
            correct += 1
    acc = correct / total
    return correct, total, acc

def label_counts(examples: List[Dict[str, str]]) -> Dict[str, int]:
    c = {lab: 0 for lab in NEWS_LABELS}
    for ex in examples:
        lab = ex.get("label")
        if lab in c:
            c[lab] += 1
    return c

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
        c, t, a = eval_accuracy_on_items(evaluator, block, infer_items)
        value_cache[key] = (c, t, a)
        return c, t, a

    _, _, v_empty = value_of(tuple())
    _, _, v_full  = value_of(tuple(range(n)))

    print(f"[SHAPLEY] iter={iter_index} seed={run_seed} | K={permutations} | TMC={int(use_tmc)} eps={epsilon:.4g} | "
          f"v(empty)={v_empty:.4f} v(full)={v_full:.4f}")

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

    phi = [ (sum(cs)/len(cs) if len(cs)>0 else 0.0) for cs in contribs ]
    std = [ _safe_std(cs) for cs in contribs ]
    samples = [ len(cs) for cs in contribs ]

    csv_path = os.path.join(
        log_dir,
        f"shapley_iter{iter_index}_seed{run_seed}_{_now_tag()}.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "seed", "index", "label", "sentence", "phi", "std", "samples",
                    "v_empty", "v_full", "permutations", "tmc", "epsilon", "cache_hits", "cache_misses"])
        for i, ex in enumerate(examples):
            w.writerow([
                iter_index, run_seed, i, ex.get("label",""), ex.get("sentence",""),
                f"{phi[i]:.6f}", f"{std[i]:.6f}", samples[i],
                f"{v_empty:.6f}", f"{v_full:.6f}", permutations, int(use_tmc), f"{epsilon:.6f}",
                cache_hits, cache_misses
            ])

    print(f"[SHAPLEY] iter={iter_index} saved CSV -> {csv_path}")
    worst_idx = min(range(n), key=lambda i: phi[i])
    print(f"[SHAPLEY] iter={iter_index} worst_index={worst_idx} phi={phi[worst_idx]:.6f} "
          f"(cache hits={cache_hits}, misses={cache_misses})")

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
    counts = label_counts(remaining_examples)
    minority = min(NEWS_LABELS, key=lambda lab: counts.get(lab, 0))
    plan_counts = [1 + (i % 3) for i in range(num_candidates)]
    plan_lines = []
    for i in range(num_candidates):
        n = plan_counts[i]
        topic = REFINEMENT_TOPICS[i % len(REFINEMENT_TOPICS)]
        plan_lines.append(f"- Example{i+1}: write exactly {n} sentence{'s' if n>1 else ''}; topic: {topic}.")
    diversity_plan = "\n".join(plan_lines)

    sys_msg = ("You are improving in-context examples for 4-way news topic classification. "
               "Generate replacements that diversify length (1–3 sentences) and topic, avoid paraphrasing, and help the task.")
    user_msg = f"""You are given the CURRENT examples (do not repeat or paraphrase them):

{rem_block}

Now create exactly {num_candidates} NEW examples in THIS STRICT format:

Example1:
Sentence: "<text>"
Label: World|Sports|Business|Tech

Example2:
Sentence: "<text>"
Label: World|Sports|Business|Tech

...
Example{num_candidates}:
Sentence: "<text>"
Label: World|Sports|Business|Tech

Diversity plan (MUST FOLLOW):
{diversity_plan}

Rules:
- Use exactly ONE 'Sentence:' line per example. If multiple sentences are needed, put them INSIDE the same quotes separated by a space.
- Each example must have exactly the number of sentences specified in the plan above (1–3).
- Keep sentences concise: typically 5–22 words each. Across the set, include very short (≤7 words) and longer (16–22 words).
- ASCII only. Do NOT include double quotes inside the text.
- Make topics clearly different from the given examples and from each other; avoid near-duplicates or paraphrases.
- Prefer balancing labels; if unsure, choose the minority label: {minority}.
- Make the topical cues explicit for the selected label.
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
    base_correct, base_total, base_acc = eval_accuracy_on_items(fast_eval, block_all, infer_items)

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
    drop_correct, drop_total, drop_acc = eval_accuracy_on_items(fast_eval, block_drop_i, infer_items)

    remaining = [ex for j, ex in enumerate(base_examples) if j != worst_idx]
    ref_messages = build_refinement_messages(remaining, num_candidates)
    ref_prompt = tokenizer.apply_chat_template(ref_messages, tokenize=False, add_generation_prompt=True)

    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max(60, 36 * num_candidates),
        stop=["\n\n\n"],
        seed=int(run_seed + 4242),
    )
    ref_text = proposer_llm.generate([ref_prompt], sp, use_tqdm=False)[0].outputs[0].text
    candidates = parse_examples(ref_text, num_candidates)

    cand_results = []
    best_cand_acc = float("-inf")
    best_cand = None
    for cand in candidates:
        trial_examples = list(base_examples)
        trial_examples[worst_idx] = cand
        block_trial = format_examples_for_evaluator(trial_examples, drop_index=None)
        c, t, a = eval_accuracy_on_items(fast_eval, block_trial, infer_items)
        cand_results.append({"candidate": cand, "correct": c, "total": t, "acc": a})
        if a > best_cand_acc:
            best_cand_acc = a
            best_cand = cand

    refined_acc = best_cand_acc if best_cand is not None else float("-inf")

    decision = "keep"
    final_examples = base_examples
    if best_cand is not None and refined_acc >= drop_acc and refined_acc >= base_acc:
        decision = "replace"
        final_examples = list(base_examples)
        final_examples[worst_idx] = best_cand
    elif drop_acc >= base_acc and len(base_examples) > 1:
        decision = "drop"
        final_examples = [ex for j, ex in enumerate(base_examples) if j != worst_idx]
    else:
        decision = "keep"

    final_acc = {"keep": base_acc, "drop": drop_acc, "replace": refined_acc}[decision]

    summary = {
        "baseline": {"correct": base_correct, "total": base_total, "acc": base_acc},
        "shapley": shapley_info, 
        "chosen_worst_index": worst_idx,
        "drop_stats": {"correct": drop_correct, "total": drop_total, "acc": drop_acc},
        "ref_text_raw": ref_text,
        "candidate_results": cand_results,
        "decision": decision,
        "final_k": len(final_examples),
        "best_candidate_acc": None if best_cand is None else refined_acc,
        "final_acc": final_acc,
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
        evaluator = TransformersEvaluator(model_path=args.model_path, max_new_tokens=args.max_new_tokens_evaluator)
    data = test_data if not args.limit or args.limit <= 0 else test_data[:args.limit]
    total = 0
    correct = 0
    for obj in tqdm(data):
        solution = str(obj.get("solution", "")).strip()
        solution_norm = solution.capitalize()
        msgs = obj.get("messages", [])
        user_turn = next((m for m in msgs if m.get("role") == "user"), None)
        if not user_turn:
            continue
        sentence = str(user_turn.get("content", "")).strip()
        if not sentence:
            continue
        pred = evaluator.predict_label(examples_block, sentence)
        total += 1
        if pred == solution_norm:
            correct += 1
    acc = (correct / total) if total else 0.0
    return correct, total, acc

def main():
    _translate_cuda_visible_devices()
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

            for seed in ordered_seed_list:
                _, init_examples = proposer.generate_examples_balanced(
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

                correct, total, acc = evaluate_on_test(
                    examples=crafted_examples,
                    tokenizer=tokenizer,
                    proposer=proposer,
                    args=args,
                    test_data=test_data,
                )

                this_row_early_stop = 1 if ((1 in seeds_list) and (seed == 1) and (acc < EARLY_STOP_THRESHOLD)) else 0

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
                    "acc_test": acc,
                    "correct": correct,
                    "total": total,
                    "craft_time_sec": round(craft_time, 4),
                    "use_vllm_eval": int(args.use_vllm_for_evaluator),
                    "early_stop": this_row_early_stop,
                    "prompt": "" if this_row_early_stop else prompt_block,
                }
                results.append(row)
                append_csv_row(args.grid_results_csv, row)

                print(f"[RANDOM] spec#{spec_id} seed={seed} | k={k} infer={infer_size} iters={craft_iters} "
                      f"cands={refine_cands} replay_add={replay_add} shapley_perm={shapley_perm} | "
                      f"acc={acc:.4f} (correct={correct}/{total}) | craft={craft_time:.3f}s "
                      f"| use_vllm_eval={row['use_vllm_eval']} | CSV -> {args.grid_results_csv}")

                if this_row_early_stop:
                    print(f"[RANDOM][EARLY-STOP] spec#{spec_id}: seed=1 acc={acc:.4f} < {EARLY_STOP_THRESHOLD:.2f} — skipping remaining seeds for this spec.")
                    early_stopped = True
                    break

            if early_stopped:
                pass

        ranked = sorted(results, key=lambda r: (-r["acc_test"], r["craft_time_sec"]))
        top2 = ranked[:2]
        print("\n[RANDOM] Top 2 runs (by acc desc, craft_time asc):")
        for i, r in enumerate(top2, 1):
            print(f"  #{i}: spec_id={r['spec_id']} seed={r['seed']} | "
                  f"k={r['k_init']}→{r['k_final']} infer={r['infer_subsample_size']} iters={r['craft_iterations']} "
                  f"cands={r['refine_candidates']} replay_add={r['replay_add']} shapley_perm={r['shapley_permutations']} | "
                  f"acc={r['acc_test']:.4f} craft={r['craft_time_sec']:.3f}s (correct={r['correct']}/{r['total']})")
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

        print(f"[GRID] Specs: k={k_list} | infer_sizes={infer_sizes_list} | craft_iters={craft_iters_list} | "
              f"refine_cands={refine_cands_list} | replay_add={replay_add_list} | seeds={seeds_list}")
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

                            for seed in ordered_seeds:
                                _, init_examples = proposer.generate_examples_balanced(
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

                                correct, total, acc = evaluate_on_test(
                                    examples=crafted_examples,
                                    tokenizer=tokenizer,
                                    proposer=proposer,
                                    args=args,
                                    test_data=test_data,
                                )

                                this_row_early_stop = 1 if ((1 in seeds_list) and (seed == 1) and (acc < EARLY_STOP_THRESHOLD)) else 0

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
                                    "acc_test": acc,
                                    "correct": correct,
                                    "total": total,
                                    "craft_time_sec": round(craft_time, 4),
                                    "use_vllm_eval": int(args.use_vllm_for_evaluator),
                                    "early_stop": this_row_early_stop,
                                    "prompt": "" if this_row_early_stop else prompt_block,
                                }
                                results.append(row)
                                append_csv_row(args.grid_results_csv, row)

                                print(f"[GRID] spec#{spec_id} seed={seed} | k={k} infer={infer_size} iters={craft_iters} "
                                      f"cands={refine_cands} replay_add={replay_add} shapley_perm={args.shapley_permutations} | "
                                      f"acc={acc:.4f} (correct={correct}/{total}) | craft={craft_time:.3f}s "
                                      f"| use_vllm_eval={row['use_vllm_eval']} | CSV -> {args.grid_results_csv}")

                                if this_row_early_stop:
                                    print(f"[GRID][EARLY-STOP] spec#{spec_id}: seed=1 acc={acc:.4f} < {EARLY_STOP_THRESHOLD:.2f} — skipping remaining seeds for this spec.")
                                    early_stopped = True
                                    break

                            if early_stopped:
                                pass

        ranked = sorted(results, key=lambda r: (-r["acc_test"], r["craft_time_sec"]))
        top2 = ranked[:2]
        print("\n[GRID] Top 2 runs (by acc desc, craft_time asc):")
        for i, r in enumerate(top2, 1):
            print(f"  #{i}: spec_id={r['spec_id']} seed={r['seed']} | "
                  f"k={r['k_init']}→{r['k_final']} infer={r['infer_subsample_size']} iters={r['craft_iterations']} "
                  f"cands={r['refine_candidates']} replay_add={r['replay_add']} shapley_perm={r['shapley_permutations']} | "
                  f"acc={r['acc_test']:.4f} craft={r['craft_time_sec']:.3f}s (correct={r['correct']}/{r['total']})")
        return

    _, examples = proposer.generate_examples_balanced(
        args.k,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens_proposer
    )
    if len(examples) < args.k:
        counts = label_counts(examples)
        print(f"[WARN] Parsed only {len(examples)} / {args.k} examples from proposer output. "
              f"({', '.join([f'{lab}={counts.get(lab,0)}' for lab in NEWS_LABELS])}) Continuing with available.")
    with open(args.save_examples_to, "w", encoding="utf-8") as f:
        f.write(format_examples_for_evaluator(examples))
    print(f"[INFO] Saved proposed examples to: {args.save_examples_to}")
    print("\n[PROPOSED EXAMPLES]\n" + format_examples_for_evaluator(examples) + "\n")

    vllm_args = {"model_path": args.model_path, "tp": args.tp, "util": args.gpu_mem_util,
                 "max_len": args.max_model_len, "quantization": args.quantization}
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
        correct, total, acc = evaluate_on_test(crafted_examples, tokenizer, proposer, args, test_data)
        print(f"[RESULT] Evaluated {total} TEST examples from: {args.dataset_path}")
        print(f"[METRIC] Accuracy: {acc:.4f}  (correct={correct}, total={total})")
    else:
        print("[INFO] Skipped TEST evaluation (run with --eval-on-test to enable).")

if __name__ == "__main__":
    main()
