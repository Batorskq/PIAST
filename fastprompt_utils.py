import subprocess
import sys
import json
from typing import List, Tuple, Dict, Optional
import random
import re
import os
import csv
import math 
import time
import argparse

_UUID_RE = re.compile(r"GPU-[0-9a-fA-F\-]+")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4, help="Number of in-context examples to propose (crafting starts at 8).")
    p.add_argument("--model-path", type=str, default="/scratch/hpc-prf-arcllm/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct", help="Local path to qwen-2.5-7b-instruct.")
    p.add_argument("--eval-on-test", action="store_true", help="If set, run final evaluation on the TEST set at --dataset-path.")
    p.add_argument("--dataset-path", type=str, default="dataset/subj_test.jsonl", help="Path to subj_test.jsonl.")
    p.add_argument("--infer-dataset-path", type=str, default="dataset/subj_infer.jsonl", help="Path to subj_infer.jsonl.")
    p.add_argument("--infer-subsample-size", type=int, default=50, help="Subsample from subj_infer per crafting iteration.")
    p.add_argument("--seed", type=int, default=123, help="Random seed base for subsampling.")
    p.add_argument("--refine-candidates", type=int, default=3, help="How many candidate replacements to try.")
    p.add_argument("--craft-iterations", type=int, default=2, help="Number of craft-refine/drop iterations.")
    p.add_argument("--replay-add", type=int, default=5, help="How many items to add to replay per iteration.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of TEST items (0 = all).")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    p.add_argument("--gpu-mem-util", type=float, default=0.92, help="vLLM GPU memory utilization.")
    p.add_argument("--max-model-len", type=int, default=2048, help="Max model length for vLLM.")
    p.add_argument("--quantization", type=str, default=None, help="Optional vLLM quantization.")
    p.add_argument("--max-new-tokens-proposer", type=int, default=0, help="Override proposer max tokens. Default = 48*k (or 64 min).")
    p.add_argument("--max-new-tokens-evaluator", type=int, default=4, help="Max new tokens for evaluator prediction.")
    p.add_argument("--use-vllm-for-evaluator", action="store_true", help="Use vLLM for FINAL evaluator instead of Transformers.")
    p.add_argument("--batch-size", type=int, default=8, help="(Transformers) batch size for evaluator.")
    p.add_argument("--save-examples-to", type=str, default="proposed_examples.txt", help="Where to save initial generated examples.")
    p.add_argument("--save-crafted-examples-to", type=str, default="crafted_examples.txt", help="Where to save crafted examples.")
    p.add_argument("--run-grid-search", action="store_true", help="Run grid search over many specs and seeds.")
    p.add_argument("--run-random-search", action="store_true", help="Run random search over hyperparameters and seeds.")
    p.add_argument("--grid-k", type=str, default="4,6,8,12,16", help="Comma-separated k values.")
    p.add_argument("--grid-infer-sizes", type=str, default="10,20,30,50,70,100", help="Comma-separated infer subsample sizes.")
    p.add_argument("--grid-craft-iters", type=str, default="1,3,5,8,10,15", help="Comma-separated craft iteration counts.")
    p.add_argument("--grid-refine-cands", type=str, default="1,3,5,10", help="Comma-separated candidate counts.")
    p.add_argument("--grid-replay-add", type=str, default="0,5,15", help="Comma-separated replay-add values.")
    p.add_argument("--grid-seeds", type=str, default="1,2,3,4", help="Comma-separated seeds.")
    p.add_argument("--grid-shapley-permutations", type=str, default="3", help="Comma-separated seeds.")
    p.add_argument("--grid-results-csv", type=str, default="results/subj_temp.csv", help="Where to save grid/random search results CSV.")
    p.add_argument("--random-specs", type=int, default=1000, help="Number of random hyperparameter specs to evaluate.")
    p.add_argument("--random-hparam-seed", type=int, default=12345, help="RNG seed for random spec sampling.")
    p.add_argument("--shapley-permutations", type=int, default=1, help="Number of random permutations for Monte-Carlo Shapley.")
    p.add_argument("--shapley-tmc", action="store_true", help="Enable Truncated Monte Carlo (stop permutation when acc≈full).")
    p.add_argument("--shapley-epsilon", type=float, default=0.0, help="Tolerance for TMC stop; 0.0 = only stop when equal to full acc.")
    p.add_argument("--shapley-log-dir", type=str, default="results_shapley", help="Directory to save per-iteration Shapley CSV logs.")
    p.add_argument("--random-shapley-permutations", type=str, default="1,2,3,5,10",
                   help="Comma-separated values for Shapley permutations to sample in random search.")
    return p.parse_args()

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_std(vals: List[float]) -> float:
    n = len(vals)
    if n <= 1:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(max(var, 0.0))

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def build_evaluator_user_prompt(examples_block: str, sentence: str, EVAL_TASK_HEADER:str) -> str:
    return f"{EVAL_TASK_HEADER}\n\n{examples_block}\n\nSentence: \"{sentence}\"\nLabel:"

def format_examples_for_evaluator(examples: List[Dict[str, str]], drop_index: Optional[int] = None) -> str:
    lines = []
    for i, ex in enumerate(examples, start=1):
        if drop_index is not None and (i - 1) == drop_index:
            continue
        lines.append(f'Example{len(lines)+1}:\nSentence: "{ex["sentence"]}"\nLabel: {ex["label"]}\n')
    return "\n".join(lines).strip()

CSV_FIELDNAMES = [
    "spec_id","seed","k_init","k_final","infer_subsample_size","craft_iterations",
    "refine_candidates","replay_add","acc_test","correct","total", "shapley_permutations",
    "craft_time_sec","use_vllm_eval","early_stop","prompt"
]

def _sentence_count_plan(k: int) -> List[int]:
    return [1 + (i * 3) // k for i in range(k)]

def _csv_has_header(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False

def append_csv_row(csv_path: str, row: Dict):
    header_exists = _csv_has_header(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not header_exists:
            writer.writeheader()
        writer.writerow(row)

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                pass
    return items

def subsample_items(items: List[Dict], n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    if n >= len(items):
        return items[:]
    return rng.sample(items, n)

def extract_user_sentence(obj: Dict) -> Optional[str]:
    msgs = obj.get("messages", [])
    user_turn = next((m for m in msgs if m.get("role") == "user"), None)
    if not user_turn:
        return None
    sentence = str(user_turn.get("content", "")).strip()
    return sentence or None

def dedup_items_by_sentence(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for obj in items:
        s = extract_user_sentence(obj)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(obj)
    return out

def sample_replay(items: List[Dict], k: int, seed: int) -> List[Dict]:
    valid = [obj for obj in items if extract_user_sentence(obj)]
    if not valid:
        return []
    rng = random.Random(seed)
    if k >= len(valid):
        return valid[:]
    return rng.sample(valid, k)


def _read_nvidia_smi_mapping() -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return mapping
    for line in out.strip().splitlines():
        m = re.search(r"^GPU\s+(\d+):.*\(UUID:\s*(GPU-[\w\-]+)\)", line)
        if m:
            idx = int(m.group(1))
            uuid = m.group(2).strip()
            mapping[uuid] = idx
    return mapping

def _translate_cuda_visible_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return
    tokens = [t.strip() for t in cvd.split(",") if t.strip()]
    if not tokens:
        return
    if not any(_UUID_RE.fullmatch(t) for t in tokens):
        return
    uuid_to_idx = _read_nvidia_smi_mapping()
    indices: List[str] = []
    unresolved = []
    for t in tokens:
        if _UUID_RE.fullmatch(t):
            if t in uuid_to_idx:
                indices.append(str(uuid_to_idx[t]))
            else:
                unresolved.append(t)
        else:
            if t.isdigit():
                indices.append(t)
            else:
                unresolved.append(t)
    if not indices:
        indices = ["0"]
    new_val = ",".join(indices)
    os.environ["CUDA_VISIBLE_DEVICES"] = new_val
    msg = f"[INFO] Remapped CUDA_VISIBLE_DEVICES from '{cvd}' to '{new_val}'."
    if unresolved:
        msg += f" Unresolved tokens: {unresolved}. (Using available indices.)"
    print(msg, file=sys.stderr)