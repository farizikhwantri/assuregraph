import os
import json
import random
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from accelerate.utils import set_seed

# ── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")


# ── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based graph-level classification: human vs gpt-4o"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--num_votes", type=int, default=5,
                        help="Majority vote per graph")
    parser.add_argument("--max_graphs", type=int, default=None)
    parser.add_argument("--max_nodes_in_prompt", type=int, default=None,
                        help="Truncate node list to avoid context overflow")
    parser.add_argument("--max_edges_in_prompt", type=int, default=None,
                        help="Truncate edge list to avoid context overflow")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str,
                        default="out/eval_llm_graph_cls")

    # ── ICL from train ──
    parser.add_argument("--icl_k", type=int, default=0,
                        help="Number of ICL examples (0 = disabled)")
    parser.add_argument("--icl_train_path", type=str, default=None)
    parser.add_argument("--icl_train_split", type=str, default="train")
    parser.add_argument("--icl_train_filter_value", type=str, default=None,
                        help="Filter train ICL by model_name")
    parser.add_argument("--icl_train_filter_mode", type=str, default="equal",
                        choices=["equal", "not_equal"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# ── Filter ────────────────────────────────────────────────────────────────────
def make_filter_function(filter_key: Optional[str],
                         filter_value: Optional[str],
                         mode: str = "equal"):
    if filter_key is None or filter_value is None:
        return lambda doc: True
    if mode == "equal":
        return lambda doc: doc.get(filter_key, "") == filter_value
    return lambda doc: doc.get(filter_key, "") != filter_value


# ── Data loading ──────────────────────────────────────────────────────────────
def load_docs_from_json(json_path: str, split: str,
                        max_docs: int = None) -> List[Dict]:
    """Load all docs from a split. model_name is used as label, not filter."""
    with open(json_path) as f:
        payload = json.load(f)
    docs = payload.get(split, payload) if isinstance(payload, dict) else payload

    results = []
    for d in docs:
        nodes: Dict = d.get("nodes", {})
        parent_child: Dict = d.get("parent_child", {})
        model_name: str = d.get("model_name", "")
        if not nodes:
            continue
        # binary label: 1 = human, 0 = gpt-4o / LLM
        label = 1 if model_name.lower() == "human" else 0
        results.append({
            "docname":      d.get("docname", ""),
            "group_docname": d.get("group_docname", ""),
            "model_name":   model_name,
            "label":        label,
            "nodes":        nodes,
            "parent_child": parent_child,
        })
        if max_docs and len(results) >= max_docs:
            break

    return results


# # ── Prompt builders ───────────────────────────────────────────────────────────
# def _serialize_graph(nodes: Dict, parent_child: Dict,
#                      max_nodes: int, max_edges: int) -> str:
#     """Convert nodes + edges to compact text."""
#     node_lines = []
#     node_ids = list(nodes.keys())[:max_nodes]
#     for nid in node_ids:
#         desc = nodes[nid].get("description", nid)
#         node_lines.append(f"  [{nid}] {desc}")
#     truncated_nodes = len(nodes) > max_nodes

#     edge_lines = []
#     count = 0
#     for parent, children in parent_child.items():
#         for child in (children or []):
#             edge_lines.append(f"  {parent} --> {child}")
#             count += 1
#             if count >= max_edges:
#                 break
#         if count >= max_edges:
#             break
#     truncated_edges = sum(len(v) for v in parent_child.values()) > max_edges

#     text = "Nodes:\n" + "\n".join(node_lines)
#     if truncated_nodes:
#         text += f"\n  ... (truncated, total={len(nodes)})"
#     text += "\nEdges:\n" + ("\n".join(edge_lines) if edge_lines else "  (none)")
#     if truncated_edges:
#         text += "\n  ... (truncated)"
#     return text

# model context window sizes
MODEL_MAX_LENGTH = {
    "llama":   131072,   # Llama-3.x
    "mistral": 32768,
    "bert":    512,
    "default": 4096,
}

def get_model_max_length(model_name: str) -> int:
    name = model_name.lower()
    for key, val in MODEL_MAX_LENGTH.items():
        if key in name:
            return val
    return MODEL_MAX_LENGTH["default"]


def _serialize_graph(nodes: Dict, parent_child: Dict,
                     max_nodes: int = None, max_edges: int = None) -> str:
    """Convert nodes + edges to compact text. None = no truncation."""
    node_ids = list(nodes.keys()) if max_nodes is None else list(nodes.keys())[:max_nodes]
    node_lines = []
    for nid in node_ids:
        desc = nodes[nid].get("description", nid)
        node_lines.append(f"  [{nid}] {desc}")
    truncated_nodes = max_nodes is not None and len(nodes) > max_nodes

    edge_lines = []
    count = 0
    for parent, children in parent_child.items():
        for child in (children or []):
            edge_lines.append(f"  {parent} --> {child}")
            count += 1
            if max_edges is not None and count >= max_edges:
                break
        if max_edges is not None and count >= max_edges:
            break
    truncated_edges = max_edges is not None and \
                      sum(len(v) for v in parent_child.values()) > max_edges

    text = "Nodes:\n" + "\n".join(node_lines)
    if truncated_nodes:
        text += f"\n  ... (truncated {len(node_ids)}/{len(nodes)})"
    text += "\nEdges:\n" + ("\n".join(edge_lines) if edge_lines else "  (none)")
    if truncated_edges:
        text += f"\n  ... (truncated {count}/{sum(len(v) for v in parent_child.values())})"
    return text


def build_cls_prompt(graph_text: str,
                     icl_examples: List[Tuple[str, int]] = None) -> str:
    header = (
        "You are an expert on assurance case graphs (safety cases).\n"
        "Your task: decide whether the following graph was created by a human expert "
        "or generated by an LLM (e.g. GPT-4o).\n"
        "Human graphs tend to be precise, concise, and domain-specific.\n"
        "LLM graphs tend to be verbose, generic, or contain hallucinated nodes.\n"
        "Answer strictly with 'human' or 'llm'.\n\n"
    )
    if icl_examples:
        ctx_lines = ["Here are labeled examples:\n"]
        for i, (g_text, lbl) in enumerate(icl_examples, 1):
            lab_str = "human" if lbl == 1 else "llm"
            ctx_lines.append(f"--- Example {i} ---")
            ctx_lines.append(g_text)
            ctx_lines.append(f"Label: {lab_str}\n")
        header += "\n".join(ctx_lines) + "\n"

    return (
        header
        + "--- Graph to classify ---\n"
        + graph_text
        + "\nLabel:"
    )


# ── Inference ─────────────────────────────────────────────────────────────────
def parse_human_llm(generated: str, prompt: str) -> int:
    cont = generated[len(prompt):].strip().lower()
    first = cont.split()
    if first:
        tok = first[0].strip(".,:")
        if tok.startswith("human"):
            return 1
        if tok in ("llm", "gpt", "ai", "no"):
            return 0
    if "human" in cont and "llm" not in cont:
        return 1
    if "llm" in cont and "human" not in cont:
        return 0
    return 0   # default: LLM


# @torch.no_grad()
# def classify_graph(model, tokenizer, prompt: str,
#                    num_votes: int, max_new_tokens: int,
#                    temperature: float, top_p: float, debug: bool = False) -> Tuple[float, List[str]]:
#     """Return (prob_human, [raw_outputs])."""
#     inputs = tokenizer(prompt, return_tensors="pt",
#                        truncation=True, max_length=2048).to(model.device
#                        if hasattr(model, "device") else DEVICE)
#     if debug:
#         print("Prompt:\n", prompt)
#     votes = 0
#     raw = []
#     for _ in range(num_votes):
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#         text = tokenizer.decode(out[0], skip_special_tokens=True)
#         pred = parse_human_llm(text, tokenizer.decode(
#             inputs["input_ids"][0], skip_special_tokens=True))
#         votes += pred
#         raw.append(text[len(prompt):].strip()[:80])
#     return votes / float(num_votes), raw

@torch.no_grad()
def classify_graph(model, tokenizer, prompt: str,
                   num_votes: int, max_new_tokens: int,
                   temperature: float, top_p: float,
                   model_max_length: int = 4096,
                   debug: bool = False) -> Tuple[float, List[str]]:
    """Return (prob_human, [raw_outputs])."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,              # hard truncate only as last resort
        max_length=model_max_length,  # respect model context window
    ).to(model.device if hasattr(model, "device") else DEVICE)

    n_tokens = inputs["input_ids"].shape[1]
    if debug:
        print(f"  Prompt tokens: {n_tokens} / {model_max_length}")
        if n_tokens >= model_max_length:
            # truncated the prompt, warn about it
            print("  ⚠️  Prompt was truncated — consider --max_nodes_in_prompt")

    votes = 0
    raw = []
    for _ in range(num_votes):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = parse_human_llm(text, tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True))
        votes += pred
        raw.append(text[len(prompt):].strip()[:80])
    return votes / float(num_votes), raw


# ── ICL sampling ──────────────────────────────────────────────────────────────
def sample_icl_examples_for_cls(
    train_docs: List[Dict],
    k: int,
    max_nodes: int,
    max_edges: int,
    balance_by_model: bool = False,
) -> List[Tuple[str, int]]:
    """Return up to k (graph_text, label) pairs."""
    if k <= 0 or not train_docs:
        return []

    def _pick(docs, n):
        chosen = random.sample(docs, min(n, len(docs)))
        return [
            (_serialize_graph(d["nodes"], d["parent_child"], max_nodes, max_edges),
             d["label"])
            for d in chosen
        ]

    if not balance_by_model:
        return _pick(train_docs, k)

    human_docs = [d for d in train_docs if d["label"] == 1]
    llm_docs   = [d for d in train_docs if d["label"] == 0]
    k_h = k // 2
    k_l = k - k_h
    examples = _pick(human_docs, k_h) + _pick(llm_docs, k_l)
    random.shuffle(examples)
    return examples[:k]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_large_model(model_name_or_path, load_in_8bit=False, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", torch_dtype=torch.float16,
                  trust_remote_code=True)
    if not torch.cuda.is_available():
        kwargs.pop("device_map")
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_compute_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate_llm_graph_cls(
    model, tokenizer,
    docs: List[Dict],
    num_votes: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_nodes: int,
    max_edges: int,
    model_max_length: int,
    icl_examples: List[Tuple[str, int]] = None,
    debug: bool = False,
) -> Tuple[List[int], List[float], List[str]]:

    y_true, y_score, raw_outputs = [], [], []

    for doc in tqdm(docs, desc="Graphs"):
        graph_text = _serialize_graph(
            doc["nodes"], doc["parent_child"], max_nodes, max_edges)
        prompt = build_cls_prompt(graph_text, icl_examples)

        # print prompt randomly
        if debug and random.random() < 0.1:
            print(f"\n=== Debug Prompt for docname={doc['docname']}  model_name={doc['model_name']} ===")
            print(prompt)
            print("=== End of Prompt ===\n")

        prob_human, raw = classify_graph(
            model, tokenizer, prompt,
            num_votes, max_new_tokens, temperature, top_p,
            model_max_length=model_max_length,
            debug=debug,)
        if debug:
            # print doc label
            print(f"\nDoc: {doc['docname']}  Model: {doc['model_name']}  Label: {doc['label']}")
        y_true.append(doc["label"])
        y_score.append(prob_human)
        raw_outputs.append(raw)

        if debug:
            print(f"\ndocname={doc['docname']}  model_name={doc['model_name']}")
            print(f"  prob_human={prob_human:.2f}  label={doc['label']}")
            print("  raw:", raw[:2])

    return y_true, y_score, raw_outputs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_llm_graph_cls")
    logger.info("args: %s", args)
    set_seed(args.seed)

    model_max_length = get_model_max_length(args.model_name)
    print(f"Model context window: {model_max_length} tokens")

    # ── load test docs (no filter — model_name is the label) ──
    docs = load_docs_from_json(args.dataset_path, args.split, args.max_graphs)
    if not docs:
        raise ValueError("No docs loaded. Check --dataset_path / --split")
    print(f"Test docs: {len(docs)}  "
          f"(human={sum(d['label']==1 for d in docs)}, "
          f"llm={sum(d['label']==0 for d in docs)})")

    # ── load ICL train docs (filter only for ICL source selection) ──
    icl_examples = None
    if args.icl_k > 0 and args.icl_train_path:
        train_filter = make_filter_function(
            "model_name", args.icl_train_filter_value, args.icl_train_filter_mode)
        train_docs = load_docs_from_json(
            args.icl_train_path, args.icl_train_split)
        # apply filter only for ICL source selection
        train_docs = [d for d in train_docs if train_filter(d)]
        print(f"Train docs for ICL: {len(train_docs)}")
        icl_examples = sample_icl_examples_for_cls(
            train_docs, args.icl_k,
            args.max_nodes_in_prompt, args.max_edges_in_prompt,
            balance_by_model=True,
        )
        print(f"Sampled {len(icl_examples)} ICL examples "
              f"(human={sum(l==1 for _,l in icl_examples)}, "
              f"llm={sum(l==0 for _,l in icl_examples)})")

    # ── load model ──
    model, tokenizer = load_large_model(
        args.model_name, args.load_in_8bit, args.load_in_4bit)
    model.eval()
    if getattr(model, "hf_device_map", None) is None:
        model.to(DEVICE)

    # ── evaluate ──
    # y_true, y_score, raw_outputs = evaluate_llm_graph_cls(
    #     model, tokenizer, docs,
    #     num_votes=args.num_votes,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     max_nodes=args.max_nodes_in_prompt,
    #     max_edges=args.max_edges_in_prompt,
    #     icl_examples=icl_examples,
    #     debug=args.debug,
    # )
    y_true, y_score, raw_outputs = evaluate_llm_graph_cls(
        model, tokenizer, docs,
        num_votes=args.num_votes,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_nodes=args.max_nodes_in_prompt,   # None = no truncation
        max_edges=args.max_edges_in_prompt,   # None = no truncation
        model_max_length=model_max_length,
        icl_examples=icl_examples,
        debug=args.debug,
    )

    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    report = classification_report(
        y_true, y_pred, target_names=["LLM(gpt-4o)", "Human"], zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")

    print("\n===== Classification Report =====")
    print(report)
    print("===== Confusion Matrix =====")
    print(cm)
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {ap:.4f}")

    # ── save ──
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_tag = args.model_name.replace("/", "-")
        fname = f"eval_graph_cls-{model_tag}-icl{args.icl_k}.json"
        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, "w") as f:
            json.dump({
                "roc_auc": roc, "pr_auc": ap,
                "report": report,
                "confusion_matrix": cm.tolist(),
                "y_true": y_true, "y_score": y_score,
                "raw_outputs": raw_outputs,
                "args": vars(args),
            }, f, indent=2)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()