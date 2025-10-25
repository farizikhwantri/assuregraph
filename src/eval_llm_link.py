import os
import random
import json
import argparse
import logging
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

# Device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based link prediction on document_aware.json")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to document_aware.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--num_votes", type=int, default=5, 
                        help="Repeat predictions per edge and average (confidence)")
    parser.add_argument("--max_graphs", type=int, default=None, 
                        help="Limit number of graphs for quick eval")
    parser.add_argument("--max_edges_per_graph", type=int, default=None, 
                        help="Subsample edges per graph")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, 
                        default=32, 
                        help="We only need short answers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, 
                        default="out/eval_llm_link", 
                        help="Directory to save results")
    # ICL controls
    parser.add_argument("--icl_k", type=int, default=0,
                        help="Number of in-context labeled examples from the same graph (0 = disabled)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_large_model(model_name_or_path: str, load_in_8bit=False, load_in_4bit=False):
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if not torch.cuda.is_available():
        kwargs.pop("device_map", None)

    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_quant_type"] = "nf4"
        kwargs["bnb_4bit_compute_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    # Ensure generation has a pad token
    if getattr(model.generation_config, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Print memory usage per GPU
    print("\nMemory usage after loading:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    return model, tokenizer

def load_graphs_from_json(json_path: str, split: str, max_graphs: int = None):
    with open(json_path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        docs = payload.get(split, [])
    else:
        docs = payload

    graphs = []
    for d in docs:
        nodes: Dict[str, Dict] = d.get("nodes", {})
        parent_child: Dict[str, List[str]] = d.get("parent_child", {})
        if not nodes or not isinstance(nodes, dict):
            continue

        node_ids = list(nodes.keys())
        id2idx = {nid: i for i, nid in enumerate(node_ids)}
        node_texts = [nodes[nid].get("description", nid) for nid in node_ids]

        # Build positive edges (directed: parent -> child)
        pos_edges = []
        for p, children in parent_child.items():
            if p not in id2idx:
                continue
            for c in children or []:
                if c in id2idx:
                    pos_edges.append([id2idx[p], id2idx[c]])

        if len(node_ids) == 0 or len(pos_edges) == 0:
            # Still keep graphs that have nodes but no edges? Skip for link prediction
            continue

        edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()  # [2, E]
        graphs.append({
            "docname": d.get("docname", ""),
            "node_ids": node_ids,
            "node_texts": node_texts,
            "edge_index": edge_index,
            "num_nodes": len(node_ids),
        })

        if max_graphs is not None and len(graphs) >= max_graphs:
            break
    return graphs

def build_link_prompt(src_text: str, dst_text: str) -> str:
    # Short prompt that encourages "Yes"/"No"
    return (
        "You are an expert on safety-case graphs.\n"
        "Determine whether there is a relationship between two nodes.\n"
        "Answer strictly with 'Yes' or 'No'.\n\n"
        f"Node 1 (candidate): {src_text}\n"
        f"Node 2 (candidate): {dst_text}\n"
        "Link present? Answer:"
    )

def parse_yes_no(generated: str, prompt: str) -> int:
    # Look at the continuation after the prompt
    cont = generated[len(prompt):].strip().lower()
    # First token heuristic
    first = cont.split()
    if len(first) > 0:
        tok = first[0]
        if tok.startswith("yes"):
            return 1, cont
        if tok.startswith("no"):
            return 0, cont
    # Fallback substring check
    if "yes" in cont and "no" not in cont:
        return 1, cont
    if "no" in cont and "yes" not in cont:
        return 0, cont
    # Default negative if uncertain
    return 0, cont

@torch.no_grad()
def score_pair_yes_prob(model, tokenizer, prompt: str, num_votes: int, 
                        max_new_tokens: int, temperature: float, 
                        top_p: float) -> float:
    votes = 0
    outputs = []
    # inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    inputs = tokenizer(prompt, return_tensors="pt", 
                       truncation=True, max_length=2048).to(DEVICE)
    for _ in range(num_votes):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            out = parse_yes_no(text, tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            votes += out[0]
            outputs.append(out[1])
    if random.random() < 0.01:
        print(f"\n{prompt}\n\n Votes:", votes, "Outputs:", outputs)
    return votes / float(num_votes), outputs

def build_balanced_edges(edge_index: torch.Tensor, num_nodes: int, max_edges_per_graph: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Positive edges
    pos_pairs = edge_index.t().contiguous()  # [E, 2]
    if max_edges_per_graph is not None and pos_pairs.size(0) > max_edges_per_graph:
        idx = torch.randperm(pos_pairs.size(0))[:max_edges_per_graph]
        pos_pairs = pos_pairs[idx]

    # Balanced negatives
    neg = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_pairs.size(0),
    ).t().contiguous()

    all_pairs = torch.cat([pos_pairs, neg], dim=0)  # [E_pos + E_neg, 2]
    labels = torch.cat([torch.ones(pos_pairs.size(0)), torch.zeros(neg.size(0))], dim=0)
    return all_pairs, labels

# def evaluate_llm_link(
#     model, tokenizer, graphs: List[Dict], num_votes: int, max_new_tokens: int, temperature: float, top_p: float,
#     max_edges_per_graph: int = None, debug: bool = False
# ):
#     y_true = []
#     y_score = []
#     outputs = []

#     for g in tqdm(graphs, desc="Graphs"):
#         edge_index = g["edge_index"]
#         num_nodes = g["num_nodes"]
#         node_texts = g["node_texts"]

#         all_pairs, labels = build_balanced_edges(edge_index, num_nodes, max_edges_per_graph)

#         for (u, v), y in tqdm(zip(all_pairs.tolist(), labels.tolist()), total=all_pairs.size(0), leave=False, desc="Edges"):
#             src = node_texts[u] if u < len(node_texts) else f"Node {u}"
#             dst = node_texts[v] if v < len(node_texts) else f"Node {v}"
#             prompt = build_link_prompt(src, dst)
#             score, output = score_pair_yes_prob(model, tokenizer, prompt, num_votes, max_new_tokens, temperature, top_p)
#             y_true.append(int(y))
#             y_score.append(float(score))
#             outputs.append(output)
#             # Debugging output
#             if debug:
#                 print("------")
#                 print(prompt)
#                 print(f"score={score:.3f}, label={y}")

#     # Metrics
#     if len(set(y_true)) < 2:
#         print("Only one class present in y_true; ROC-AUC is undefined.")
#         roc = float("nan")
#         ap = float("nan")
#     else:
#         roc = roc_auc_score(y_true, y_score)
#         ap = average_precision_score(y_true, y_score)

#     return roc, ap, y_true, y_score, outputs

def build_link_prompt(src_text: str, dst_text: str) -> str:
    # Short prompt that encourages "Yes"/"No"
    return (
        "You are an expert on safety-case graphs.\n"
        "Determine whether there is a relationship between two nodes.\n"
        "Answer strictly with 'Yes' or 'No'.\n\n"
        f"Node 1 (candidate): {src_text}\n"
        f"Node 2 (candidate): {dst_text}\n"
        "Link present? Answer:"
    )

# New: sample ICL examples (balanced positives/negatives) from the same graph, excluding the queried nodes.
def sample_icl_examples(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_texts: List[str],
    exclude_pair: Tuple[int, int],
    k: int,
) -> List[Tuple[str, str, int]]:
    """
    Returns up to k examples as (src_text, dst_text, label) from the same graph.
    - Half positives from existing edges.
    - Half negatives from negative_sampling.
    - Excludes examples that involve either node in exclude_pair.
    """
    if k <= 0:
        return []

    u_ex, v_ex = exclude_pair
    examples: List[Tuple[str, str, int]] = []

    # Positives
    pos_pairs = edge_index.t().contiguous()  # [E, 2]
    pos_keep = []
    for uv in pos_pairs.tolist():
        u, v = uv
        if u not in (u_ex, v_ex) and v not in (u_ex, v_ex):
            pos_keep.append((u, v))
    random.shuffle(pos_keep)
    k_pos = min(len(pos_keep), k // 2)
    pos_keep = pos_keep[:k_pos]
    for (u, v) in pos_keep:
        examples.append((node_texts[u], node_texts[v], 1))

    # Negatives (ensure not using either excluded node)
    want_neg = k - len(examples)
    if want_neg > 0:
        # Ask for more than needed; filter and truncate.
        num_try = min(want_neg * 4, max(1, num_nodes * num_nodes))
        neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_try,
        ).t().contiguous()
        neg_keep = []
        seen = set()
        for (u, v) in neg.tolist():
            if (u == u_ex) or (u == v_ex) or (v == u_ex) or (v == v_ex):
                continue
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            neg_keep.append((u, v))
            if len(neg_keep) >= want_neg:
                break
        for (u, v) in neg_keep[:want_neg]:
            examples.append((node_texts[u], node_texts[v], 0))

    return examples


# New: prompt builder with ICL context.
def build_link_prompt_with_icl(src_text: str, dst_text: str, icl_examples: List[Tuple[str, str, int]]) -> str:
    header = (
        "You are an expert on safety-case graphs.\n"
        "Decide if a directed parent→child link exists between two nodes.\n"
        "Answer strictly with 'Yes' or 'No'.\n\n"
    )

    if not icl_examples:
        return header + \
            f"Node 1 (candidate): {src_text}\n" \
            f"Node 2 (candidate): {dst_text}\n" \
            "Link present? Answer:"

    ctx = ["Here are labeled examples from the same document:"]
    for i, (s, t, y) in enumerate(icl_examples, 1):
        lab = "Yes" if y == 1 else "No"
        ctx.append(f"Example {i}:")
        ctx.append(f"- Node A: {s}")
        ctx.append(f"- Node B: {t}")
        ctx.append(f"- Link: {lab}")
    ctx_text = "\n".join(ctx)

    query = (
        "\nNow, consider the following candidate pair:\n"
        f"Node 1 (candidate): {src_text}\n"
        f"Node 2 (candidate): {dst_text}\n"
        "Link present? Answer:"
    )
    return header + ctx_text + "\n\n" + query


def evaluate_llm_link(
    model, tokenizer, graphs: List[Dict], num_votes: int, \
    max_new_tokens: int, temperature: float, top_p: float,
    max_edges_per_graph: int = None, \
    debug: bool = False, icl_k: int = 0):
    y_true = []
    y_score = []
    outputs = []

    for g in tqdm(graphs, desc="Graphs"):
        edge_index = g["edge_index"]
        num_nodes = g["num_nodes"]
        node_texts = g["node_texts"]

        all_pairs, labels = build_balanced_edges(edge_index, num_nodes, max_edges_per_graph)

        for (u, v), y in tqdm(zip(all_pairs.tolist(), labels.tolist()), total=all_pairs.size(0), leave=False, desc="Edges"):
            src = node_texts[u] if u < len(node_texts) else f"Node {u}"
            dst = node_texts[v] if v < len(node_texts) else f"Node {v}"

            # Compose prompt (with ICL if requested)
            if icl_k > 0:
                icl_examples = sample_icl_examples(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    node_texts=node_texts,
                    exclude_pair=(u, v),
                    k=icl_k,
                )
                prompt = build_link_prompt_with_icl(src, dst, icl_examples)
            else:
                prompt = build_link_prompt(src, dst)

            score, output = score_pair_yes_prob(model, tokenizer, prompt, num_votes, max_new_tokens, temperature, top_p)
            y_true.append(int(y))
            y_score.append(float(score))
            outputs.append(output)

            if debug:
                print("------")
                print(prompt)
                print(f"score={score:.3f}, label={y}")

    if len(set(y_true)) < 2:
        print("Only one class present in y_true; ROC-AUC is undefined.")
        roc = float("nan")
        ap = float("nan")
    else:
        roc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

    return roc, ap, y_true, y_score, outputs


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_llm_link")
    logger.info(f"Args: {args}")
    set_seed(args.seed)

    graphs = load_graphs_from_json(args.dataset_path, args.split, args.max_graphs)
    if len(graphs) == 0:
        raise ValueError("No valid graphs found. Check dataset_path/split and that graphs contain nodes + parent_child.")

    model, tokenizer = load_large_model(args.model_name, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    model.eval()
    # model.to(DEVICE)
    # If the model is sharded (device_map present), do NOT .to(DEVICE)
    if getattr(model, "hf_device_map", None) is None:
        model.to(DEVICE)

    print(f"Loaded {len(graphs)} graphs. Starting LLM-based link prediction...")
    # roc, ap, y_true, y_score, outputs = evaluate_llm_link(
    #     model, tokenizer, graphs,
    #     num_votes=args.num_votes,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     max_edges_per_graph=args.max_edges_per_graph,
    #     debug=args.debug
    # )
    roc, ap, y_true, y_score, outputs = evaluate_llm_link(
        model, tokenizer, graphs,
        num_votes=args.num_votes,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_edges_per_graph=args.max_edges_per_graph,
        debug=args.debug,
        icl_k=args.icl_k,  # pass ICL k
    )
    # calculate the precision-recall f1 score
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    output = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print("Results:")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"Precision: {output[0]:.4f}, Recall: {output[1]:.4f}, F1-score: {output[2]:.4f}")

    # get model base name for saving
    model_name = args.model_name.replace("/", "-").replace("\\", "-")
    dataset_name = args.dataset_path.split("/")[-1].replace(".json", "").replace(".csv", "")
    print(f"Model name: {model_name}, Dataset name: {dataset_name}")

    if args.output_dir:
        filename = f"results-{model_name}-{dataset_name}.json"
        if args.icl_k > 0:
            filename = f"results-{model_name}-{dataset_name}-icl{args.icl_k}.json"
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, filename)
        with open(results_path, "w") as f:
            json.dump({
                "roc_auc": roc,
                "average_precision": ap,
                "precision": output[0],
                "recall": output[1],
                "f1_score": output[2],
                "y_true": y_true,
                "y_score": y_score,
                "outputs": outputs,
                "args": vars(args),
            }, f, indent=4)
        print(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
