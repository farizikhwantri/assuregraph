import sys
import argparse
import re
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import pandas as pd
from tabulate import tabulate

_bleu_smooth = SmoothingFunction().method1


METRIC_COLS = [
    "node_recall",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "ancestor_precision",
    "ancestor_recall",
    "ancestor_f1",
]

METRIC_DISPLAY = {
    "node_recall":         "Node R",
    "edge_precision":      "Edge P",
    "edge_recall":         "Edge R",
    "edge_f1":             "Edge F1",
}


def cosine_similarity_text(a: str, b: str) -> float:
    vec = TfidfVectorizer().fit_transform([a, b])
    return cosine_similarity(vec[0], vec[1])[0, 0]

def bleu_similarity(a: str, b: str) -> float:
    """
    Symmetric sentence BLEU in [0,1]
    """
    ref = [a.lower().split()]
    hyp = b.lower().split()
    return sentence_bleu(ref, hyp, smoothing_function=_bleu_smooth)


def exact_match(a: str, b: str) -> float:
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0


def _desc_similarity(a: str, b: str) -> float:
    """Return similarity in [0,1]."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def text_similarity(a: str, b: str, method: str) -> float:
    if method == "exact":
        return exact_match(a, b)
    elif method == "bleu":
        return bleu_similarity(a, b)
    elif method == "cosine":
        return cosine_similarity_text(a, b)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def _node_subst_cost(n1, n2):
    """
    Substitution cost for nodes.
    - Different type: max cost
    - Same type: 1 - description similarity
    """
    if n1["type"] != n2["type"]:
        return 1.0
    return 1.0 - _desc_similarity(n1["description"], n2["description"])


def _node_del_cost(n):
    return 1.0


def _node_ins_cost(n):
    return 1.0


def _edge_subst_cost(e1, e2):
    return 0.0  # edges are unlabeled


def _edge_del_cost(e):
    return 1.0


def _edge_ins_cost(e):
    return 1.0


def _build_graph(acase: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    # nodes
    for nid, node in acase["nodes"].items():
        G.add_node(
            nid,
            type=node["type"],
            description=node["description"]
        )

    # edges (parent -> child)
    for parent, children in acase["parent_child"].items():
        for child in children:
            if parent in G and child in G:
                G.add_edge(parent, child)

    return G

# def normalize_llm_key(key: str) -> str:
#     """
#     Normalize LLM docname keys like:
#     'r8 acas xu 0 0' -> 'acas xu'
#     """
#     key = key.lower()

#     # remove run prefix: r8, r10, etc.
#     key = re.sub(r"\br\d+\b", "", key)

#     # remove trailing numeric tokens
#     key = re.sub(r"\b\d+\b", "", key)

#     # normalize whitespace
#     key = re.sub(r"\s+", " ", key).strip()

#     return key


def match_llm_to_human(human_keys, llm_keys):
    """
    Returns mapping: llm_key -> human_key
    """
    mapping = {}

    for llm in llm_keys:
        norm = normalize_llm_key(llm)

        # exact or substring match against human keys
        for h in human_keys:
            if h in norm:
                mapping[llm] = h
                break

    return mapping



# def _base_docname(docname: str) -> str:
#     """
#     Normalize docname so human & LLM outputs align.
#     """
#     name = Path(docname).stem.lower()
#     name = name.replace("gpt-4o", "").replace("_", " ")
#     return name.strip()


def _base_docname(name: str) -> str:
    """
    Canonicalize docname or group_docname so that:
    'r8 acas xu 0 0' -> 'acas xu'
    'acas_xu.txt'   -> 'acas xu'
    """
    name = name.lower()

    # remove file extension
    name = re.sub(r"\.txt$", "", name)

    # remove run prefix: r8, r10, etc.
    name = re.sub(r"\br\d+\b", "", name)

    # remove standalone numbers (generation indices)
    name = re.sub(r"\b\d+\b", "", name)

    # normalize separators
    name = name.replace("_", " ")

    # collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name

# def normalize_llm_key(key: str) -> str:
#     key = key.lower()
#     # # remove 'experiment-' or 'experiment ' at the start
#     # key = re.sub(r"^experiment-?\s*", "", key)
#     # # remove tokens
#     # key = re.sub(r"\b-safety case\b", "", key)
#     # key = re.sub(r"\bexperiments?\b", "", key)

#     key = re.sub(r"\br\d+\b", "", key)   # remove r8, r7, etc.
#     key = re.sub(r"\b\d+\b", "", key)    # remove run indices
#     key = re.sub(r"\s+", " ", key).strip()
#     return key

def normalize_llm_key(key: str) -> str:
    key = key.lower()
    # remove 'experiment-' or 'experiment ' at the start
    key = re.sub(r"^experiment-?\s*", "", key)
    # remove tokens
    key = re.sub(r"\bsafety case\b", "", key)
    key = re.sub(r"\bexperiments?\b", "", key)
    # remove run indices and standalone numbers
    key = re.sub(r"\br\d+\b", "", key)
    key = re.sub(r"\b\d+\b", "", key)
    # remove stray hyphens at boundaries or next to spaces (preserve inner hyphens)
    key = re.sub(r"(^|\s)-\s*", r"\1", key)   # leading "- "
    key = re.sub(r"\s*-(\s|$)", r"\1", key)   # trailing " -"
    # collapse whitespace
    key = re.sub(r"\s+", " ", key).strip()
    return key

def match_nodes_semantic(
    h_nodes: dict,
    l_nodes: dict,
    method: str,
    threshold: float,
):
    """
    h_nodes, l_nodes: {node_id: {"description": str, ...}}

    Returns:
        dict {human_node_id: llm_node_id}
    """

    scores = []

    for hi, h in h_nodes.items():
        for li, l in l_nodes.items():
            sim = text_similarity(
                h["description"],
                l["description"],
                method=method,
            )
            if sim >= threshold:
                scores.append((sim, hi, li))

    # highest similarity first
    scores.sort(reverse=True, key=lambda x: x[0])

    matched_h, matched_l = set(), set()
    matches = {}

    for sim, hi, li in scores:
        if hi in matched_h or li in matched_l:
            continue
        matches[hi] = li
        matched_h.add(hi)
        matched_l.add(li)

    return matches, scores



def graph_edit_distance_same_doc(
    human_cases: list,
    llm_cases: list,
    llm_model_name: str = "gpt-4o",
):
    """
    Compute GED between human and LLM assurance cases
    with the same group_docname.

    Returns:
        dict {(human_doc, llm_doc): ged}
    """

    # 1. index human cases (1 per dataset)
    human_map = {
        _base_docname(c["docname"]): c
        for c in human_cases
        if c["model_name"] == "human"
    }

    # 2. group LLM cases (many per dataset)
    llm_map = defaultdict(list)
    for c in llm_cases:
        key = _base_docname(c["docname"])
        key = normalize_llm_key(key)
        # remove model name prefix if present
        if key.startswith(llm_model_name):
            key = key[len(llm_model_name):].strip()
        llm_map[key].append(c)

    print("Human keys:", human_map.keys())
    print("LLM keys:", llm_map.keys())

    results = {}

    # 3. compute GED: each LLM run → its human reference
    for doc, llm_cases_for_doc in llm_map.items():
        print(f"Processing doc: {doc} with {len(llm_cases_for_doc)} LLM cases")
        if doc not in human_map:
            continue

        h_case = human_map[doc]
        Gh = _build_graph(h_case)

        for l_case in llm_cases_for_doc:
            Gl = _build_graph(l_case)

            # ged = nx.graph_edit_distance(
            #     Gh,
            #     Gl,
            #     node_subst_cost=_node_subst_cost,
            #     node_del_cost=_node_del_cost,
            #     node_ins_cost=_node_ins_cost,
            #     edge_subst_cost=_edge_subst_cost,
            #     edge_del_cost=_edge_del_cost,
            #     edge_ins_cost=_edge_ins_cost,
            # )
            ged = nx.optimize_graph_edit_distance(
                Gh,
                Gl,
                node_subst_cost=_node_subst_cost,
                node_del_cost=_node_del_cost,
                node_ins_cost=_node_ins_cost,
                edge_subst_cost=_edge_subst_cost,
                edge_del_cost=_edge_del_cost,
                edge_ins_cost=_edge_ins_cost,
            )
            # save the generator object for later retrieval of the actual edit path if needed
            ged = next(ged)  # get the first (lowest cost) edit distance
            results[(doc, l_case["docname"])] = ged

    return results


def node_recall_and_edge_prf1_same_doc(
    human_cases: list,
    llm_cases: list,
    llm_model_name: str = "gpt-4o",
):
    """
    Compute node recall and edge precision / recall / F1
    between human and LLM assurance cases with the same dataset.

    Returns:
        dict {
            (human_doc, llm_doc): {
                "node_recall": float,
                "edge_precision": float,
                "edge_recall": float,
                "edge_f1": float,
            }
        }
    """

    # 1. index human cases (1 per dataset)
    human_map = {
        _base_docname(c["docname"]): c
        for c in human_cases
        if c["model_name"] == "human"
    }

    # 2. group LLM cases (many per dataset)
    llm_map = defaultdict(list)
    for c in llm_cases:
        key = _base_docname(c["docname"])
        key = normalize_llm_key(key)
        if key.startswith(llm_model_name):
            key = key[len(llm_model_name):].strip()
        llm_map[key].append(c)

    print("Human keys:", human_map.keys())
    print("LLM keys:", llm_map.keys())

    results = {}

    # 3. LLM → human evaluation
    for doc, llm_cases_for_doc in llm_map.items():
        if doc not in human_map:
            continue

        h_case = human_map[doc]

        # human nodes & edges
        Vh = set(h_case["nodes"].keys())
        Eh = {
            (p, c)
            for p, children in h_case["parent_child"].items()
            for c in children
        }

        for l_case in llm_cases_for_doc:
            Vl = set(l_case["nodes"].keys())

            # --- node recall ---
            intersection_nodes = Vh & Vl
            node_recall = len(intersection_nodes) / len(Vh) if Vh else 0.0

            # --- edge PRF1 on intersection nodes ---
            El = {
                (p, c)
                for p, children in l_case["parent_child"].items()
                for c in children
            }

            Eh_i = {
                (p, c)
                for (p, c) in Eh
                if p in intersection_nodes and c in intersection_nodes
            }

            El_i = {
                (p, c)
                for (p, c) in El
                if p in intersection_nodes and c in intersection_nodes
            }

            tp = len(Eh_i & El_i)
            fp = len(El_i - Eh_i)
            fn = len(Eh_i - El_i)

            edge_precision = tp / (tp + fp) if tp + fp else 0.0
            edge_recall    = tp / (tp + fn) if tp + fn else 0.0
            edge_f1        = (
                2 * edge_precision * edge_recall / (edge_precision + edge_recall)
                if edge_precision + edge_recall else 0.0
            )

            results[(doc, l_case["docname"])] = {
                "node_recall": node_recall,
                "edge_precision": edge_precision,
                "edge_recall": edge_recall,
                "edge_f1": edge_f1,
            }

    return results




def semantic_node_recall_and_edge_prf1_same_doc(
    human_cases: list,
    llm_cases: list,
    llm_model_name: str = "gpt-4o",
    similarity_method: str = "cosine",   # "exact" | "bleu" | "cosine"
    similarity_threshold: float = 0.0,
):
    """
    Returns:
        dict {
            (human_doc, llm_doc): {
                "node_recall": float,
                "edge_precision": float,
                "edge_recall": float,
                "edge_f1": float,
            }
        }
    """

    # 1. index human cases
    human_map = {
        _base_docname(c["docname"]): c
        for c in human_cases
        if c["model_name"] == "human"
    }

    # 2. group LLM cases
    llm_map = defaultdict(list)
    for c in llm_cases:
        key = normalize_llm_key(_base_docname(c["docname"]))
        if key.startswith(llm_model_name):
            key = key[len(llm_model_name):].strip()
        llm_map[key].append(c)

    results = {}

    print("Human keys:", human_map.keys())
    print("LLM keys:", llm_map.keys())  

    # 3. evaluate each LLM run against its human reference
    for doc, llm_list in llm_map.items():
        if doc not in human_map:
            continue

        h_case = human_map[doc]
        h_nodes = h_case["nodes"]

        # human edges
        Eh = {
            (p, c)
            for p, children in h_case["parent_child"].items()
            for c in children
        }

        for l_case in llm_list:
            l_nodes = l_case["nodes"]

            # --- semantic node matching ---
            matches, scores = match_nodes_semantic(
                h_nodes,
                l_nodes,
                method=similarity_method,
                threshold=similarity_threshold,
            )

            # calculate the average similarity scores
            avg_score = sum(sim for sim, _, _ in scores) / len(scores) if scores else 0.0

            # --- node recall ---
            node_recall = len(matches) / len(h_nodes) if h_nodes else 0.0

            # invert mapping
            inv_matches = {v: k for k, v in matches.items()}

            # --- edge sets restricted to matched nodes ---
            Eh_mapped = {
                (matches[p], matches[c])
                for (p, c) in Eh
                if p in matches and c in matches
            }

            El = {
                (p, c)
                for p, children in l_case["parent_child"].items()
                for c in children
                if p in inv_matches and c in inv_matches
            }

            tp = len(Eh_mapped & El)
            fp = len(El - Eh_mapped)
            fn = len(Eh_mapped - El)

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall    = tp / (tp + fn) if tp + fn else 0.0
            f1        = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            results[(doc, l_case["docname"])] = {
                similarity_method + "_similarity": avg_score,
                "node_recall": node_recall,
                "edge_precision": precision,
                "edge_recall": recall,
                "edge_f1": f1,
            }

    return results

# ── Table helpers ─────────────────────────────────────────────────────────────

def build_per_run_table(results: dict, similarity_method: str) -> pd.DataFrame:
    """
    Table 1: every (human_doc, llm_run) pair as a row.

    Columns: Document | LLM Docname | Sim Score | Node R | Edge P | ... 
    """
    sim_col = f"{similarity_method}_similarity"
    rows = []
    for (human_doc, llm_doc), metrics in sorted(results.items()):
        row = {
            "Document":    human_doc,
            "LLM Docname": llm_doc,
            "Sim Score":   round(metrics.get(sim_col, float("nan")), 4),
        }
        for col in METRIC_COLS:
            row[METRIC_DISPLAY[col]] = round(metrics.get(col, float("nan")), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_doc_aggregated_table(
    results: dict, similarity_method: str
) -> pd.DataFrame:
    """
    Table 2: one row per human_doc — mean ± std across all LLM runs.

    Last row is the overall average across all documents.
    """
    sim_col = f"{similarity_method}_similarity"
    doc_accum = defaultdict(lambda: defaultdict(list))

    for (human_doc, _), metrics in results.items():
        doc_accum[human_doc]["Sim Score"].append(
            metrics.get(sim_col, float("nan")))
        for col in METRIC_COLS:
            doc_accum[human_doc][METRIC_DISPLAY[col]].append(
                metrics.get(col, float("nan")))

    rows = []
    all_accum = defaultdict(list)

    for doc in sorted(doc_accum.keys()):
        m = doc_accum[doc]
        row = {"Document": doc, "N Runs": len(m["Sim Score"])}
        for display_col in ["Sim Score"] + [METRIC_DISPLAY[c] for c in METRIC_COLS]:
            vals = [v for v in m[display_col] if v == v]
            row[display_col] = (
                f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if vals else "nan"
            )
            all_accum[display_col].extend(vals)
        rows.append(row)

    # ── overall average row ──
    avg_row = {"Document": "── AVERAGE ──", "N Runs": len(results)}
    for display_col in ["Sim Score"] + [METRIC_DISPLAY[c] for c in METRIC_COLS]:
        vals = [v for v in all_accum[display_col] if v == v]
        avg_row[display_col] = (
            f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if vals else "nan"
        )
    rows.append(avg_row)

    return pd.DataFrame(rows)


def build_overall_summary_table(
    results: dict, similarity_method: str
) -> pd.DataFrame:
    """
    Table 3: one row per metric — mean / std / min / max across all pairs.
    """
    sim_col = f"{similarity_method}_similarity"
    accum = defaultdict(list)

    for metrics in results.values():
        accum["Sim Score"].append(metrics.get(sim_col, float("nan")))
        for col in METRIC_COLS:
            accum[METRIC_DISPLAY[col]].append(metrics.get(col, float("nan")))

    rows = []
    for display_col in ["Sim Score"] + [METRIC_DISPLAY[c] for c in METRIC_COLS]:
        vals = [v for v in accum[display_col] if v == v]
        rows.append({
            "Metric": display_col,
            "Mean":   round(float(np.mean(vals)), 4) if vals else float("nan"),
            "Std":    round(float(np.std(vals)),  4) if vals else float("nan"),
            "Min":    round(float(np.min(vals)),  4) if vals else float("nan"),
            "Max":    round(float(np.max(vals)),  4) if vals else float("nan"),
            "N":      len(vals),
        })
    return pd.DataFrame(rows)


def print_and_save_table(
    df: pd.DataFrame,
    title: str,
    output_path: str = None,
    tablefmt: str = "rounded_outline",
):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(tabulate(df, headers="keys", tablefmt=tablefmt,
                   showindex=False, floatfmt=".4f"))
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  → saved to {output_path}")


def visualize_results(
    results: dict,
    similarity_method: str = "cosine",
    output_dir: str = None,
    tablefmt: str = "rounded_outline",
):
    """
    Entry point: print all three tables and optionally save CSVs.

    Args:
        results:          output of semantic_node_recall_and_edge_prf1_same_doc()
        similarity_method: "cosine" | "bleu" | "exact"
        output_dir:       if set, saves each table as a CSV
        tablefmt:         tabulate format string
    """
    tables = [
        (
            build_per_run_table(results, similarity_method),
            "Table 1 — Per-run  (human ground-truth vs every LLM run)",
            "per_run_results.csv",
        ),
        (
            build_per_doc_aggregated_table(results, similarity_method),
            "Table 2 — Per-document aggregated  (mean ± std across LLM runs)",
            "per_doc_aggregated.csv",
        ),
        (
            build_overall_summary_table(results, similarity_method),
            "Table 3 — Overall metric summary",
            "overall_summary.csv",
        ),
    ]

    for df, title, fname in tables:
        out_path = (
            str(Path(output_dir) / fname) if output_dir else None
        )
        print_and_save_table(df, title, out_path, tablefmt)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute graph metrics: human ground-truth vs LLM-generated assurance cases"
    )
    p.add_argument("--human_path", type=str, required=True,
                   help="Path to human cases JSON (e.g. human.json)")
    p.add_argument("--llm_paths", type=str, nargs="+", required=True,
                   help="One or more LLM output JSON files (will be merged)")
    p.add_argument("--llm_model_name", type=str, default="gpt-4o",
                   help="LLM model name prefix to strip from docnames")
    p.add_argument("--similarity_method", type=str, default="cosine",
                   choices=["exact", "bleu", "cosine"],
                   help="Node similarity method")
    p.add_argument("--similarity_threshold", type=float, default=0.0,
                   help="Minimum similarity score for node matching")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save CSV tables (optional)")
    p.add_argument("--tablefmt", type=str, default="rounded_outline",
                   help="tabulate format: rounded_outline, github, latex, etc.")
    return p.parse_args()


if __name__ == "__main__":
    # read from sys.argv or hardcode paths to human.json and llm.json
    # ground_truth_file = sys.argv[1] if len(sys.argv) > 1 else "human.json"
    # llm_output_file = sys.argv[2] if len(sys.argv) > 2 else "llm.json"
    # similarity_method = sys.argv[3] if len(sys.argv) > 3 else "cosine"
    # output_dir        = sys.argv[4] if len(sys.argv) > 4 else None

    args = parse_args()
    ground_truth_file = args.human_path
    llm_output_files = args.llm_paths
    similarity_method = args.similarity_method
    similarity_threshold = args.similarity_threshold
    output_dir = args.output_dir

    print(f"Comparing {ground_truth_file} and {', '.join(llm_output_files)}...")
    with open(ground_truth_file) as f:
        human_cases = json.load(f)

    # with open(llm_output_file) as f:
    #     llm_cases = json.load(f)
        # ── merge all LLM files ───────────────────────────────────────────────────
    llm_cases = []
    for llm_path in args.llm_paths:
        with open(llm_path) as f:
            llm_cases += json.load(f)
        print(f"  Loaded {llm_path} → total LLM cases: {len(llm_cases)}")

    # ged_scores = graph_edit_distance_same_doc(human_cases, llm_cases)
    prf1_scores = semantic_node_recall_and_edge_prf1_same_doc(
        human_cases, llm_cases, similarity_method=similarity_method,
        similarity_threshold=similarity_threshold,
    )

    # for doc, score in ged_scores.items():
        # print(doc, score)

    for doc, metrics in prf1_scores.items():
        print(doc, metrics)

    # calculate average metrics across all documents
    avg_metrics = defaultdict(float)
    for metrics in prf1_scores.values(): 
        for k, v in metrics.items(): 
            avg_metrics[k] += v 
    num_docs = len(prf1_scores)

    for k in avg_metrics:
        avg_metrics[k] /= num_docs

    print("Average metrics across all documents:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    visualize_results(
        prf1_scores,
        similarity_method=similarity_method,
        output_dir=output_dir,
        tablefmt="rounded_outline",
    )
