import argparse
import json
import logging
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ClassLabel
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer, CaptumExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.explain.metric import fidelity, unfaithfulness

from pipeline import get_graph_dataset
from graph_model import (
    SentenceGGraphClassifier,
    SentenceGATGraphClassifier,
    SentenceSAGEGraphClassifier,
    SentenceDGConvGraphClassifier,
    SentenceDGATGraphClassifier,
    SentenceDGSAGEGraphClassifier,
)
from unigraph import UniGraph
from unigraph import UniGraphGGraphClassifier

from graph_prompt import (
    SentenceGraphPromptClassifier,
    SentenceGATGraphPromptClassifier,
    SentenceSAGEGraphPromptClassifier,
)


def model_mapper(model_name):
    model_map = {
        "SentenceGGraphClassifier": SentenceGGraphClassifier,
        "SentenceGATGraphClassifier": SentenceGATGraphClassifier,
        "SentenceSAGEGraphClassifier": SentenceSAGEGraphClassifier,
        "SentenceDGConvGraphClassifier": SentenceDGConvGraphClassifier,
        "SentenceDGATGraphClassifier": SentenceDGATGraphClassifier,
        "SentenceDGSAGEGraphClassifier": SentenceDGSAGEGraphClassifier,
        "UniGraphGGraphClassifier": UniGraphGGraphClassifier,
        "SentenceGraphPromptClassifier": SentenceGraphPromptClassifier,
        "SentenceGATGraphPromptClassifier": SentenceGATGraphPromptClassifier,
        "SentenceSAGEGraphPromptClassifier": SentenceSAGEGraphPromptClassifier,
    }
    return model_map.get(model_name, None)


def build_label_encoder(dataset, graph_label_key):
    unique = set()
    for item in dataset:
        y = item.get(graph_label_key, None)
        if y is None:
            continue
        if isinstance(y, list):
            for v in y:
                unique.add(v)
        else:
            unique.add(y)
    names = sorted(list(unique))
    return ClassLabel(names=names)


@torch.no_grad()
def evaluate_graph_classification_loader(model, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    for tokenized, batch, targets in loader:
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        batch = batch.to(device)
        logits, _, _ = model(tokenized, batch.edge_index, batch_ids=batch.batch)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(targets.detach().cpu().tolist())
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0
    pre, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    return pre, rec, f1


def collect_doc_metrics(model, dataset: List[Dict[str, Any]], tokenizer, label_encoder: ClassLabel,
                        task_label_key: str, device: torch.device):
    model.eval()
    metrics = []
    for doc_idx, doc in enumerate(dataset):
        sentences = doc["sentences"]
        ei = doc["edge_index"]
        if not torch.is_tensor(ei):
            ei = torch.tensor(ei, dtype=torch.long)
        ei = ei.to(device)
        batch_vec = torch.zeros(len(sentences), dtype=torch.long, device=device)
        tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits, _, _ = model(tokenized, ei, batch_ids=batch_vec)
            probs = torch.softmax(logits if logits.dim() == 2 else logits.unsqueeze(0), dim=-1)[0]
            y_pred = int(torch.argmax(probs, dim=-1).item())
            top2 = torch.topk(probs, k=min(2, probs.size(0))).values
            margin = float((top2[0] - (top2[1] if top2.numel() > 1 else 0.0)).item())
            pmax = float(probs[y_pred].item())
        y = doc.get(task_label_key, None)
        if isinstance(y, list):
            y = y[0]
        y_true = label_encoder.str2int(y) if isinstance(y, str) else int(y)
        metrics.append({
            "doc_idx": doc_idx,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_max": pmax,
            "margin": margin,
            "correct": bool(y_true == y_pred),
            "num_nodes": len(sentences),
            "num_edges": int(ei.size(1)),
        })
    return metrics


def sample_random_docs(metrics, budget, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(metrics), generator=g)[:min(budget, len(metrics))].tolist()
    return [metrics[i]["doc_idx"] for i in idx]


def sample_stratified_by_true_docs(metrics, per_class, seed=42):
    buckets, selected, g = {}, [], torch.Generator().manual_seed(seed)
    for m in metrics:
        buckets.setdefault(m["y_true"], []).append(m)
    for _, items in buckets.items():
        if len(items) <= per_class:
            selected.extend([m["doc_idx"] for m in items])
        else:
            order = torch.randperm(len(items), generator=g)[:per_class].tolist()
            selected.extend([items[i]["doc_idx"] for i in order])
    return selected


def sample_confidence_mix_docs(metrics, low_k=20, high_k=20):
    margins = torch.tensor([m["margin"] for m in metrics])
    low_idx = torch.topk(-margins, k=min(low_k, len(metrics))).indices.tolist()
    high_idx = torch.topk(margins, k=min(high_k, len(metrics))).indices.tolist()
    chosen = set(low_idx + high_idx)
    return [metrics[i]["doc_idx"] for i in chosen]


def sample_errors_first_docs(metrics, budget):
    errors = [m for m in metrics if not m["correct"]]
    corrects = [m for m in metrics if m["correct"]]
    selected = [m["doc_idx"] for m in errors[:budget]]
    remaining = budget - len(selected)
    if remaining > 0 and len(corrects) > 0:
        corrects_sorted = sorted(corrects, key=lambda m: -m["margin"])
        selected.extend([m["doc_idx"] for m in corrects_sorted[:remaining]])
    return selected


def sample_by_size_docs(metrics, budget, buckets=3, key="num_nodes", seed=42):
    vals = torch.tensor([m[key] for m in metrics], dtype=torch.float)
    if buckets <= 1 or len(metrics) == 0:
        return sample_random_docs(metrics, budget, seed=seed)
    q = torch.quantile(vals, torch.linspace(0, 1, buckets + 1))
    groups = [[] for _ in range(buckets)]
    for i, v in enumerate(vals):
        b = int(torch.clamp(((v >= q[:-1]) & (v <= q[1:])).nonzero(as_tuple=False).flatten(), 0, buckets - 1)[0].item())
        groups[b].append(metrics[i])
    per = max(1, budget // buckets)
    g = torch.Generator().manual_seed(seed)
    selected = []
    for grp in groups:
        if len(grp) == 0:
            continue
        order = torch.randperm(len(grp), generator=g)[:min(per, len(grp))].tolist()
        selected.extend([grp[i]["doc_idx"] for i in order])
    if len(selected) < budget:
        rest = [m["doc_idx"] for m in metrics if m["doc_idx"] not in set(selected)]
        selected.extend(rest[: budget - len(selected)])
    return selected[:budget]


def encode_nodes_from_sentence_encoder(model, tokenized) -> torch.Tensor:
    """
    Build node features [N, H] from model.encoder (CLS embeddings), detached and requires_grad for IG/GNNExplainer.
    """
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise AttributeError("Model has no 'encoder' attribute.")
    hf_model = getattr(enc, "model", enc)
    x = None
    if isinstance(hf_model, UniGraph):
        hf_model = hf_model.lm_encoder
        out = hf_model(
            **tokenized,
            output_hidden_states=True,
            return_dict=True,
        )
        out = out.hidden_states[-1][:, 0, :]  
        # print(out.size())
        x = out.detach().requires_grad_()
    else:
        out = hf_model(
            tokenized_inputs=tokenized,
            return_token_grad=False,
        )
        x = out[0].detach().requires_grad_()  # leaf tensor with grad
    return x


class TokenBatchWrapper(nn.Module):
    """
    forward(x, edge_index, batch) -> logits
    - x must be a Tensor for Explainer.device, but is ignored; we use stored tokenized.
    - Use when explaining edges (node_mask_type=None).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self._tokenized = None

    def set_tokenized(self, tokenized):
        self._tokenized = tokenized

    def forward(self, x, edge_index, batch):
        assert self._tokenized is not None, "Call set_tokenized(tokenized) before explaining."
        logits, _, _ = self.model(tokenized_inputs=self._tokenized, edge_index=edge_index, batch_ids=batch)
        return logits


class NodeFeatureWrapper(nn.Module):
    """
    forward(x, edge_index, batch) -> logits using x as node features.
    - Runs model's GNN stack (if exposed) + pool + classifier.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self.gnn = None
        for name in ["gcn","gnn", "convs", "gat_layers", "sage_layers", "gcn_layers", "gnn_encoder"]:
            mod = getattr(base_model, name, None)
            if isinstance(mod, (nn.ModuleList, nn.Sequential, nn.Module)):
                self.gnn = mod
                print(f"[NodeFeatureWrapper] Using '{name}' as GNN module.")
                break
        self.classifier = getattr(base_model, "classifier", None)
        self.num_classes = getattr(base_model, "num_classes", 2)
        self._fallback_head = None

    def _run_gnn(self, x, edge_index):
        if self.gnn is None:
            return x
        layers = list(self.gnn) if isinstance(self.gnn, (nn.ModuleList, nn.Sequential)) else [self.gnn]
        h = x
        for layer in layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        return h

    def _classify(self, graph_emb):
        if self.classifier is not None:
            return self.classifier(graph_emb)
        if self._fallback_head is None:
            in_dim = graph_emb.size(-1)
            hid = max(64, in_dim // 2)
            self._fallback_head = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, self.num_classes)).to(graph_emb.device)
        return self._fallback_head(graph_emb)

    def forward(self, x, edge_index, batch):
        assert torch.is_tensor(x), "x must be node features [N, H]"
        h = self._run_gnn(x, edge_index)
        if batch is None or batch.numel() != h.size(0):
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        g = global_mean_pool(h, batch)
        return self._classify(g)


def explain_per_graph(
    model,
    dataset: List[Dict[str, Any]],
    tokenizer,
    explainer: Explainer,
    mode: str,  # "edge" or "node"
    label_encoder: ClassLabel,
    task_label_key: str,
    use_ground_truth_target: bool,
    device: torch.device,
    out_jsonl: Optional[str],
    topk_edges: int = 10,
):
    wrapped = explainer.model
    fout = open(out_jsonl, "w") if out_jsonl else None
    results = []

    for doc_idx, doc in enumerate(dataset):
        sentences = doc["sentences"]
        edge_idx = doc["edge_index"]
        if not torch.is_tensor(edge_idx):
            edge_idx = torch.tensor(edge_idx, dtype=torch.long)
        edge_idx = edge_idx.to(device)
        batch_vec = torch.zeros(len(sentences), dtype=torch.long, device=device)

        tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits, _, _ = model(tokenized, edge_idx, batch_ids=batch_vec)

        if use_ground_truth_target:
            y = doc.get(task_label_key, None)
            if isinstance(y, list):
                y = y[0]
            target_label = label_encoder.str2int(y) if isinstance(y, str) else int(y)
        else:
            target_label = int(logits.argmax(dim=-1).item() if logits.dim() > 1 else logits.argmax().item())
        target_vec = torch.tensor([target_label], dtype=torch.long, device=device)

        if mode == "edge":
            wrapped.set_tokenized(tokenized)
            x_for_device = tokenized["input_ids"]  # any tensor on device
            explanation = explainer(x=x_for_device, edge_index=edge_idx, target=target_vec, index=0, batch=batch_vec)
            edge_mask = explanation.get("edge_mask")
            node_mask = explanation.get("node_mask")
        else:
            x_nodes = encode_nodes_from_sentence_encoder(model, tokenized)  # [N, H], requires_grad
            explanation = explainer(x=x_nodes, edge_index=edge_idx, target=target_vec, index=0, batch=batch_vec)
            edge_mask = explanation.get("edge_mask")
            node_mask = explanation.get("node_mask")

        # 3) Optionally, evaluate explanation quality: unfaithfulness and fidelity
        unfaithfulness_score = unfaithfulness(explainer, explanation)
        pos_fid, neg_fid = fidelity(explainer, explanation)
        print(f"Unfaithfulness score: {unfaithfulness_score}")
        print(f"Fidelity score: pos {pos_fid} neg {neg_fid}")

        record = {
            "doc_index": doc_idx,
            "doc_id": doc.get("id", None),
            "sentences": sentences,
            "target_label": int(target_label),
            "prediction": int(logits.argmax(dim=-1).item() if logits.dim() > 1 else logits.argmax().item()),
            "unfaithfulness": float(unfaithfulness_score),
            "fidelity_pos": pos_fid,
            "fidelity_neg": neg_fid,
            # save num nodes/edges
            "num_nodes": edge_idx.size(0),
            "num_edges": edge_idx.size(1),
            # save the edge_index as list of lists
            "edge_index": edge_idx.detach().cpu().tolist(),
        }

        if node_mask is not None:
            nm = node_mask.detach().cpu().view(-1)
            vals, idxs = torch.topk(nm, k=min(10, nm.numel()))
            record["node_mask"] = [float(x) for x in nm.tolist()]
            record["top_nodes"] = [{"node": int(idxs[i]), "importance": float(vals[i])} for i in range(vals.numel())]

        if edge_mask is not None:
            em = edge_mask.detach().cpu().view(-1)
            sub_ei = explanation.edge_index.detach().cpu()
            row, col = sub_ei[0].tolist(), sub_ei[1].tolist()
            k = min(topk_edges, em.numel())
            top_vals, top_idx = torch.topk(em, k=k)
            top_edges = []
            for r in range(k):
                e = int(top_idx[r].item())
                top_edges.append({"u": int(row[e]), "v": int(col[e]), "importance": float(top_vals[r].item())})
            record["edge_index"] = list(zip(row, col))
            record["edge_mask"] = [float(x) for x in em.tolist()]
            record["top_edges"] = top_edges

        print(f"[Explain] doc={doc_idx} tgt={record['target_label']} pred={record['prediction']} "
              f"nodes={len(sentences)} edges={edge_idx.size(1)}")
        if "top_edges" in record:
            for te in record["top_edges"][:5]:
                print(f"  ({te['u']}, {te['v']}) -> {te['importance']:.4f}")
        if "top_nodes" in record:
            for tn in record["top_nodes"][:5]:
                print(f"  node {tn['node']} -> {tn['importance']:.4f}")

        if fout:
            fout.write(json.dumps(record) + "\n")
            fout.flush()
        results.append(record)

    if fout:
        fout.close()
    return results


def parse_args():
    p = argparse.ArgumentParser("Per-graph classifier explainer")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    p.add_argument("--model_type", type=str, default="SentenceGGraphClassifier",
                   choices=[
                       "SentenceGGraphClassifier",
                       "SentenceGATGraphClassifier",
                       "SentenceSAGEGraphClassifier",
                       "SentenceDGConvGraphClassifier",
                       "SentenceDGATGraphClassifier",
                       "SentenceDGSAGEGraphClassifier",
                       "UniGraphGGraphClassifier",
                       "SentenceGraphPromptClassifier",
                       "SentenceGATGraphPromptClassifier",
                       "SentenceSAGEGraphPromptClassifier",
                   ])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--task_label_key", type=str, default="graph_label")
    p.add_argument("--encoder_grad", action="store_true", default=True)

    # Explainer options
    p.add_argument("--explain_mode", type=str, default="edge", choices=["edge", "node"],
                   help="'edge' -> edge attributions; 'node' -> node attribute attributions")
    p.add_argument("--explainer_name", type=str, default="GNNExplainer",
                   choices=["GNNExplainer", "CaptumExplainer"])
    p.add_argument("--gnnexplainer_epochs", type=int, default=200)
    p.add_argument("--gnnexplainer_lr", type=float, default=0.01)

    # Split + sampling
    p.add_argument("--target_split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--sample_strategy", type=str, default="none",
                   choices=["none", "random", "stratified", "confidence", "errors", "size", "first_n"])
    p.add_argument("--sample_budget", type=int, default=0)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--sample_per_class", type=int, default=10)
    p.add_argument("--sample_low_k", type=int, default=20)
    p.add_argument("--sample_high_k", type=int, default=20)
    p.add_argument("--sample_size_buckets", type=int, default=3)

    p.add_argument("--use_ground_truth_target", action="store_true")
    p.add_argument("--output_jsonl", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--num_debug_graphs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("explainer")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load datasets
    def accept_all(_): return True
    train_dataset = get_graph_dataset(args.dataset_name, args.dataset_path, "train",
                                      model_name=args.model_name, label_key=args.task_label_key,
                                      padding="max_length", filter_function=accept_all, negative_sampling=False)
    val_dataset = get_graph_dataset(args.dataset_name, args.dataset_path, "val",
                                    model_name=args.model_name, label_key=args.task_label_key,
                                    padding="max_length", filter_function=accept_all, negative_sampling=False)
    test_dataset = get_graph_dataset(args.dataset_name, args.dataset_path, "test",
                                     model_name=args.model_name, label_key=args.task_label_key,
                                     padding="max_length", filter_function=accept_all, negative_sampling=False)
    logger.info(f"Loaded datasets: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    # Labels
    all_datasets = train_dataset + val_dataset + test_dataset
    label_encoder = build_label_encoder(all_datasets, args.task_label_key)
    num_class = len(label_encoder.names)
    logger.info(f"Num classes: {num_class} -> {label_encoder.names}")

    # Model
    model_cfg = {"encoder_grad": args.encoder_grad, "num_classes": num_class}
    if args.model_type in ["SentenceDGConvGraphClassifier", "SentenceDGATGraphClassifier", "SentenceDGSAGEGraphClassifier"]:
        model_cfg["num_layers"] = 3
        model_cfg["hidden_channels"] = 64
    if args.model_type == "UniGraphGGraphClassifier":
        model_cfg["embedding_choice"] = "graph"
        model_cfg["gradient_mode"] = True
    # if model_type is from graph_prompt set freeze_backbone to false to allow gradients
    if "Prompt" in args.model_type:
        model_cfg["freeze_backbone"] = False

    model_cls = model_mapper(args.model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    model = model_cls(encoder_model=args.model_name, **model_cfg).to(device)
    tokenizer = model.encoder.tokenizer

    if args.checkpoint is not None:
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # Optional quick eval
    # You can wire val_loader if needed. Skipped to keep script minimal.

    # Choose split
    target_dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}[args.target_split]
    if args.debug:
        target_dataset = random.sample(target_dataset, min(args.num_debug_graphs, len(target_dataset)))

    # Sampling
    selected_doc_indices = None
    if args.sample_strategy != "none" and args.sample_budget > 0:
        doc_metrics = collect_doc_metrics(model, target_dataset, tokenizer, label_encoder,
                                          args.task_label_key, device)
        if args.sample_strategy == "random":
            selected_doc_indices = sample_random_docs(doc_metrics, args.sample_budget, seed=args.sample_seed)
        elif args.sample_strategy == "stratified":
            selected_doc_indices = sample_stratified_by_true_docs(doc_metrics, per_class=args.sample_per_class, seed=args.sample_seed)
        elif args.sample_strategy == "confidence":
            selected_doc_indices = sample_confidence_mix_docs(doc_metrics, low_k=args.sample_low_k, high_k=args.sample_high_k)
            selected_doc_indices = selected_doc_indices[:args.sample_budget]
        elif args.sample_strategy == "errors":
            selected_doc_indices = sample_errors_first_docs(doc_metrics, budget=args.sample_budget)
        elif args.sample_strategy == "size":
            selected_doc_indices = sample_by_size_docs(doc_metrics, budget=args.sample_budget,
                                                       buckets=args.sample_size_buckets, key="num_nodes", seed=args.sample_seed)
        elif args.sample_strategy == "first_n":
            selected_doc_indices = list(range(min(args.sample_budget, len(target_dataset))))
        # Narrow dataset to sampled docs
        idx_set = set(selected_doc_indices)
        target_dataset = [d for i, d in enumerate(target_dataset) if i in idx_set]
        logger.info(f"Sampling selected {len(target_dataset)} graphs to explain.")

    # Build explainer
    if args.explain_mode == "edge":
        wrapped = TokenBatchWrapper(model).to(device).eval()
        explainer = Explainer(
            model=wrapped,
            algorithm=GNNExplainer(epochs=args.gnnexplainer_epochs, lr=args.gnnexplainer_lr),
            explanation_type="phenomenon",
            node_mask_type=None,
            edge_mask_type="object",  # mask edges
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )
    else:
        wrapped = NodeFeatureWrapper(model).to(device).eval()
        if args.explainer_name == "CaptumExplainer":
            algorithm = CaptumExplainer("IntegratedGradients")
        else:
            algorithm = GNNExplainer(epochs=args.gnnexplainer_epochs, lr=args.gnnexplainer_lr)
        explainer = Explainer(
            model=wrapped,
            algorithm=algorithm,
            explanation_type="phenomenon",
            node_mask_type="attributes",  # mask node features
            edge_mask_type=None,
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

    # Explain per-graph
    results = explain_per_graph(
        model=model,
        dataset=target_dataset,
        tokenizer=tokenizer,
        explainer=explainer,
        mode=args.explain_mode,
        label_encoder=label_encoder,
        task_label_key=args.task_label_key,
        use_ground_truth_target=args.use_ground_truth_target,
        device=device,
        out_jsonl=args.output_jsonl,
        topk_edges=10,
    )
    logging.getLogger("explainer").info(f"Explaining done, total graphs explained: {len(results)}")


if __name__ == "__main__":
    main()
