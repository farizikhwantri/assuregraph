import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ModelConfig

from graph_model import create_document_graph  # import your example builder
from unigraph import UniGraphGraphClassifier

from pipeline import get_graph_dataset
from utils_cli import parse_args


def build_subgraphs_from_document(doc: Dict[str, Any], hop: int = 1) -> List[Dict[str, Any]]:
    """
    Build 1-hop (or k-hop) induced subgraphs around every node.
    Label a subgraph 1 if it contains any positive labeled edge from doc["edge_pairs"]/doc["labels"], else 0.
    """
    sentences = doc["sentences"]
    edge_index = doc["edge_index"]  # [2, E]
    edge_pairs = doc["edge_pairs"]  # [M, 2]
    labels = doc["labels"].float()  # [M]

    pos_edges = set()
    if edge_pairs is not None and labels is not None:
        for (u, v), y in zip(edge_pairs.tolist(), labels.tolist()):
            if int(y) == 1:
                pos_edges.add((u, v))
                pos_edges.add((v, u))  # treat as undirected presence for labeling

    N = len(sentences)
    subgraphs = []
    for seed in range(N):
        nodes, sub_ei, _, node_map = k_hop_subgraph(
            seed, hop, edge_index, relabel_nodes=True, num_nodes=N
        )
        # Build subgraph sentences and edge_index in local node ids
        sub_sentences = [sentences[int(n)] for n in nodes.tolist()]
        # Label subgraph: positive if any edge in original indices within nodes is positive
        y = 0
        for e in sub_ei.t().tolist():
            u_local, v_local = int(e[0]), int(e[1])
            u, v = int(nodes[u_local]), int(nodes[v_local])
            if (u, v) in pos_edges:
                y = 1
                break

        subgraphs.append({
            "sentences": sub_sentences,
            "edge_index": sub_ei,   # [2, e_sub]
            "label": torch.tensor(y, dtype=torch.long),
        })
    return subgraphs


def collate_subgraphs(
    subs: List[Dict[str, Any]],
    tokenizer,
    device: torch.device
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int, int, int]]]:
    """
    Collate a list of subgraphs into a single batch for UniGraphGraphClassifier.
    Returns:
      tokenized_inputs, batch_edge_index, batch_vec, labels, and spans per subgraph
      spans[i] = (node_start, node_end, edge_start, edge_end) to help map explanations.
    """
    all_sentences = []
    all_edges = []
    batch_vec = []
    labels = []
    spans = []

    node_offset = 0
    edge_offset = 0
    for i, sg in enumerate(subs):
        sents = sg["sentences"]
        ei = sg["edge_index"]
        y = sg["label"]
        num_nodes = len(sents)
        num_edges = ei.size(1)

        # accumulate sentences and labels
        all_sentences.extend(sents)
        labels.append(y)

        # offset edge_index and collect
        if num_edges > 0:
            all_edges.append(ei + node_offset)
        # batch vector for nodes in this subgraph
        batch_vec.append(torch.full((num_nodes,), i, dtype=torch.long))

        # track spans
        spans.append((node_offset, node_offset + num_nodes, edge_offset, edge_offset + num_edges))
        node_offset += num_nodes
        edge_offset += num_edges

    # tokenize concatenated sentences
    tokenized = tokenizer(
        all_sentences, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # build batched edge_index
    if len(all_edges) > 0:
        batch_edge_index = torch.cat(all_edges, dim=1).to(device)
    else:
        batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    batch_vec = torch.cat(batch_vec, dim=0).to(device)
    labels = torch.stack(labels, dim=0).to(device)

    return tokenized, batch_edge_index, batch_vec, labels, spans


class UniGraphSubgraphClassifier(nn.Module):
    """
    Thin wrapper around UniGraphGraphClassifier to expose PyG Explainer-friendly shape.
    forward(x, edge_index, batch) -> logits [num_graphs, num_classes]
    The model ignores x because it builds features from tokenized inputs.
    """
    def __init__(self, base_model: UniGraphGraphClassifier):
        super().__init__()
        self.model = base_model

    def forward(self, x, edge_index, batch):
        out = self.model(
            tokenized_inputs=x,
            edge_index=edge_index,
            batch_ids=batch
        )
        # out["logits"]: #[num_graphs, num_classes]
        return out["logits"]


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build a toy document (reuse your helper from graph_model.py)
    doc = create_document_graph(
        sentences=[
            "Healthcare organizations must implement access controls.",
            "Patient data requires encryption at rest.",
            "Audit logs must be maintained for security events.",
            "Administrative safeguards protect electronic health information."
        ],
        edge_index=torch.tensor([[0, 1, 2, 0, 3],
                                 [1, 2, 3, 3, 0]], dtype=torch.long),
        edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]]),
        node_labels=torch.tensor([2, 1, 1, 2], dtype=torch.float),
        labels=torch.tensor([1, 1, 1, 1, 0], dtype=torch.float),
    )

    # 2) Build 1-hop subgraphs and labels
    subgraphs = build_subgraphs_from_document(doc, hop=1)

    # 3) Instantiate graph classifier
    clf = UniGraphGraphClassifier(
        lm_type="bert-base-uncased",
        hidden_size=768,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        lam=0.0,
        num_classes=2,
        embedding_choice="graph"
    ).to(device)

    # 4) Collate subgraphs into a batch
    tokenized, batch_edge_index, batch_vec, labels, spans = collate_subgraphs(
        subgraphs, clf.tokenizer, device
    )

    # 5) Train briefly
    optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-4, weight_decay=0.01)
    clf.train()
    for epoch in range(2):
        out = clf(
            tokenized_inputs=tokenized,
            edge_index=batch_edge_index,
            batch_ids=batch_vec,
            labels=labels
        )
        if out["loss"] is None:
            raise RuntimeError("Classifier did not return loss; labels were not passed correctly.")
        loss = out["loss"]
        # logits = out[0]
        # loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # preds = logits.argmax(dim=-1)
            preds = out["logits"].argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        print(f"[Epoch {epoch+1}] loss={loss.item():.4f} acc={acc:.4f}")

    # 6) Wrap classifier for PGExplainer (graph-level explanations)
    clf.eval()
    wrapped = UniGraphSubgraphClassifier(clf).to(device).eval()
    explainer = Explainer(
        model=wrapped,
        algorithm=PGExplainer(epochs=30, lr=0.01),
        explanation_type="phenomenon",   # required for PGExplainer
        # explanation_type="model",
        node_mask_type=None,             # PGExplainer learns edge masks
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",  # or "binary_classification"
            task_level="graph",
            return_type="raw",                  # wrapped returns logits
        ),
    )

    # 6.1) Train the PGExplainer parameters on the current batch (phenomenon = model prediction)
    with torch.no_grad():
        batch_outs = wrapped(tokenized, batch_edge_index, batch_vec)
        # batch_logits = batch_outs["logits"]  # [num_graphs, num_classes]
        batch_logits = batch_outs  # [num_graphs, num_classes]
        targets = batch_logits.argmax(dim=-1)
    graph_indices = torch.arange(targets.size(0), device=device)
    explainer.algorithm.train(
        model=wrapped,
        x=tokenized,
        edge_index=batch_edge_index,
        target=targets,
        index=graph_indices,
        batch=batch_vec,
        epoch=30,
    )

    # 7) Explain one subgraph (index into the batch)
    target_idx = 0
    target_label = int(labels[target_idx].item())
    explanation = explainer(
        x=tokenized,                                        # tokenized inputs for all nodes in batch
        edge_index=batch_edge_index,
        target=target_label,                                  # class index to explain
        index=target_idx,                                     # graph index in batch
        batch=batch_vec
    )

    # 8) Edge explanations for the chosen subgraph (and derived node scores)
    node_start, node_end, edge_start, edge_end = spans[target_idx]
    sub_ei = subgraphs[target_idx]["edge_index"]

    edge_mask = explanation.edge_mask[edge_start:edge_end].detach().cpu()  # [E_sub]
    print(f"Explaining subgraph {target_idx} (label={target_label})")
    print("Top edges (u, v, importance):")
    topk_edges = torch.topk(edge_mask, k=min(10, edge_mask.numel()))
    for i, s in zip(topk_edges.indices.tolist(), topk_edges.values.tolist()):
        u_local = int(sub_ei[0, i]); v_local = int(sub_ei[1, i])
        print(f"  ({u_local}, {v_local}) -> {s:.4f}")

    # Derive node importance by summing incident edge scores
    num_nodes_sub = node_end - node_start
    node_imp = torch.zeros(num_nodes_sub)
    for e_idx in range(edge_mask.numel()):
        u = int(sub_ei[0, e_idx]); v = int(sub_ei[1, e_idx])
        s = float(edge_mask[e_idx])
        node_imp[u] += s; node_imp[v] += s

    print("Top nodes (local idx, derived importance):")
    topk_nodes = torch.topk(node_imp, k=min(10, node_imp.numel()))
    for i, s in zip(topk_nodes.indices.tolist(), topk_nodes.values.tolist()):
        print(f"  node {i} -> {s:.4f} | text='{subgraphs[target_idx]['sentences'][i]}'")


if __name__ == "__main__":
    test()