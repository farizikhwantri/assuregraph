from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from graph_model import SentenceEncoder
from graph_model import SentenceGraphLinkPredictor
from graph_model import create_document_graph  # reuse helper

class SentenceGraphPromptClassifier(nn.Module):
    """
    GraphPrompt-style classifier:
      - Backbone = SentenceEncoder + GNN (same as link predictor)
      - Prompts = learnable class prototypes in the hidden_dim space
      - Logits = temperature-scaled cosine similarity between graph embeddings and prompts
    Interface matches SentenceGGraphClassifier: forward(...) -> (logits, node_features, edge_pairs_features).
    """
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        GNN=GCNConv,
        pooling: str = "mean",            # "mean" | "sum" | "max"
        temperature: float = 0.07,
        normalize: bool = True,
        freeze_backbone: bool = True,     # freeze encoder+GNN, train prompts only
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooling = pooling.lower()
        self.temperature = temperature
        self.normalize = normalize
        self.freeze_backbone = freeze_backbone

        # Backbone matches SentenceGraphLinkPredictor/SentenceGGraphClassifier
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size
        self.gcn = GNN(in_dim, hidden_dim)

        # Learnable class prompts (prototypes)
        self.prompts = nn.Parameter(torch.randn(num_classes, hidden_dim))

        if self.freeze_backbone:
            self._set_backbone_trainable(False)

    def _set_backbone_trainable(self, flag: bool):
        for p in self.encoder.parameters():
            p.requires_grad = flag
        # GNN may be a ModuleList or a single layer; handle both
        if isinstance(self.gcn, ModuleList):
            for m in self.gcn:
                for p in m.parameters():
                    p.requires_grad = flag
        else:
            for p in self.gcn.parameters():
                p.requires_grad = flag

    def _pool_graph(self, x: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        if batch_ids is None:
            batch_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if self.pooling == "sum":
            return global_add_pool(x, batch_ids)
        if self.pooling == "max":
            return global_max_pool(x, batch_ids)
        return global_mean_pool(x, batch_ids)

    @torch.no_grad()
    def init_prompts_from_support(
        self,
        tokenized_inputs,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        batch_ids: torch.Tensor = None,
    ):
        """
        Initialize prompts as per-class means of graph embeddings from a labeled support set.
        labels: [B] graph class ids for each item in the batch.
        """
        self.eval()
        x = self.encoder(tokenized_inputs)[0]
        x = F.relu(self.gcn(x, edge_index))
        g = self._pool_graph(x, batch_ids)  # [B, hidden]
        C = self.num_classes
        proto = []
        for c in range(C):
            mask = (labels == c)
            if mask.any():
                proto.append(g[mask].mean(dim=0))
            else:
                proto.append(torch.randn_like(g[0]))
        proto = torch.stack(proto, dim=0)
        self.prompts.data.copy_(proto)

    def load_from_link_predictor(
        self,
        src,                      # checkpoint path, nn.Module, or state_dict
        device: str = "cpu",
        strict_shapes: bool = True,
        verbose: bool = True,
        copy_encoder: bool = True,
        copy_gnn: bool = True,
    ):
        """
        Copy encoder/gcn weights from a SentenceGraphLinkPredictor (or compatible).
        """
        if isinstance(src, str):
            ckpt = torch.load(src, map_location=device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        elif isinstance(src, nn.Module):
            sd = src.state_dict()
        elif isinstance(src, dict):
            sd = src
        else:
            raise ValueError("Unsupported src input for load_from_link_predictor.")

        tgt_sd = self.state_dict()
        moved = 0
        skipped = []

        def copy_block(prefix: str):
            nonlocal moved
            for k, v in sd.items():
                if not k.startswith(prefix):
                    continue
                if k in tgt_sd:
                    if (not strict_shapes) or (tgt_sd[k].shape == v.shape):
                        tgt_sd[k] = v
                        moved += 1
                    else:
                        skipped.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))

        if copy_encoder:
            copy_block("encoder.encoder.")
        if copy_gnn:
            copy_block("gcn.")

        self.load_state_dict(tgt_sd, strict=False)
        if verbose:
            print(f"[GraphPrompt] Copied {moved} params from link predictor.")
            if skipped:
                print(f"[GraphPrompt] Skipped {len(skipped)} due to shape mismatch.")

        if self.freeze_backbone:
            self._set_backbone_trainable(False)

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None, batch_ids=None):
        """
        Returns:
          - logits: [B, C] if batched, else [C]
          - node_features: [N, hidden_dim]
          - edge_pairs_features: [E, 2*hidden_dim]
        """
        # Backbone features
        x = self.encoder(tokenized_inputs)[0]          # [N, in_dim]
        x = F.relu(self.gcn(x, edge_index))            # [N, hidden_dim]

        # Edge-pair features for compatibility with other heads
        if edge_pairs is None:
            edge_pairs = edge_index.t()
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        edge_pairs_features = torch.cat([src, tgt], dim=1)

        # Graph embeddings
        g = self._pool_graph(x, batch_ids)             # [B, hidden_dim] or [1, hidden_dim]

        # Normalize and compute similarities with prompts
        P = self.prompts
        if self.normalize:
            g = F.normalize(g, dim=-1)
            P = F.normalize(P, dim=-1)
        logits = (g @ P.t()) / self.temperature        # [B, C]

        if batch_ids is None:
            logits = logits.squeeze(0)

        return logits, x, edge_pairs_features


class SentenceGATGraphPromptClassifier(SentenceGraphPromptClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean",
        temperature: float = 0.07,
        normalize: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=GATConv,
            pooling=pooling,
            temperature=temperature,
            normalize=normalize,
            freeze_backbone=freeze_backbone,
        )


class SentenceSAGEGraphPromptClassifier(SentenceGraphPromptClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean",
        temperature: float = 0.07,
        normalize: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=SAGEConv,
            pooling=pooling,
            temperature=temperature,
            normalize=normalize,
            freeze_backbone=freeze_backbone,
        )

class SentenceDGConvGraphPromptClassifier(nn.Module):
    """
    Deep-GNN GraphPrompt classifier:
      - Backbone: SentenceEncoder + stacked GNN layers (like SentenceDGCNN)
      - Node features: concat of intermediate GNN features (final_latent_dim = hidden_channels * num_layers + 1)
      - Prompts: learnable class prototypes in final_latent_dim space
      - Logits: temperature-scaled cosine similarity between pooled graph embeddings and prompts
    Interface matches graph classifiers: forward(...) -> (logits, node_features, edge_pairs_features)
    """
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        GNN=GCNConv,
        pooling: str = "mean",      # "mean" | "sum" | "max"
        temperature: float = 0.07,
        normalize: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooling = pooling.lower()
        self.temperature = temperature
        self.normalize = normalize
        self.freeze_backbone = freeze_backbone

        # Encoder
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size

        # Deep GNN (ModuleList) like SentenceDGCNN
        self.gcn = ModuleList()
        self.gcn.append(GNN(in_dim, hidden_channels))
        for _ in range(0, num_layers - 1):
            self.gcn.append(GNN(hidden_channels, hidden_channels))
        self.gcn.append(GNN(hidden_channels, 1))
        self.final_latent_dim = hidden_channels * num_layers + 1

        # Prompts (class prototypes)
        self.prompts = nn.Parameter(torch.randn(num_classes, self.final_latent_dim))

        if self.freeze_backbone:
            self._set_backbone_trainable(False)

    def _set_backbone_trainable(self, flag: bool):
        for p in self.encoder.parameters():
            p.requires_grad = flag
        for layer in self.gcn:
            for p in layer.parameters():
                p.requires_grad = flag

    def _pool_graph(self, x: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        if batch_ids is None:
            batch_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if self.pooling == "sum":
            return global_add_pool(x, batch_ids)
        if self.pooling == "max":
            return global_max_pool(x, batch_ids)
        return global_mean_pool(x, batch_ids)

    @torch.no_grad()
    def init_prompts_from_support(self, tokenized_inputs, edge_index, labels: torch.Tensor, batch_ids: torch.Tensor = None):
        """
        Initialize prompts using per-class means of graph embeddings from a labeled support batch.
        labels: [B] class ids for graphs in the batch
        """
        self.eval()
        enc_out = self.encoder(tokenized_inputs)
        x0 = enc_out[0]
        xs = [x0]
        for conv in self.gcn:
            xs.append(conv(xs[-1], edge_index).tanh())
        x = torch.cat(xs[1:], dim=-1)  # [N, final_latent_dim]

        g = self._pool_graph(x, batch_ids)  # [B, final_latent_dim]
        C = self.num_classes
        proto = []
        for c in range(C):
            m = (labels == c)
            if m.any():
                proto.append(g[m].mean(dim=0))
            else:
                proto.append(torch.randn_like(g[0]))
        proto = torch.stack(proto, dim=0)
        self.prompts.data.copy_(proto)

    def load_from_link_predictor(
        self,
        src,                      # checkpoint path, nn.Module, or state_dict
        device: str = "cpu",
        strict_shapes: bool = True,
        verbose: bool = True,
        copy_encoder: bool = True,
        copy_gnn: bool = True,
    ):
        """
        Copy encoder + deep GNN weights from a SentenceDGCNN (or compatible link predictor).
        """
        if isinstance(src, str):
            ckpt = torch.load(src, map_location=device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        elif isinstance(src, nn.Module):
            sd = src.state_dict()
        elif isinstance(src, dict):
            sd = src
        else:
            raise ValueError("Unsupported src input for load_from_link_predictor.")

        tgt_sd = self.state_dict()
        moved, skipped = 0, []

        def copy_block(prefix: str):
            nonlocal moved
            for k, v in sd.items():
                if not k.startswith(prefix):
                    continue
                if k in tgt_sd:
                    if (not strict_shapes) or (tgt_sd[k].shape == v.shape):
                        tgt_sd[k] = v
                        moved += 1
                    else:
                        skipped.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))

        if copy_encoder:
            copy_block("encoder.encoder.")
        if copy_gnn:
            copy_block("gcn.")

        self.load_state_dict(tgt_sd, strict=False)
        if verbose:
            print(f"[DG-GraphPrompt] Copied {moved} params from link predictor.")
            if skipped:
                print(f"[DG-GraphPrompt] Skipped {len(skipped)} due to shape mismatch.")

        if self.freeze_backbone:
            self._set_backbone_trainable(False)

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None, batch_ids=None):
        """
        Returns:
          - logits: [B, C] (or [C] if single graph without batch_ids)
          - node_features: [N, final_latent_dim]
          - edge_pairs_features: [E, 2*final_latent_dim]
        """
        enc_out = self.encoder(tokenized_inputs)
        x0 = enc_out[0]
        xs = [x0]
        for conv in self.gcn:
            xs.append(conv(xs[-1], edge_index).tanh())
        x = torch.cat(xs[1:], dim=-1)  # [N, final_latent_dim]

        # Edge pair features (compatibility)
        if edge_pairs is None:
            edge_pairs = edge_index.t()
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        edge_pairs_features = torch.cat([src, tgt], dim=1)

        # Pool to graph embeddings
        g = self._pool_graph(x, batch_ids)  # [B, final_latent_dim]

        # Prompt similarities
        P = self.prompts
        if self.normalize:
            g = F.normalize(g, dim=-1)
            P = F.normalize(P, dim=-1)
        logits = (g @ P.t()) / self.temperature

        if batch_ids is None:
            logits = logits.squeeze(0)

        return logits, x, edge_pairs_features


class SentenceDGATGraphPromptClassifier(SentenceDGConvGraphPromptClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean",
        temperature: float = 0.07,
        normalize: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=GATConv,
            pooling=pooling,
            temperature=temperature,
            normalize=normalize,
            freeze_backbone=freeze_backbone,
        )


class SentenceDGSAGEGraphPromptClassifier(SentenceDGConvGraphPromptClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean",
        temperature: float = 0.07,
        normalize: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=SAGEConv,
            pooling=pooling,
            temperature=temperature,
            normalize=normalize,
            freeze_backbone=freeze_backbone,
        )

def load_test(link_predictor_path: str):
    # Load link predictor (already trained)
    lp = SentenceGraphLinkPredictor(hidden_dim=256, encoder_model="bert-base-uncased")
    lp.load_state_dict(torch.load(link_predictor_path)["model_state_dict"])

    # Build GraphPrompt classifier and copy backbone weights
    gp = SentenceGraphPromptClassifier(num_classes=2, hidden_dim=256, freeze_backbone=True)
    gp.load_from_link_predictor(lp, device="cpu")

    # Train only gp.prompts with cross-entropy on graph labels
    for p in gp.parameters():
        p.requires_grad = p.requires_grad  # prompts True, backbone False
    
    # Load link predictor (already trained)
    lp = SentenceGraphLinkPredictor(hidden_dim=256, encoder_model="bert-base-uncased")
    lp.load_state_dict(torch.load(link_predictor_path)["model_state_dict"])

    # Build GraphPrompt classifier and copy backbone weights
    gp = SentenceGraphPromptClassifier(num_classes=2, hidden_dim=256, freeze_backbone=True)
    gp.load_from_link_predictor(lp, device="cpu")

    # Train only gp.prompts with cross-entropy on graph labels
    for p in gp.parameters():
        p.requires_grad = p.requires_grad  # prompts True, backbone False

# ...existing classes (SentenceGraphPromptClassifier, variants)...

def _pack_batch(
    docs: List[Dict],
    tokenizer,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack multiple document graphs into a single batch:
      - tokenized_inputs: HF tensors for all sentences concatenated
      - edge_index: [2, E_tot] with node indices shifted by cumulative offsets
      - batch_ids: [N_tot] mapping each node to its graph id (0..B-1)
      - graph_labels: [B] (from doc['graph_label'])
    """
    all_sentences: List[str] = []
    shifts: List[int] = []
    total = 0
    for d in docs:
        n = len(d["sentences"])
        shifts.append(total)
        total += n
        all_sentences.extend(d["sentences"])

    # Build concatenated edge_index and batch vector
    ei_list = []
    batch_vec = torch.zeros(total, dtype=torch.long)
    for gid, d in enumerate(docs):
        shift = shifts[gid]
        ei = d["edge_index"].clone().long()
        ei = ei + shift  # shift both rows
        ei_list.append(ei)
        # fill batch ids for nodes of this graph
        n = len(d["sentences"])
        batch_vec[shift:shift + n] = gid

    if ei_list:
        edge_index = torch.cat(ei_list, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Tokenize all sentences at once
    toks = tokenizer(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    return toks, edge_index.to(device), batch_vec.to(device), torch.tensor([d["graph_label"] for d in docs], dtype=torch.long, device=device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build two toy documents (graph_label: 0 vs 1)
    documents = [
        create_document_graph(
            sentences=[
                "The cat sat on the mat.",
                "It was a sunny day.",
                "Cats love sunlight.",
                "The mat was warm."
            ],
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 0, 3, 1]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 2], [1, 3], [2, 3]]),
            node_labels=torch.tensor([1, 0, 2, 0], dtype=torch.float),
            labels=torch.tensor([1, 0, 1], dtype=torch.float)
        ),
        create_document_graph(
            sentences=[
                "Rain causes flooding.",
                "Flooding damages houses.",
                "People must evacuate.",
                "Safety is important."
            ],
            edge_index=torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 2], [1, 3], [2, 3]]),
            node_labels=torch.tensor([1, 0, 2, 0], dtype=torch.float),
            labels=torch.tensor([1, 1, 0], dtype=torch.float)
        )
    ]
    # Assign graph-level labels (class ids)
    documents[0]["graph_label"] = 0
    documents[1]["graph_label"] = 1

    # Split few-shot support and query (here: 1 support per class, both used as query for demo)
    support_docs = [documents[0], documents[1]]
    query_docs = [documents[0], documents[1]]

    # 2) Build GraphPrompt model
    num_classes = 2
    gp = SentenceGraphPromptClassifier(
        num_classes=num_classes,
        hidden_dim=256,
        encoder_model="bert-base-uncased",
        encoder_grad=False,          # freeze encoder weights
        pooling="mean",
        temperature=0.07,
        normalize=True,
        freeze_backbone=True,        # train prompts only
    ).to(device)

    # Optional: initialize backbone from a trained link predictor checkpoint
    # ckpt_path = "checkpoints/final_checkpoint.pth"
    # if os.path.exists(ckpt_path):
    #     gp.load_from_link_predictor(ckpt_path, device=device, strict_shapes=False)

    # 3) Initialize prompts using the labeled support set
    tokenizer = gp.encoder.tokenizer
    sup_tok, sup_ei, sup_batch, sup_y = _pack_batch(support_docs, tokenizer, device=device)
    gp.init_prompts_from_support(
        tokenized_inputs=sup_tok,
        edge_index=sup_ei,
        labels=sup_y,
        batch_ids=sup_batch,
    )
    print("Initialized prompts from support set.")

    # 4) Evaluate before training (zero-shot after prompt init)
    qry_tok, qry_ei, qry_batch, qry_y = _pack_batch(query_docs, tokenizer, device=device)
    with torch.no_grad():
        logits, _, _ = gp(qry_tok, qry_ei, batch_ids=qry_batch)
        preds = logits.argmax(dim=-1)
        acc0 = (preds == qry_y).float().mean().item()
    print(f"Zero-shot (after init) accuracy: {acc0:.3f}")

    # 5) Train prompts only (few steps)
    opt = torch.optim.Adam([gp.prompts], lr=5e-2)
    ce = nn.CrossEntropyLoss()
    gp.train()
    for epoch in range(10):
        opt.zero_grad()
        logits, _, _ = gp(qry_tok, qry_ei, batch_ids=qry_batch)
        loss = ce(logits, qry_y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == qry_y).float().mean().item()
        if (epoch + 1) % 2 == 0:
            print(f"[Prompt] epoch {epoch+1:02d}  loss={loss.item():.4f}  acc={acc:.3f}")

    # 6) Final accuracy
    with torch.no_grad():
        logits, _, _ = gp(qry_tok, qry_ei, batch_ids=qry_batch)
        preds = logits.argmax(dim=-1)
        acc = (preds == qry_y).float().mean().item()
    print(f"Final accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
