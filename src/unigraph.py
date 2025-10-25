# import random
import os
import argparse
from typing import Dict, Any, Tuple, List

from types import SimpleNamespace
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM


from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from graph_model import create_document_graph
# from pipeline import get_graph_dataset

def _build_gnn_stack(in_dim: int, hidden_size: int, num_layers: int, gnn_type: str, num_heads: int, dropout: float):
    """
    Build a stack of GNN layers that outputs hidden_size at every layer:
      - GAT: uses heads=num_heads and out_channels=hidden_size // num_heads
      - GCN/SAGE: uses out_channels=hidden_size
    """
    gnn_type = gnn_type.lower()
    layers = nn.ModuleList()

    if gnn_type == "gat":
        if hidden_size % max(1, num_heads) != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads} for GAT.")
        out_per_head = hidden_size // num_heads
        for _ in range(num_layers):
            layers.append(GATConv(in_dim, out_per_head, heads=num_heads, dropout=dropout))
            in_dim = hidden_size
    elif gnn_type == "gcn":
        for _ in range(num_layers):
            layers.append(GCNConv(in_dim, hidden_size))
            in_dim = hidden_size
    elif gnn_type in ("sage", "graphsage"):
        for _ in range(num_layers):
            layers.append(SAGEConv(in_dim, hidden_size))
            in_dim = hidden_size
    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type} (use 'gat', 'gcn', or 'sage')")
    return layers


class UniGraph(nn.Module):
    """UniGraph: Learning a Unified Cross-Domain Foundation Model 
        for Text-Attributed Graphs (PyTorch Geometric Version)"""
    def __init__(self, lm_type: str, hidden_size: int, 
                 num_heads: int, num_layers: int, 
                 dropout: float, lam: float = 0.0, gnn_type: str = "gat"):
        super().__init__()
        self.lm_type = lm_type
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.lam = lam
        self.gnn_type = gnn_type.lower()

        # Language model encoder with MLM head
        self.lm_encoder = AutoModelForMaskedLM.from_pretrained(self.lm_type)
        self.encoder = self.lm_encoder  # for compatibility with SentenceGGraphClassifier   
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_type)

        assert self.lm_encoder.config.hidden_size == self.hidden_size, \
            f"Expected hidden size {self.hidden_size}, but got LM {self.lm_encoder.config.hidden_size}"

        # GNN encoder using PyTorch Geometric's GATConv
        # self.gnn_encoder = nn.ModuleList([
        #     GATConv(self.hidden_size, 
        #             self.hidden_size//self.num_heads, 
        #             heads=self.num_heads, 
        #             dropout=self.dropout,)
        #     for _ in range(self.num_layers)
        # ])
        self.gnn_encoder = _build_gnn_stack(
            in_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            gnn_type=self.gnn_type,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # Fusion layer to combine LM and GNN outputs
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU()
        )

        # Projector for latent space regularization
        if self.lam > 0:
            self.projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )

            # Target networks for latent space regularization
            self.target_lm_encoder = AutoModelForMaskedLM.from_pretrained(self.lm_type)
            # self.target_gnn_encoder = nn.ModuleList([
            #     GATConv(self.hidden_size, 
            #             self.hidden_size//self.num_heads, 
            #             heads=self.num_heads, 
            #             dropout=self.dropout)
            #     for _ in range(self.num_layers)
            # ])
            self.target_gnn_encoder = _build_gnn_stack(
                in_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                gnn_type=self.gnn_type,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
            self.target_fusion = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU()
            )
            self.target_projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )

            # Initialize target networks
            self._init_target_networks()

    def _init_target_networks(self):
        """Initialize target networks with the same weights as online networks"""
        self.target_lm_encoder.load_state_dict(self.lm_encoder.state_dict())
        for target_layer, layer in zip(self.target_gnn_encoder, self.gnn_encoder):
            target_layer.load_state_dict(layer.state_dict())
        self.target_fusion.load_state_dict(self.fusion.state_dict())
        self.target_projector.load_state_dict(self.projector.state_dict())

        # Freeze target networks
        for param in self.target_lm_encoder.parameters():
            param.requires_grad = False
        for layer in self.target_gnn_encoder:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.target_fusion.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_networks(self, tau=0.99):
        """Update target networks using exponential moving average"""
        for target_param, param in zip(self.target_lm_encoder.parameters(), self.lm_encoder.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * param.data
        for target_layer, layer in zip(self.target_gnn_encoder, self.gnn_encoder):
            for target_param, param in zip(target_layer.parameters(), layer.parameters()):
                target_param.data = tau * target_param.data + (1 - tau) * param.data
        for target_param, param in zip(self.target_fusion.parameters(), self.fusion.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * param.data
        for target_param, param in zip(self.target_projector.parameters(), self.projector.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * param.data

    def forward(self, input_ids, attention_mask, token_type_ids, edge_index, 
                masked_input_ids=None, batch_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            batch: Dictionary containing:
                - masked_input_ids: Token IDs with masked tokens
                - attention_mask: Attention mask
                - token_type_ids: Token type IDs
                - graph: PyTorch Geometric Data object

        Returns:
            Tuple of (total_loss, latent_loss)
        """
        lm_outputs = self.lm_encoder(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=input_ids, # HuggingFace MLM loss is computed internally,
            output_hidden_states=True
        )
        # print("LM outputs:", lm_outputs)
        node_features = lm_outputs.hidden_states[-1][:, 0]  # [CLS] token
        mlm_loss = lm_outputs.loss  # HuggingFace MLM loss

        # Get graph embeddings from GNN
        x = node_features
        # print("x shape:", x.shape, "edge_index shape:", edge_index.shape)
        for layer in self.gnn_encoder:
            x = layer(x, edge_index)
            x = F.relu(x)

        # Global pooling - need to handle case where batch_ids might have different length
        if batch_ids is None:
            # Single document case
            batch_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Ensure batch_ids matches number of nodes
        if batch_ids.size(0) != x.size(0):
            batch_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embeddings = global_mean_pool(x, batch_ids)  # Aggregate node embeddings

        # Combine LM and GNN outputs
        # print("node_features shape:", node_features.shape, \
        #       "graph_embeddings shape:", graph_embeddings.shape)

        # Expand graph embeddings to match node_features shape
        # For single document: repeat graph embedding for each node
        num_nodes = node_features.size(0)
        num_graphs = graph_embeddings.size(0)
        
        if num_graphs == 1:
            # Single document case: repeat the graph embedding for each node
            graph_embeddings_expanded = graph_embeddings.repeat(num_nodes, 1)
        else:
            # Multi-document case: use batch information to expand properly
            graph_embeddings_expanded = graph_embeddings[batch_ids]
        
        # print("node_features shape:", node_features.shape, 
            # "graph_embeddings_expanded shape:", graph_embeddings_expanded.shape)
    
        # Combine LM and GNN outputs
        concat_feat = torch.cat([node_features, graph_embeddings_expanded], dim=-1)
        # print("concat_feat shape:", concat_feat.shape)
        combined = self.fusion(concat_feat)

        # Initialize latent loss
        latent_loss = torch.tensor(0.0, device=mlm_loss.device)

        # Compute latent space regularization loss if enabled
        if self.lam > 0:
            # Get target embeddings
            with torch.no_grad():
                target_lm_outputs = self.target_lm_encoder(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True
                )
                target_node_features = target_lm_outputs.hidden_states[-1][:, 0]
                target_x = target_node_features
                for target_layer in self.target_gnn_encoder:
                    target_x = target_layer(target_x, edge_index)
                    target_x = F.relu(target_x)

                # Global pooling for target networks
                if batch_ids is None:
                    batch_ids = torch.zeros(target_x.size(0), dtype=torch.long, device=target_x.device)
                if batch_ids.size(0) != target_x.size(0):
                    batch_ids = torch.zeros(target_x.size(0), dtype=torch.long, device=target_x.device)                
                target_graph_embeddings = global_mean_pool(target_x, batch_ids)


                if num_graphs == 1:
                    target_graph_embeddings_expanded = target_graph_embeddings.repeat(num_nodes, 1)
                else:
                    target_graph_embeddings_expanded = target_graph_embeddings[batch_ids]

                target_combined = self.target_fusion(torch.cat([target_node_features, 
                                                                target_graph_embeddings_expanded], dim=-1))
                target_embeddings = self.target_projector(target_combined)

            # Get online embeddings
            online_embeddings = self.projector(combined)

            # Compute latent loss
            latent_loss = F.mse_loss(online_embeddings, target_embeddings)

            # Update target networks
            self._update_target_networks()

        # Combine losses
        total_loss = mlm_loss + self.lam * latent_loss

        return {
            "total_loss": total_loss,
            "latent_loss": latent_loss,
            "mlm_loss": mlm_loss,
            "node_embeddings": node_features,
            "graph_embeddings": graph_embeddings,
            "combined_embeddings": combined
        }

    def get_embeddings(self, input_ids, attention_mask, token_type_ids, edge_index, batch_ids=None) -> torch.Tensor:
        """
        Get embeddings for the input batch without computing losses.

        Args:
            batch: Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - token_type_ids: Token type IDs
                - graph: PyTorch Geometric Data object

        Returns:
            Node embeddings from the model.
        """
        with torch.no_grad():
            lm_outputs = self.lm_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True
            )
            node_features = lm_outputs.hidden_states[-1][:, 0]  # [CLS] token

            # graph_embeddings = node_features

            x = node_features

            edge_index = edge_index
            for layer in self.gnn_encoder:
                x = layer(x, edge_index)
                x = F.relu(x)

            # Global pooling - need to handle case where batch_ids might have different length
            if batch_ids is None:
                # Single document case
                batch_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            # Ensure batch_ids matches number of nodes
            if batch_ids.size(0) != x.size(0):
                batch_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            graph_embeddings = global_mean_pool(x, batch_ids)  # Aggregate node embeddings

            # Combine LM and GNN outputs
            # print("node_features shape:", node_features.shape, \
            #     "graph_embeddings shape:", graph_embeddings.shape)
            
            # Expand graph embeddings to match node_features shape
            # For single document: repeat graph embedding for each node
            num_nodes = node_features.size(0)
            num_graphs = graph_embeddings.size(0)
            
            if num_graphs == 1:
                # Single document case: repeat the graph embedding for each node
                graph_embeddings_expanded = graph_embeddings.repeat(num_nodes, 1)
            else:
                # Multi-document case: use batch information to expand properly
                graph_embeddings_expanded = graph_embeddings[batch_ids]
            
            # print("node_features shape:", node_features.shape, 
                # "graph_embeddings_expanded shape:", graph_embeddings_expanded.shape)

            # Combine LM and GNN outputs
            concat_feat = torch.cat([node_features, 
                                     graph_embeddings_expanded], dim=-1)
            # print("concat_feat shape:", concat_feat.shape)
            combined = self.fusion(concat_feat)

            return combined, node_features, graph_embeddings

class UniGraphLinkPredictor(nn.Module):
    """UniGraph model with link prediction head for pretraining and fine-tuning"""
    
    def __init__(self, lm_type: str, hidden_size: int, 
                 num_heads: int, num_layers: int, 
                 dropout: float, lam: float = 0.0):
        super().__init__()
        
        # Initialize base UniGraph model
        self.unigraph = UniGraph(lm_type, hidden_size, num_heads, num_layers, dropout, lam)
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lm_type)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None,
                masked_input_ids=None, batch_ids=None, mode="pretrain") -> Dict[str, Any]:
        """
        Forward pass for pretraining or fine-tuning
        
        Args:
            tokenized_inputs: Tokenized inputs from the tokenizer
            batch: Dictionary containing graph data
            mode: "pretrain" or "finetune"
            
        Returns:
            Dictionary with losses and predictions
        """
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        token_type_ids = tokenized_inputs["token_type_ids"]

        pos_edge_pairs = edge_index.t()  # Positive edges

        device = input_ids.device

        if edge_pairs is None:
            # Negative sampling
            neg_edge_pairs = negative_sampling(
                edge_index=edge_index,
                num_nodes=input_ids.size(0),
                num_neg_samples=pos_edge_pairs.size(0)
            ).t()

            edge_pairs = torch.cat([pos_edge_pairs, neg_edge_pairs], dim=0)
            link_labels = torch.cat([torch.ones(pos_edge_pairs.size(0), 1, device=device),
                                     torch.zeros(neg_edge_pairs.size(0), 1, device=device)], dim=0)
        else:
            link_labels = torch.ones(edge_pairs.size(0), 1, device=device)

        if mode == "pretrain":
            # Pretraining mode
            output = self.unigraph(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                edge_index=edge_index,
                masked_input_ids=masked_input_ids,
                batch_ids=batch_ids
            )

            combined_embeddings = output["combined_embeddings"]
            # mlm_loss = output["mlm_loss"]
            latent_loss = output["latent_loss"]
            total_loss = output["total_loss"]

            link_logits = self.predict_links(combined_embeddings, edge_pairs)
            link_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
            
            total_loss += link_loss
            
            return {
                "pretrain_loss": total_loss,
                "latent_loss": latent_loss,
                "link_loss": link_loss
            }
        
        elif mode == "finetune":
            # Fine-tuning mode
            # node_embeddings = self.unigraph.get_embeddings(batch)
            combined_embeddings, node_embeddings, _ = self.unigraph.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                edge_index=edge_index,
                batch_ids=batch_ids
            )
            
            # link_logits = self.predict_links(node_embeddings, edge_pairs)
            link_logits = self.predict_links(combined_embeddings, edge_pairs)
            link_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
            return {
                "link_loss": link_loss,
                "logits": link_logits,
                "node_embeddings": node_embeddings
            }
    
    def predict_links(self, node_embeddings: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        Predict link probabilities for given edge pairs
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_size]
            edge_pairs: Edge pairs to predict [num_edges, 2]
            
        Returns:
            Link predictions [num_edges, 1]
        """
        # Get embeddings for source and target nodes
        src_embeddings = node_embeddings[edge_pairs[:, 0]]
        dst_embeddings = node_embeddings[edge_pairs[:, 1]]
        
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        
        # Predict link probability
        link_logits = self.link_predictor(edge_embeddings)
        
        return link_logits
    
class UniGraphGraphClassifier(nn.Module):
    """UniGraph model with graph classification head"""
    
    def __init__(self, lm_type: str, hidden_size: int, 
                 num_heads: int, num_layers: int, 
                 dropout: float, lam: float = 0.0, 
                 num_classes: int = 2,
                 embedding_choice: str = "graph"):
        super().__init__()
        
        # Initialize base UniGraph model
        self.unigraph = UniGraph(lm_type, hidden_size, num_heads, num_layers, dropout, lam)
        self.embedding_choice = embedding_choice

        assert embedding_choice in ["node", "graph", "combined"], "Invalid embedding choice"
        classifier_input_size = hidden_size
        if embedding_choice == "combined":
            classifier_input_size = hidden_size * 2

        # Graph classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize tokenizer
        self.tokenizer = self.unigraph.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, tokenized_inputs, edge_index, batch_ids=None,
                masked_input_ids=None, labels=None) -> Dict[str, Any]:
        """
        Forward pass for graph classification
        
        Args:
            tokenized_inputs: Tokenized inputs from the tokenizer
            batch: Dictionary containing graph data
            
        Returns:
            Dictionary with loss and predictions
        """
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        token_type_ids = tokenized_inputs["token_type_ids"]

        # device = input_ids.device

        combined_embeddings, node_embeddings, graph_embeddings = self.unigraph.get_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            edge_index=edge_index,
            batch_ids=batch_ids
        )

        # Use graph embeddings for classification
        logits = self.classifier(graph_embeddings)

        output = {
            "logits": logits
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output["loss"] = loss

        return {
            "logits": logits,
            "loss": output.get("loss", None),
            "combined_embeddings": combined_embeddings,
            "node_embeddings": node_embeddings,
            "graph_embeddings": graph_embeddings
        }

class UniGraphGGraphClassifier(nn.Module):
    """
    UniGraph-based graph/subgraph classifier with a SentenceGGraphClassifier-like interface.

    Returns:
      - logits: [num_graphs, num_classes] (or [num_classes] if batch_ids is None)
      - node_features: [num_nodes, H] node-level features used for edge-pair features
      - edge_pairs_features: [num_edges, 2H] if edge_pairs is given; else None
    """
    def __init__(
        self,
        encoder_model: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        lam: float = 0.0,
        num_classes: int = 2,
        # encoder_grad: bool = True,
        embedding_choice: str = "graph",  # "graph" | "node" | "combined",
        checkpoint_path: str = None,
        gradient_mode: bool = True,
        **kwargs
    ):
        super().__init__()
        self.unigraph = UniGraph(
            lm_type=encoder_model,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            lam=lam
        )
        self.gradient_mode = gradient_mode
        if checkpoint_path is not None:
            self.load_unigraph_weights(checkpoint_path)

        self.embedding_choice = embedding_choice
        assert embedding_choice in ["node", "graph", "combined"]

        classifier_in = hidden_size if embedding_choice in ["node", "graph"] else hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # Expose tokenizer compatibly with SentenceGGraphClassifier
        self.tokenizer = self.unigraph.tokenizer
        self.encoder = self.unigraph
        # self.encoder = SimpleNamespace(tokenizer=self.tokenizer)

    def forward(
        self,
        tokenized_inputs,
        edge_index: torch.Tensor,
        batch_ids: torch.Tensor = None,
        edge_pairs: torch.Tensor = None
    ):
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs.get("attention_mask", None)
        token_type_ids = tokenized_inputs.get("token_type_ids", None)


        if not self.gradient_mode:
            combined_embeddings, node_embeddings, graph_embeddings = self.unigraph.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                edge_index=edge_index,
                batch_ids=batch_ids
            )
        else:
            output = self.unigraph(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                edge_index=edge_index,
                masked_input_ids=input_ids,  # No masking during fine-tuning
                batch_ids=batch_ids
            )
            combined_embeddings = output["combined_embeddings"]
            node_embeddings = output["node_embeddings"]
            graph_embeddings = output["graph_embeddings"]

        if self.embedding_choice == "graph":
            logits = self.classifier(graph_embeddings)
            node_feats_for_pairs = node_embeddings
        elif self.embedding_choice == "node":
            # Pool node embeddings before classification
            if batch_ids is None:
                pooled = global_mean_pool(node_embeddings, torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device))
            else:
                pooled = global_mean_pool(node_embeddings, batch_ids)
            logits = self.classifier(pooled)
            node_feats_for_pairs = node_embeddings
        else:
            # combined: concatenate node and graph-level contexts at graph-level
            if batch_ids is None:
                pooled_nodes = global_mean_pool(combined_embeddings, torch.zeros(combined_embeddings.size(0), dtype=torch.long, device=combined_embeddings.device))
            else:
                pooled_nodes = global_mean_pool(combined_embeddings, batch_ids)
            logits = self.classifier(pooled_nodes)
            node_feats_for_pairs = combined_embeddings

        edge_pairs_features = None
        if edge_pairs is not None:
            src = node_feats_for_pairs[edge_pairs[:, 0]]
            dst = node_feats_for_pairs[edge_pairs[:, 1]]
            edge_pairs_features = torch.cat([src, dst], dim=-1)

        return logits, node_embeddings, edge_pairs_features
    
    def load_unigraph_weights(self, checkpoint_path: str):
        """Load only UniGraph weights from a checkpoint into self.unigraph.

        Supports checkpoints saved as:
          - raw state_dict
          - {'model_state_dict': state_dict}
          - {'state_dict': state_dict}
        Keys may be prefixed with 'module.' and or 'unigraph.'.
        """
        device = next(self.parameters()).device
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        target_keys = set(self.unigraph.state_dict().keys())
        cleaned = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("unigraph."):
                key = key[len("unigraph."):]
            if key in target_keys:
                cleaned[key] = v

        if len(cleaned) == 0:
            print("No matching UniGraph keys found in checkpoint. Nothing loaded.")
            return

        load_res = self.unigraph.load_state_dict(cleaned, strict=False)
        missing = getattr(load_res, "missing_keys", [])
        unexpected = getattr(load_res, "unexpected_keys", [])

        print(f"Loaded UniGraph weights from {checkpoint_path}")
        print(f"Matched keys: {len(cleaned)} | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        self.to(device)


    
def prepare_batch_for_unigraph(documents: List[Data], tokenizer, device, 
                               mask_rate: float=0.15) -> Dict[str, Any]:
    """
    Prepare a batch of documents for UniGraph model
    
    Args:
        documents: List of PyTorch Geometric Data objects representing documents
        tokenizer: Tokenizer for the language model
        device: Device to move tensors to
    Returns:
        Dictionary containing tokenized inputs and graph data
    """ 

    
    # Tokenize sentences
    sentences = []
    graphs = []
    for doc in documents:
        # print(doc)
        sentences.extend(doc["sentences"])
        graph = Data(
            x=None,  # Node features will be obtained from LM
            edge_index=doc["edge_index"],
            edge_pairs=doc["edge_pairs"],
            node_labels=doc["node_labels"],
            labels=doc["labels"]
        )
        graphs.append(graph)

    batch = Batch.from_data_list(graphs).to(device)


    tokenized = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Create masked inputs for MLM
    input_ids = tokenized.input_ids.to(device)
    # attention_mask = tokenized.attention_mask.to(device)
    # token_type_ids = tokenized.token_type_ids.to(device)
    
    # # Create masked input ids
    masked_input_ids = input_ids.clone()
    # rand = torch.rand(input_ids.shape).to(device)
    # mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)
    # selection = torch.flatten(mask_arr.nonzero()).tolist()
    # masked_input_ids[selection] = tokenizer.mask_token_id
    # masked_input_ids = masked_input_ids.to(device)
    # Build a boolean mask with the same shape as input_ids [B, L]

    rand = torch.rand_like(input_ids, dtype=torch.float, device=device)
    # avoid masking special tokens
    special = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    if getattr(tokenizer, "cls_token_id", None) is not None:
        special |= input_ids.eq(tokenizer.cls_token_id)
    if getattr(tokenizer, "sep_token_id", None) is not None:
        special |= input_ids.eq(tokenizer.sep_token_id)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        special |= input_ids.eq(tokenizer.pad_token_id)

    mask_arr = (rand < mask_rate) & (~special)  # boolean mask [B, L]

    # Direct boolean indexing (no flattening)
    masked_input_ids[mask_arr] = tokenizer.mask_token_id
    masked_input_ids = masked_input_ids.to(device)


    return {
        # "input_ids": input_ids,
        # "attention_mask": attention_mask,
        # "token_type_ids": token_type_ids,
        "tokenized_inputs": tokenized.to(device),
        "masked_input_ids": masked_input_ids,
        # 
        "edge_index": batch.edge_index,
        "batch_ids": batch.batch
    }


def train_pretrain(args, model: UniGraphLinkPredictor, train_loader, optimizer, epoch: int):
    """
    Pretraining function similar to the UniGraph repository
    
    Args:
        args: Training arguments
        model: UniGraph link predictor model
        train_loader: Training data loader
        optimizer: Optimizer
        epoch: Current epoch
        
    Returns:
        Tuple of (pretrain_loss, latent_loss)
    """
    model.train()
    total_pretrain_loss = 0.0
    total_latent_loss = 0.0
    total_link_loss = 0.0
    num_batches = 0
    
    for batch_idx, documents in enumerate(train_loader):
        # Prepare batch for UniGraph
        batch = prepare_batch_for_unigraph(
            documents, 
            model.tokenizer, 
            next(model.parameters()).device
        )

        # print(batch)
        
        # Forward pass
        outputs = model(**batch, mode="pretrain")
        
        pretrain_loss = outputs["pretrain_loss"]
        latent_loss = outputs["latent_loss"]
        link_loss = outputs["link_loss"]
        
        # Backward pass
        optimizer.zero_grad()
        pretrain_loss.backward()
        
        # Gradient clipping
        if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        # Accumulate losses
        total_pretrain_loss += pretrain_loss.item()
        total_latent_loss += latent_loss.item()
        total_link_loss += link_loss.item()
        num_batches += 1
        
        # Log progress
        if batch_idx % args.log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Pretrain Loss: {pretrain_loss.item():.4f}, '
                  f'Latent Loss: {latent_loss.item():.4f}, '
                  f'Link Loss: {link_loss.item():.4f}')
    
    avg_pretrain_loss = total_pretrain_loss / num_batches
    avg_latent_loss = total_latent_loss / num_batches
    avg_link_loss = total_link_loss / num_batches
    
    print(f'Epoch {epoch} - Avg Pretrain Loss: {avg_pretrain_loss:.4f}, '
          f'Avg Latent Loss: {avg_latent_loss:.4f}, '
          f'Avg Link Loss: {avg_link_loss:.4f}')
    
    return avg_pretrain_loss, avg_latent_loss

def evaluate_link_prediction(model: UniGraphLinkPredictor, eval_loader) -> float:
    """
    Evaluate link prediction performance on validation/test set
    
    Args:
        model: UniGraph link predictor model
        eval_loader: Evaluation data loader
        
    Returns:
        Average link prediction accuracy
    """
    model.eval()
    total_correct = 0
    total_edges = 0

    result = {}

    all_labels = []
    all_preds = []
    all_probs = []
    all_edges = []
    
    with torch.no_grad():
        for documents in eval_loader:
            # Prepare batch for UniGraph
            batch = prepare_batch_for_unigraph(
                documents, 
                model.tokenizer, 
                next(model.parameters()).device
            )
            
            # Forward pass
            input_ids = batch["tokenized_inputs"]["input_ids"]
            attention_mask = batch["tokenized_inputs"]["attention_mask"]
            token_type_ids = batch["tokenized_inputs"]["token_type_ids"]
            combined, node_embeddings, _ = model.unigraph.get_embeddings(
                # input_ids=batch["input_ids"],
                # attention_mask=batch["attention_mask"],
                # token_type_ids=batch["token_type_ids"],
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                edge_index=batch["edge_index"],
                batch_ids=batch["batch_ids"]
            )

            # print("node_embeddings shape:", node_embeddings.shape)
            # print("combined shape:", combined.shape)

            edge_index = batch["edge_index"]
            pos_edge_pairs = edge_index.t()  # Positive edges
            neg_edge_pairs = negative_sampling(
                edge_index=edge_index,
                num_nodes=input_ids.size(0),
                num_neg_samples=pos_edge_pairs.size(0)
            ).t()

            device = node_embeddings.device
            all_edge_pairs = torch.cat([pos_edge_pairs, neg_edge_pairs], dim=0)
            link_labels = torch.cat([torch.ones(pos_edge_pairs.size(0), 1, device=device),
                                     torch.zeros(neg_edge_pairs.size(0), 1, device=device)], 
                                     dim=0)
            
            # link_logits = model.predict_links(node_embeddings, all_edge_pairs)
            link_logits = model.predict_links(combined, all_edge_pairs)
            link_probs = torch.sigmoid(link_logits)
            predictions = (link_probs > 0.5).float()

            total_correct += (predictions == link_labels).sum().item()
            total_edges += link_labels.size(0)
            all_labels.extend(link_labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(link_probs.cpu().numpy())
            all_edges.extend(all_edge_pairs.cpu().numpy())
    
    accuracy = total_correct / total_edges if total_edges > 0 else 0.0
    print(f'Link Prediction Accuracy: {accuracy:.4f}')
    # compute roc auc
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(f'Link Prediction ROC AUC: {roc_auc:.4f}')
    except ValueError:
        print("ROC AUC could not be computed due to lack of positive or negative samples.")
        roc_auc = None
    
    # compute precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    result['accuracy'] = accuracy
    result['roc_auc'] = roc_auc
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1 
    return result


def main():
    """
    Main function demonstrating UniGraph link prediction pretraining
    """
    # Argument parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lm_type", type=str, default="bert-base-uncased")
    argparser.add_argument("--hidden_size", type=int, default=768)
    argparser.add_argument("--num_heads", type=int, default=8)
    argparser.add_argument("--num_layers", type=int, default=2)
    argparser.add_argument("--dropout", type=float, default=0.1)
    argparser.add_argument("--lam", type=float, default=0.1, help="Latent regularization weight")
    argparser.add_argument("--learning_rate", type=float, default=1e-4)
    argparser.add_argument("--num_epochs", type=int, default=10)
    argparser.add_argument("--log_interval", type=int, default=10)
    argparser.add_argument("--max_grad_norm", type=float, default=1.0)
    argparser.add_argument("--model_save_path", type=str, default=None, help="Path to save the trained model")



    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = UniGraphLinkPredictor(
        lm_type=args.lm_type,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lam=args.lam
    ).to(device)

    # Initialize optimizer
    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Compliance and regulatory document graphs
    documents = [
        create_document_graph(
            sentences=[
                "Healthcare organizations must implement access controls.",
                "Patient data requires encryption at rest.",
                "Audit logs must be maintained for security events.",
                "Administrative safeguards protect electronic health information."
            ],
            edge_index=torch.tensor([[0, 1, 2, 0, 3], [1, 2, 3, 3, 0]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]]),
            node_labels=torch.tensor([2, 1, 1, 2], dtype=torch.float),  # 0: general, 1: technical, 2: administrative
            labels=torch.tensor([1, 1, 1, 1, 0], dtype=torch.float)  # 1: compliant relationship, 0: non-compliant
        ),
        create_document_graph(
            sentences=[
                "Financial institutions must verify customer identity.",
                "Anti-money laundering policies are mandatory.",
                "Suspicious activities require immediate reporting.",
                "Customer due diligence procedures must be documented."
            ],
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]]),
            node_labels=torch.tensor([2, 1, 0, 2], dtype=torch.float),  # 0: reporting, 1: policy, 2: procedure
            labels=torch.tensor([1, 1, 1, 0], dtype=torch.float)
        ),
        create_document_graph(
            sentences=[
                "Data processors must obtain explicit consent.",
                "Personal data retention has time limits.",
                "Individuals have the right to data portability.",
                "Privacy impact assessments are required for high-risk processing."
            ],
            edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]]),
            node_labels=torch.tensor([1, 0, 2, 1], dtype=torch.float),  # 0: rights, 1: obligations, 2: procedures
            labels=torch.tensor([1, 1, 1, 1], dtype=torch.float)
        ),
        create_document_graph(
            sentences=[
                "Cloud providers must ensure data sovereignty.",
                "Cross-border data transfers require adequacy decisions.",
                "Data localization laws vary by jurisdiction.",
                "Service level agreements must specify data protection measures."
            ],
            edge_index=torch.tensor([[0, 1, 2, 1], [1, 2, 3, 3]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [1, 3]]),
            node_labels=torch.tensor([1, 0, 0, 2], dtype=torch.float),  # 0: legal, 1: technical, 2: contractual
            labels=torch.tensor([1, 1, 0, 1], dtype=torch.float)
        ),
        create_document_graph(
            sentences=[
                "Security incidents must be reported within 72 hours.",
                "Breach notification procedures should be tested regularly.",
                "Data subjects must be informed of high-risk breaches.",
                "Supervisory authorities require detailed incident reports."
            ],
            edge_index=torch.tensor([[0, 1, 2, 0, 3], [1, 2, 3, 3, 2]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [3, 2]]),
            node_labels=torch.tensor([0, 1, 2, 0], dtype=torch.float),  # 0: reporting, 1: procedure, 2: notification
            labels=torch.tensor([1, 1, 1, 1, 0], dtype=torch.float)
        ),
        create_document_graph(
            sentences=[
                "Software development requires secure coding practices.",
                "Code reviews must identify security vulnerabilities.",
                "Penetration testing validates security controls.",
                "Security training is mandatory for developers."
            ],
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            edge_pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            node_labels=torch.tensor([1, 1, 0, 2], dtype=torch.float),  # 0: testing, 1: development, 2: training
            labels=torch.tensor([1, 1, 1, 1], dtype=torch.float)
        )
    ]
    
    # Create simple data loader (in practice, use PyTorch DataLoader)
    train_loader = [documents]  # Simplified for demo
    
    # Training loop
    print("Starting UniGraph Link Prediction Pretraining...")
    
    for epoch in range(args.num_epochs):
        pretrain_loss, pretrain_latent_loss = train_pretrain(
            args, model, train_loader, pretrain_optimizer, epoch
        )
        
        print(f"Epoch {epoch + 1}/{args.num_epochs} completed")
        print(f"Pretrain Loss: {pretrain_loss:.4f}")
        print(f"Latent Loss: {pretrain_latent_loss:.4f}")
        print("-" * 50)
    
    print("Pretraining completed!")

    # Evaluate on the same data for demonstration (in practice, use separate validation/test set)
    eval_accuracy = evaluate_link_prediction(model, train_loader)
    print("Evaluation Accuracy:", eval_accuracy)
    
    # Save model
    if args.model_save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
            'epoch': args.num_epochs
        }, f'{args.model_save_path}/unigraph_link_predictor.pth')

        print("Model saved to", f'{args.model_save_path}/unigraph_link_predictor.pth')

if __name__ == "__main__":
    main()
