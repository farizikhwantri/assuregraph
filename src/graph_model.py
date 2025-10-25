import random
# from typing import Dict

import networkx as nx
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from transformers import AutoModel, AutoTokenizer

from torch_geometric.nn import GCNConv, SAGEConv, GATConv
# from torch_geometric.nn import GAT
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import mask_feature

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def save_checkpoint(model, config: dict, optimizer=None, epoch=None, loss=None, filename="checkpoint.pth"):
    """
    Save a checkpoint including model state, optimizer state, model parameters, and configuration.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer whose state to save.
        epoch (int): Current epoch number.
        loss (float): The loss value.
        config (dict): Configuration settings used by the model.
        filename (str): File name for the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        # Additionally, you can even store a snapshot of model parameters (for debugging/comparison)
        "model_parameters": {name: param.detach().cpu() for name, param in model.named_parameters()},
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def contrastive_criterion(logits, labels, temperature=0.1):
    """
    Custom criterion function to compute contrastive loss by separating positive and negative scores.

    Args:
        logits (torch.Tensor): Predicted scores (logits) for all edges.
        labels (torch.Tensor): Ground truth labels (1 for positive, 0 for negative).
        temperature (float): Temperature scaling for logits. Default is 0.1.

    Returns:
        torch.Tensor: Computed contrastive loss.
    """
    # Separate positive and negative scores
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = logits[pos_mask]
    neg_scores = logits[neg_mask]

    # Compute contrastive loss
    pos_loss = -F.logsigmoid(pos_scores / temperature).mean()
    neg_loss = -F.logsigmoid(-neg_scores / temperature).mean()

    return pos_loss + neg_loss

def load_checkpoint(model, optimizer, filename="checkpoint.pth", device="cpu"):
    """
    Load a checkpoint and restore model state, optimizer state, configuration, and model parameters.

    Args:
        model (nn.Module): The model into which state is loaded.
        optimizer (Optimizer): The optimizer into which state is loaded.
        filename (str): Checkpoint file name.
        device (str): Device to map the checkpoint.
    
    Returns:
        epoch (int): The epoch when the checkpoint was saved.
        loss (float): The saved loss.
        config (dict): The configuration used by the model.
        model_parameters (dict): Snapshot of the model parameters.
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    config = checkpoint["config"]
    model_parameters = checkpoint.get("model_parameters", None)
    print(f"Checkpoint loaded from {filename}: epoch {epoch}, loss {loss}")
    return epoch, loss, config, model_parameters

# Sentence encoder that now accepts already tokenized inputs.
class SentenceEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", require_grad=False, embedding_layer=None):
        super().__init__()
        print(f"Loading encoder model: {model_name}, require_grad={require_grad}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if not require_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.embedding_layer = embedding_layer
        # get self.encoder's embedding layer using getattr
        if isinstance(self.encoder, str):
            self.embedding_layer = getattr(self.encoder, self.embedding_layer)


    def forward(self, tokenized_inputs, return_token_grad=False):
        """
        Expects tokenized_inputs as a dict containing:
          - input_ids (tensor)
          - attention_mask (tensor)
        """
        # print(f"Input type: {type(tokenized_inputs)}")
        if isinstance(tokenized_inputs, torch.Tensor):
            input_ids = tokenized_inputs
            attention_mask = None
            # convert the input_ids to long tensor if not already
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            # print("Received raw tensor input_ids.", type(input_ids))
        else:
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            # print("Received tokenized input_ids and attention_mask.", \
            #       type(input_ids), type(attention_mask))

        # Get embeddings from the encoder's embedding layer (using the provided token IDs)
        # inputs_embeds = self.encoder.embeddings.word_embeddings(input_ids)
        # get inputs_embeds from the embedding layer
        outputs = None
        if return_token_grad == False:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        elif return_token_grad:
            inputs_embeds = None
            if self.embedding_layer is not None:
                inputs_embeds = self.embedding_layer(input_ids)
            else:
                inputs_embeds = self.encoder.embeddings(input_ids)
            inputs_embeds = inputs_embeds.clone().detach().requires_grad_()

            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
        # print(f"Encoder outputs: {outputs}")
        # check if pooler output is available
        if hasattr(outputs, 'pooler_output'):
            cls_embeds = outputs.pooler_output
        else:
            # Use the CLS token embedding as the sentence representation
            # This is typically the first token in the sequence
            if hasattr(outputs, 'last_hidden_state'):
                cls_embeds = outputs.last_hidden_state[:, 0]

        inputs_embeds = outputs.last_hidden_state
        # cls_embeds = outputs.last_hidden_state[:, 0]  # CLS token embedding
        # if return_token_grad:
        return cls_embeds, inputs_embeds, input_ids, attention_mask
        # return cls_embeds

class SentenceLinkPredictor(nn.Module):
    def __init__(self, encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__()
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size
        self.gcn = self.forward_embed

        # Classifier that takes two concatenated sentence embeddings as input
        self.classifier = nn.Linear(in_dim * 2, 1)

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None):
        """
        tokenized_inputs: dict containing "input_ids" and "attention_mask" (for all sentences)
        edge_pairs: tensor of shape (num_edges, 2) containing indices of sentence pairs.
        """
        # Obtain continuous node features for all sentences.
        sentence_reps = self.encoder(tokenized_inputs)[0]  # shape: (num_sentences, in_dim)
        if edge_pairs is None:
            # make index pairs from edge_index
            edge_pairs = edge_index.t()
        # Gather representations for each edge
        src = sentence_reps[edge_pairs[:, 0]]
        tgt = sentence_reps[edge_pairs[:, 1]]
        # Concatenate and classify
        combined = torch.cat([src, tgt], dim=1)
        logits = self.classifier(combined).squeeze()
        return logits
    
    def forward_embed(self, x, edge_index):
        """
        tokenized_inputs: dict containing "input_ids" and "attention_mask" (for all sentences)
        edge_index: tensor indicating graph connectivity
        """
        # print(f"Encoded node features shape: {x.shape}")
        # use identity function for now
        return x


# Graph Link Predictor Model using the revised SentenceEncoder
class SentenceGraphLinkPredictor(nn.Module):
    # def __init__(self, in_dim=768, hidden_dim=256, encoder_model="bert-base-uncased", encoder_grad=False):
    def __init__(self, hidden_dim = 256, encoder_model="bert-base-uncased", encoder_grad=False,
                 GNN=GCNConv):
        super().__init__()
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size
        self.gcn = GNN(in_dim, hidden_dim)
        self.link_classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None):
        """
        Expects:
          tokenized_inputs: a dict with "input_ids" and "attention_mask" (pre-tokenized tensor inputs)
          edge_index: tensor indicating graph connectivity
          edge_pairs: tensor of node pairs for link prediction (required)
        """
        x = self.encoder(tokenized_inputs)  # shape: (num_nodes, in_dim)
        # print(f"Encoded node features shape: {x}")
        x = x[0]  # Get the CLS token embedding
        x = F.relu(self.gcn(x, edge_index))   # shape: (num_nodes, hidden_dim)
        # print(f"Encoded node features shape: {x.shape}, {edge_index.shape}")
        if edge_pairs is None:
            # make index pairs from edge_index
            edge_pairs = edge_index.t()
        # print(f"edge_index shape: {edge_index.shape}, edge pairs shape: {edge_pairs.shape}")
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        out = self.link_classifier(torch.cat([src, tgt], dim=1)).squeeze()
        # print(f"Output logits shape: {out.shape}")
        return out
    
    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward_link(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index).view(-1)
    

class SentenceGATLinkPredictor(SentenceGraphLinkPredictor):
    def __init__(self, hidden_dim=256, encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__(hidden_dim=hidden_dim, encoder_model=encoder_model, encoder_grad=encoder_grad,
                         GNN=GATConv)  # Using GAT as an example GNN layer

class SentenceGraphSAGELinkPredictor(SentenceGraphLinkPredictor):
    def __init__(self, hidden_dim=256, encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__(hidden_dim=hidden_dim, encoder_model=encoder_model, encoder_grad=encoder_grad,
                         GNN=SAGEConv)  # Using SAGEConv as an example GNN layer
    

class SentenceGraphNodeClassifier(SentenceGraphLinkPredictor):
    def __init__(self, num_classes, hidden_dim=256, 
                 encoder_model="bert-base-uncased", 
                 encoder_grad=False):
        super().__init__(hidden_dim=hidden_dim, 
                         encoder_model=encoder_model, 
                         encoder_grad=encoder_grad)
        self.node_classifier = nn.Linear(hidden_dim, num_classes)  # Multi-class classification

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None):
        x = self.encoder(tokenized_inputs)[0]  # shape: (num_nodes, in_dim)
        x = F.relu(self.gcn(x, edge_index))   # shape: (num_nodes, in_dim)
        logits = self.node_classifier(x).squeeze()  # shape: (num_nodes, num_classes)
        if edge_pairs is None:
            # make index pairs from edge_index
            edge_pairs = edge_index.t()
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        edge_features = torch.cat([src, tgt], dim=1)
        return logits, x, edge_features
    
    def load_from_link_predictor(self, link_predictor):
        """
        Load weights from a link predictor model into this node classifier.
        """
        # load the sentence encoder weights
        self.encoder.load_state_dict(link_predictor.encoder.state_dict())
        # load the GNN layers
        self.gcn.load_state_dict(link_predictor.gcn.state_dict())


class SentenceGATNodeClassifier(SentenceGraphNodeClassifier):
    def __init__(self, num_classes, hidden_dim=256, 
                 encoder_model="bert-base-uncased", 
                 encoder_grad=False):
        super().__init__(num_classes=num_classes, 
                         hidden_dim=hidden_dim, 
                         encoder_model=encoder_model, 
                         encoder_grad=encoder_grad)
        self.gcn = GATConv(self.encoder.encoder.config.hidden_size, hidden_dim)  # Using GATConv

class SentenceSAGENodeClassifier(SentenceGraphNodeClassifier):
    def __init__(self, num_classes, hidden_dim=256, 
                 encoder_model="bert-base-uncased", 
                 encoder_grad=False):
        super().__init__(num_classes=num_classes, 
                         hidden_dim=hidden_dim, 
                         encoder_model=encoder_model, 
                         encoder_grad=encoder_grad)
        self.gcn = SAGEConv(self.encoder.encoder.config.hidden_size, hidden_dim)  # Using SAGEConv

# The remainder of your file (e.g. UnsupervisedLinkPredictor, evaluation functions, etc.) remains unchanged.

class SentenceDGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2, GNN=GCNConv, 
                 encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__()
        # self.activation = activation
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size

        self.gcn = ModuleList()
        self.gcn.append(GNN(in_dim, hidden_channels))
        for _ in range(0, num_layers - 1):
            self.gcn.append(GNN(hidden_channels, hidden_channels))
        self.gcn.append(GNN(hidden_channels, 1))

        final_latent_dim = hidden_channels * num_layers + 1

        self.link_classifier = nn.Linear(final_latent_dim * 2, 1)

    def forward(self, x, edge_index, edge_pairs):

        x = self.encoder(x)  # shape: (num_nodes, in_dim)
        x = x[0] # Get the CLS token embedding
        xs = [x]

        for conv in self.gcn:
            xs += [conv(xs[-1], edge_index).tanh()]
            # print(f"Intermediate node features shape: {xs[-1].shape}")
        x = torch.cat(xs[1:], dim=-1)
        # print(f"Encoded node features shape: {x.shape}")

        if edge_pairs is None:
            # make index pairs from edge_index
            edge_pairs = edge_index.t()
        # print edge_pairs shape
        # print(f"edge_index shape: {edge_index.shape}, edge pairs shape: {edge_pairs.shape}")
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        out = self.link_classifier(torch.cat([src, tgt], dim=1)).squeeze()
        # print(f"Output logits shape: {out.shape}")
        return out

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward_link(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index).view(-1)
    

class SentenceDGATLinkPredictor(SentenceDGCNN):
    def __init__(self, hidden_channels=256, num_layers=2, encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__(hidden_channels=hidden_channels, num_layers=num_layers, 
                         GNN=GATConv, encoder_model=encoder_model, encoder_grad=encoder_grad)
        
class SentenceDGSAGELinkPredictor(SentenceDGCNN):
    def __init__(self, hidden_channels=256, num_layers=2, encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__(hidden_channels=hidden_channels, num_layers=num_layers, 
                         GNN=SAGEConv, encoder_model=encoder_model, encoder_grad=encoder_grad)
        

class SentenceDGConvNodeClassifier(SentenceDGCNN):
    def __init__(self, num_classes, hidden_channels=256, num_layers=2, 
                encoder_model="bert-base-uncased", encoder_grad=False, GNN=GCNConv):
        super().__init__(hidden_channels=hidden_channels, 
                         num_layers=num_layers, 
                         GNN=GNN, 
                         encoder_model=encoder_model, 
                         encoder_grad=encoder_grad)
        self.num_class = num_classes
        self.node_classifier = nn.Linear(hidden_channels * num_layers + 1, num_classes)  # Multi-class classification

    def forward(self, x, edge_index, edge_pairs=None):
        x = self.encoder(x)  # shape: (num_nodes, in_dim)
        node_features = x[0]  # Get the CLS token embedding
        x = x[1:]  # Get the rest of the node features
        xs = [node_features]
        for conv in self.gcn:
            node_features = conv(node_features, edge_index).tanh()
            xs += [node_features]
            # print(f"Intermediate node features shape: {node_features.shape}")
        node_features = torch.cat(xs[1:], dim=-1)  # Concatenate all intermediate features
        # print(f"Encoded node features shape: {node_features.shape}")
        if edge_pairs is None:
            # make index pairs from edge_index
            edge_pairs = edge_index.t()
        # print edge_pairs shape
        # print(f"edge_index shape: {edge_index.shape}, edge pairs shape: {edge_pairs.shape}")
        src = node_features[edge_pairs[:, 0]]
        tgt = node_features[edge_pairs[:, 1]]
        # concatenate source and target edge features
        edge_pairs_features = torch.cat([src, tgt], dim=1)

        logits = self.node_classifier(node_features).squeeze()
        return logits, node_features, edge_pairs_features
    
    def load_from_link_predictor(self, link_predictor):
        """
        Load weights from a link predictor model into this node classifier.
        """
        # load the sentence encoder weights
        self.encoder.load_state_dict(link_predictor.encoder.state_dict())
        # load the GNN layers
        if isinstance(link_predictor.gcn, ModuleList):
            for i, layer in enumerate(self.gcn):
                layer.load_state_dict(link_predictor.gcn[i].state_dict())
        else:
            raise ValueError("Link predictor GNN layers are not a ModuleList.")
        print("Loaded weights from link predictor to node classifier.")

class SentenceDGSAGENodeClassifier(SentenceDGConvNodeClassifier):
    def __init__(self, num_classes, hidden_channels=256, num_layers=2, 
                 encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__(num_classes=num_classes, 
                         hidden_channels=hidden_channels, num_layers=num_layers, 
                         GNN=SAGEConv, 
                         encoder_model=encoder_model, encoder_grad=encoder_grad)

class SentenceDGATNodeClassifier(SentenceDGConvNodeClassifier):
    def __init__(self, num_classes, hidden_channels=256, num_layers=2, 
                 encoder_model="bert-base-uncased", encoder_grad=False):
        super().__init__(num_classes=num_classes, 
                         hidden_channels=hidden_channels, num_layers=num_layers, 
                         GNN=GATConv, 
                         encoder_model=encoder_model, encoder_grad=encoder_grad)
        

class SentenceGGraphClassifier(nn.Module):
    """
    Sentence-based graph/subgraph classifier with an interface similar to SentenceGraphNodeClassifier.
    Returns:
      - logits: [num_graphs, num_classes] (or [num_classes] if single graph without batch_ids)
      - node_features: [num_nodes, hidden_dim]
      - edge_pairs_features: [num_edges, hidden_dim * 2]
    """
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        GNN=GCNConv,
        pooling: str = "mean"  # "mean" | "sum" | "max"
    ):
        super().__init__()
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size
        self.gcn = GNN(in_dim, hidden_dim)
        self.num_classes = num_classes
        self.pooling = pooling.lower()
        self.graph_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None, batch_ids=None):
        # Encode sentence nodes and run one GNN layer
        x = self.encoder(tokenized_inputs)[0]
        x = F.relu(self.gcn(x, edge_index))

        # Build edge-pair features for compatibility
        if edge_pairs is None:
            edge_pairs = edge_index.t()
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        edge_pairs_features = torch.cat([src, tgt], dim=1)

        # Graph pooling
        if batch_ids is None:
            # Single graph
            if self.pooling == "sum":
                g = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            elif self.pooling == "max":
                g = global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            else:
                g = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            logits = self.graph_classifier(g).squeeze(0)
        else:
            if self.pooling == "sum":
                g = global_add_pool(x, batch_ids)
            elif self.pooling == "max":
                g = global_max_pool(x, batch_ids)
            else:
                g = global_mean_pool(x, batch_ids)
            logits = self.graph_classifier(g)

        return logits, x, edge_pairs_features
    
    # --- NEW: load link predictor weights into this classifier ---
    def load_from_link_predictor(
        self,
        src,                      # path to .pth/.pt, nn.Module, or state_dict
        device: str = "cpu",
        map_encoder: bool = True,
        map_gcn: bool = True,
        strict_shapes: bool = True,
        verbose: bool = True,
    ):
        """
        Load weights from a SentenceGraphLinkPredictor (or compatible) into this classifier.
        Copies matching keys for:
          - encoder.* (HF encoder inside SentenceEncoder)
          - gcn.*     (PyG conv layer)
        src: checkpoint path, module instance, or state_dict
        """
        # Resolve state_dict from various inputs
        if isinstance(src, str):
            ckpt = torch.load(src, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                src_sd = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict):
                src_sd = ckpt
            else:
                raise ValueError("Unsupported checkpoint format.")
        elif isinstance(src, nn.Module):
            src_sd = src.state_dict()
        elif isinstance(src, dict):
            src_sd = src
        else:
            raise ValueError("src must be a path, a nn.Module, or a state_dict dict.")

        tgt_sd = self.state_dict()
        updated = 0
        skipped = []

        def copy_block(prefix: str):
            nonlocal updated
            for k, v in src_sd.items():
                if not k.startswith(prefix):
                    continue
                if k in tgt_sd:
                    if (not strict_shapes) or (tgt_sd[k].shape == v.shape):
                        tgt_sd[k] = v
                        updated += 1
                    else:
                        skipped.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))
                # allow mapping common subkeys if direct prefix mismatch (rare)

        if map_encoder:
            # SentenceEncoder parameters live under 'encoder.encoder.*'
            copy_block("encoder.encoder.")
        if map_gcn:
            copy_block("gcn.")

        self.load_state_dict(tgt_sd, strict=False)

        if verbose:
            print(f"[Init] Loaded {updated} params from link predictor into classifier.")
            if skipped:
                print(f"[Init] Skipped {len(skipped)} due to shape mismatch (set strict_shapes=False to ignore):")
                for name, s_src, s_tgt in skipped[:10]:
                    print(f"  - {name}: src{repr(s_src)} -> tgt{repr(s_tgt)}")
                if len(skipped) > 10:
                    print(f"  ... and {len(skipped) - 10} more")


class SentenceGATGraphClassifier(SentenceGGraphClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean"
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=GATConv,
            pooling=pooling
        )


class SentenceSAGEGraphClassifier(SentenceGGraphClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean"
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=SAGEConv,
            pooling=pooling
        )

class SentenceDGConvGraphClassifier(nn.Module):
    """
    DGCNN-style variant: stacks multiple GNN layers and concatenates intermediate features,
    then applies graph pooling and classifies.
    Returns (logits, node_features_concat, edge_pairs_features) for compatibility.
    """
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        GNN=GCNConv,
        pooling: str = "mean"
    ):
        super().__init__()
        self.encoder = SentenceEncoder(model_name=encoder_model, require_grad=encoder_grad)
        in_dim = self.encoder.encoder.config.hidden_size

        self.gcn = ModuleList()
        self.gcn.append(GNN(in_dim, hidden_channels))
        for _ in range(0, num_layers - 1):
            self.gcn.append(GNN(hidden_channels, hidden_channels))
        self.gcn.append(GNN(hidden_channels, 1))

        self.final_latent_dim = hidden_channels * num_layers + 1
        self.pooling = pooling.lower()
        self.graph_classifier = nn.Linear(self.final_latent_dim, num_classes)

    def forward(self, tokenized_inputs, edge_index, edge_pairs=None, batch_ids=None):
        enc_out = self.encoder(tokenized_inputs)
        x0 = enc_out[0]
        xs = [x0]
        for conv in self.gcn:
            xs.append(conv(xs[-1], edge_index).tanh())
        x = torch.cat(xs[1:], dim=-1)

        if edge_pairs is None:
            edge_pairs = edge_index.t()
        src = x[edge_pairs[:, 0]]
        tgt = x[edge_pairs[:, 1]]
        edge_pairs_features = torch.cat([src, tgt], dim=1)

        if batch_ids is None:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch_vec = batch_ids

        if self.pooling == "sum":
            g = global_add_pool(x, batch_vec)
        elif self.pooling == "max":
            g = global_max_pool(x, batch_vec)
        else:
            g = global_mean_pool(x, batch_vec)

        logits = self.graph_classifier(g)
        if batch_ids is None:
            logits = logits.squeeze(0)

        return logits, x, edge_pairs_features
    
class SentenceDGATGraphClassifier(SentenceDGConvGraphClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean"
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=GATConv,
            pooling=pooling
        )

class SentenceDGSAGEGraphClassifier(SentenceDGConvGraphClassifier):
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        encoder_model: str = "bert-base-uncased",
        encoder_grad: bool = False,
        pooling: str = "mean"
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            encoder_model=encoder_model,
            encoder_grad=encoder_grad,
            GNN=SAGEConv,
            pooling=pooling
        )


def sample_negative_edges(num_nodes, positive_edges, num_samples):
    """Sample negative edges not in the positive set."""
    neg_edges = set()
    pos_set = set((u.item(), v.item()) for u, v in positive_edges)
    while len(neg_edges) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in pos_set and (v, u) not in pos_set:
            neg_edges.add((u, v))
    return torch.tensor(list(neg_edges), dtype=torch.long)

def sample_negative_edges_v2(num_nodes, positive_edges, num_samples):
    """
    Sample negative edges that are not in the positive set and have no shortest path between nodes.

    Args:
        num_nodes (int): Number of nodes in the graph.
        positive_edges (torch.Tensor): Tensor of positive edges (direct connections).
        graph (networkx.Graph): Graph representation for shortest path computation.
        num_samples (int): Number of negative edges to sample.

    Returns:
        torch.Tensor: Tensor of sampled negative edges.
    """
    # create a graph from positive edges
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(positive_edges.tolist())
    neg_edges = set()
    pos_set = set((u.item(), v.item()) for u, v in positive_edges)

    while len(neg_edges) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in pos_set and (v, u) not in pos_set:
            try:
                # Check if a shortest path exists between u and v
                if not nx.has_path(graph, u, v):
                    neg_edges.add((u, v))
            except nx.NetworkXError:
                # Handle cases where nodes are not in the graph
                neg_edges.add((u, v))

    return torch.tensor(list(neg_edges), dtype=torch.long)


# -- Build per-document graph data object --
def create_document_graph(sentences, edge_index, edge_pairs, labels, node_labels=None):
    output = {
        'sentences': sentences,
        'edge_index': edge_index,
        'edge_pairs': edge_pairs,
        'labels': labels,
    }
    if node_labels is not None:
        output['node_labels'] = node_labels
    return output

def precision_at_k(logits, labels, k=2):
    """Compute precision@k for binary relevance."""
    # print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
    topk = torch.topk(logits, k=k).indices
    relevant = labels[topk]
    return relevant.sum().item() / k

def hit_at_1(pos_scores, neg_scores):
    """
    Compute Hit@1 metric for link prediction.
    
    Args:
        pos_scores (Tensor): [N] scores for positive edges (e.g., model outputs).
        neg_scores (Tensor): [N, K] scores for K negative edges per positive edge.

    Returns:
        hit1: float = (number of times positive edge ranked highest) / N
    """
    N = pos_scores.size(0)
    K = neg_scores.size(1)

    # Compare pos_score to each of its K negative scores
    better = pos_scores.view(-1, 1) > neg_scores  # shape: [N, K]

    # Hit@1: count if positive score is greater than all K negatives
    hit1 = (better.sum(dim=1) == K).float().mean().item()
    return hit1

def evaluate_link_prediction(model, documents, device="cpu", k=2, threshold=0.5, 
                             use_probabilities=False):
    model.eval()
    all_logits = []
    all_labels = []
    total_precision_at_k = 0

    with torch.no_grad():
        for doc in documents:
            edge_index = doc['edge_index'].to(device)
            edge_pairs = doc['edge_pairs'].to(device)
            labels = doc['labels'].to(device)
            sentences = doc['sentences']
            tokenized_inputs = model.encoder.tokenizer(sentences, padding=True, 
                                                       truncation=True, 
                                                       return_tensors="pt").to(device)
            # logits = model(sentences, edge_index, edge_pairs)
            logits = model(tokenized_inputs, edge_index, edge_pairs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            total_precision_at_k += precision_at_k(logits, labels, k)

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    # calculate link or no link based on threshold 0.5 1 for true, 0 for false
    # binary_labels = (labels_cat > 0.5).float()

    if use_probabilities:
        probs = torch.sigmoid(logits_cat)
        preds = (probs.numpy() >= threshold).astype("int32")
    else:
        preds = (logits_cat.numpy() >= threshold).astype("int32")


    auc = roc_auc_score(labels_cat.numpy(), preds)
    avg_p_at_k = total_precision_at_k / len(documents)

    # print("Labels cat:", labels_cat, preds)

    pre, rec, f1, _ = precision_recall_fscore_support(
        labels_cat.numpy(), 
        preds,
        average='binary')

    print(f"[Eval] AUC: {auc:.4f}, Precision@{k}: {avg_p_at_k:.4f}")
    print(f"[Eval] Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return auc, avg_p_at_k

def evaluate_node_classification(model, documents, device="cpu"):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for doc in documents:
            edge_index = doc['edge_index'].to(device)
            node_labels = doc['node_labels'].to(device)
            sentences = doc['sentences']
            tokenized_inputs = model.encoder.tokenizer(sentences, padding=True, 
                                                       truncation=True, 
                                                       return_tensors="pt").to(device)
            logits, _, _ = model(tokenized_inputs, edge_index)  # Get logits and edge features
            all_logits.append(logits.cpu())
            all_labels.append(node_labels.cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    pre, rec, f1, _ = precision_recall_fscore_support(
        labels_cat.numpy(), 
        logits_cat.argmax(dim=1).numpy(), 
        average='macro')

    print(f"[Eval] Node Classification - Precision: {pre:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    return pre, rec, f1

def main():
    # Example for 2 documents
    epochs = 3
    encoder_name = "bert-base-uncased"
    learning_rate = 1e-4

    config = {
        "encoder_model": encoder_name,
        "learning_rate": learning_rate,
        "num_epochs": epochs,
        # add additional settings as needed
    }

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

    # --- Initialize model and optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    link_prediction = SentenceGraphLinkPredictor(encoder_model=encoder_name, encoder_grad=True).to(device)
    # # model = SentenceGraphSAGELinkPredictor(encoder_model=encoder_name, encoder_grad=True).to(device)
    # # model = SentenceLinkPredictor(encoder_model=encoder_name, encoder_grad=False).to(device)
    # num_nodes = [len(doc['sentences']) for doc in documents]
    # model = SentenceDGCNN(hidden_channels=32, num_layers=3,
    #                       encoder_model=encoder_name, 
    #                       encoder_grad=True).to(device)
    # node_classifier = SentenceDGConvNodeClassifier(num_class=3, hidden_channels=32, num_layers=3,
    #                                                encoder_model=encoder_name,
    #                                                encoder_grad=True).to(device)

    node_classifier = SentenceGraphNodeClassifier(num_classes=3, hidden_dim=32,
                                                  encoder_model=encoder_name,
                                                  encoder_grad=True).to(device)                                              

    # model = link_prediction  # or node_classifier, depending on your task
    model = node_classifier  # or link_prediction, depending on your task

    tokenizer = model.encoder.tokenizer
    # model = UnsupervisedLinkPredictor().to(device)
    # print(learning_rate, type(learning_rate))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # link_criterion = nn.BCEWithLogitsLoss()
    link_criterion = contrastive_criterion  # Use custom contrastive loss
    node_criterion = nn.CrossEntropyLoss()  # For node classification

    # --- Training loop ---
    for epoch in range(epochs):
        total_loss = 0
        for doc in documents:
            model.train()
            edge_index = doc['edge_index'].to(device)
            pos_edge_pairs = doc['edge_pairs'].to(device)
            link_labels = doc['labels'].to(device)
            sentences = doc['sentences']
            
            node_labels = doc.get('node_labels', None)

            tokenized_inputs = tokenizer(sentences, padding=True, 
                                         truncation=True, return_tensors="pt")

            # logits = model(sentences, edge_index, edge_pairs)
            # loss = criterion(logits, labels)

            # Negative sampling
            num_nodes = len(sentences)
            neg_edge_pairs = sample_negative_edges(num_nodes, pos_edge_pairs, len(pos_edge_pairs))
            # print(f"Negative edges: {neg_edge_pairs}")

            # Combine pos + neg
            edge_pairs = torch.cat([pos_edge_pairs, neg_edge_pairs], dim=0)
            link_labels = torch.cat([torch.ones(len(pos_edge_pairs)), torch.zeros(len(neg_edge_pairs))])
            # print(f"edge pairs: {edge_pairs}, size: {len(edge_pairs.size())}")
            # print(f"labels: {labels}, length: {len(labels)}")

            # Forward pass
            # logits = model(sentences, edge_index, edge_pairs)
            outputs = model(tokenized_inputs, edge_index, edge_pairs)
            # print(f"Logits: {logits}, shape: {logits.size()}")

            if isinstance(outputs, tuple) and node_labels is not None:
                # If model returns logits and edge features
                # transform node_labels to long tensor
                node_labels = node_labels.to(device).long()
                logits, _, _ = outputs
                # print(f"Logits shape: {logits.shape}")
                # print(f"Edge features shape: {edge_features.shape}")
                loss = node_criterion(logits, node_labels)
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
                loss = link_criterion(logits, link_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    if isinstance(model, (SentenceDGConvNodeClassifier, SentenceGraphNodeClassifier)):
        # Evaluate node classification
        p, r, f1 = evaluate_node_classification(model, documents, device=device)
        print(f"Node Classification - Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f1:.4f}")
    elif isinstance(model, (SentenceGraphLinkPredictor, SentenceGATLinkPredictor, 
                            SentenceGraphSAGELinkPredictor, SentenceDGCNN,
                            SentenceDGATLinkPredictor, SentenceDGSAGELinkPredictor)):
        # Evaluate link prediction
        k = 2  # Precision@k
        auc, avg_precision = evaluate_link_prediction(model, documents, device=device, k=k)
        print(f"Link Prediction - AUC: {auc:.4f}, Avg Precision@{k}: {avg_precision:.4f}")

    # save_checkpoint(model, config, optimizer=optimizer, 
    #                     epoch=epoch, loss=total_loss, 
    #                     filename="final_checkpoint.pth")


if __name__ == "__main__":
    main()