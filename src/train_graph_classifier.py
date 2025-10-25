import argparse
import logging
import os
from typing import List, Dict, Any, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch

from datasets import ClassLabel
from accelerate.utils import set_seed
from sklearn.metrics import precision_recall_fscore_support

from utils_cli import parse_args
from pipeline import get_graph_dataset
from graph_model import (
    SentenceGGraphClassifier,
    SentenceGATGraphClassifier,
    SentenceSAGEGraphClassifier,
    SentenceDGConvGraphClassifier,
    SentenceDGATGraphClassifier,
    SentenceDGSAGEGraphClassifier,
    save_checkpoint,
    # load_checkpoint
)

from graph_prompt import (
    SentenceGraphPromptClassifier,
    SentenceGATGraphPromptClassifier,
    SentenceSAGEGraphPromptClassifier,
)

from unigraph import UniGraphGGraphClassifier

# Device selection (CUDA -> MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


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


def train_graph_parse_args():
    train_parser = parse_args("Train graph or subgraph classification models from graph datasets")

    train_parser.add_argument("--task_label_key", type=str, default="graph_label",
                              help="Key for graph-level labels inside each document")
    train_parser.add_argument("--continue_training", action="store_true",
                              help="Continue training from a checkpoint")
    train_parser.set_defaults(continue_training=False)
    train_parser.add_argument("--model_type", type=str, default="SentenceGGraphClassifier",
                              help="Model class to use for graph classification")
    train_parser.add_argument("--encoder_grad", action="store_true",
                              help="Allow gradients for the encoder model")
    train_parser.add_argument("--disable_encoder_grad", action="store_false",
                              dest="encoder_grad",
                              help="Disable gradients for the encoder model")
    train_parser.add_argument("--filter_key", type=str, default="model_name",
                              help="Key to filter documents in the dataset")
    train_parser.add_argument("--filter_value", type=str, default=None,
                              help="Value to filter documents in the dataset")
    train_parser.add_argument("--do_eval", type=int, default=5,
                              help="Evaluate the model every n epochs")
    train_parser.add_argument("--unigraph_checkpoint", type=str, default=None,
                              help="Path to a UniGraph checkpoint to initialize weights")
    
    # NEW: initialize from a SentenceGraphLinkPredictor checkpoint
    train_parser.add_argument("--link_predictor_checkpoint", type=str, default=None,
                              help="Path to a SentenceGraphLinkPredictor checkpoint (.pt/.pth)")
    train_parser.set_defaults(encoder_grad=True)

    args = train_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def build_label_encoder(dataset, graph_label_key):
    unique = set()
    dist = {}
    for item in dataset:
        y = item.get(graph_label_key, None)
        if y is None:
            continue
        if isinstance(y, list):
            for v in y:
                unique.add(v)
                dist[v] = dist.get(v, 0) + 1
        else:
            unique.add(y)
            dist[y] = dist.get(y, 0) + 1

    names = sorted(list(unique))
    encoder = ClassLabel(names=names)
    print(f"Graph label encoder with {len(names)} classes: {names}")
    return encoder, dist


def collate_graph_batch(
    docs: List[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    label_encoder: ClassLabel,
    task_label_key: str
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of graph docs into one batch:
      returns tokenized_inputs, batch_edge_index, batch_vec, targets
    """
    all_sentences: List[str] = []
    all_edges: List[torch.Tensor] = []
    batch_vec_list: List[torch.Tensor] = []
    targets: List[int] = []

    node_offset = 0
    for i, doc in enumerate(docs):
        sentences = doc["sentences"]
        ei = doc["edge_index"]
        if not torch.is_tensor(ei):
            ei = torch.tensor(ei, dtype=torch.long)
        num_nodes = len(sentences)
        num_edges = int(ei.size(1))

        all_sentences.extend(sentences)

        if num_edges > 0:
            all_edges.append(ei + node_offset)

        batch_vec_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        y = doc.get(task_label_key, None)
        if isinstance(y, list):
            y = y[0]
        y_idx = label_encoder.str2int(y) if isinstance(y, str) else int(y)
        targets.append(y_idx)

        node_offset = node_offset + num_nodes

    tokenized = tokenizer(
        all_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    if len(all_edges) > 0:
        batch_edge_index = torch.cat(all_edges, dim=1).to(device)
    else:
        batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    batch_vec = torch.cat(batch_vec_list, dim=0).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return tokenized, batch_edge_index, batch_vec, targets_tensor


def to_pyg_data(doc, label_encoder, task_label_key):
    sentences = doc["sentences"]
    ei = doc["edge_index"]
    if not torch.is_tensor(ei):
        ei = torch.tensor(ei, dtype=torch.long)
    y = doc.get(task_label_key, None)
    if isinstance(y, list):
        y = y[0]
    y_idx = label_encoder.str2int(y) if isinstance(y, str) else int(y)
    data = Data(
        edge_index=ei,
        y=torch.tensor(y_idx, dtype=torch.long),
        num_nodes=len(sentences)
    )
    data.sentences = sentences  # keep raw text; Batch will keep a list
    return data

def collate_graph_batch_pyg(
    docs, tokenizer, device, label_encoder, task_label_key
):
    data_list = [to_pyg_data(d, label_encoder, task_label_key) for d in docs]
    batch = Batch.from_data_list(data_list)  # provides batch.edge_index and batch.batch

    # gather and tokenize all sentences across graphs in this batch
    all_sentences = []
    for d in data_list:
        all_sentences.extend(d.sentences)
    tokenized = tokenizer(
        all_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    targets = batch.y.to(device)  # shape [num_graphs]
    return tokenized, batch, targets


@torch.no_grad()
def evaluate_graph_classification_loader(
    model,
    val_loader: DataLoader,
    device: torch.device = DEVICE
):
    model.eval()
    all_preds = []
    all_labels = []

    # for tokenized, batch_edge_index, batch_vec, targets in val_loader:
    for tokenized, batch, targets in val_loader:
        # logits, _, _ = model(tokenized, batch_edge_index, batch_ids=batch_vec)  # [G, C]
        logits, _, _ = model(tokenized, batch.edge_index, batch_ids=batch.batch)  # [G, C]
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(targets.detach().cpu().tolist())

    if len(all_labels) == 0:
        print("[Eval] No samples.")
        return 0.0, 0.0, 0.0

    pre, rec, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )
    print(f"[Eval] Graph Classification - Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    logging.info(f"[Eval] Graph Classification - Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return pre, rec, f1


def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device = DEVICE
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    # for tokenized, batch_edge_index, batch_vec, targets in train_loader:
    for tokenized, batch, targets in train_loader:
        # logits, _, _ = model(tokenized, batch_edge_index, batch_ids=batch_vec)  # [G, C]
        logits, _, _ = model(tokenized, batch.edge_index, batch_ids=batch.batch)  # [G, C]
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + float(loss.item())
        num_batches = num_batches + 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def main():
    args = train_graph_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info("args: %s", args)

    if args.seed is not None:
        set_seed(args.seed)

    def filter_function(doc):
        return doc.get(args.filter_key, "") != args.filter_value

    print("start loading the dataset", "path:", args.dataset_path)
    train_dataset = get_graph_dataset(
        data_name=args.dataset_name,
        model_name=args.model_name,
        path=args.dataset_path,
        split="train",
        label_key=args.label_key,
        padding="max_length",
        filter_function=filter_function,
        negative_sampling=False
    )
    val_dataset = get_graph_dataset(
        data_name=args.dataset_name,
        model_name=args.model_name,
        path=args.dataset_path,
        split="val",
        label_key=args.label_key,
        padding="max_length",
        filter_function=filter_function,
        negative_sampling=False
    )
    test_dataset = get_graph_dataset(
        data_name=args.dataset_name,
        model_name=args.model_name,
        path=args.dataset_path,
        split="test",
        label_key=args.label_key,
        padding="max_length",
        filter_function=filter_function,
        negative_sampling=False
    )
    print("finished loading the dataset")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    all_datasets = train_dataset + val_dataset
    _, train_dist = build_label_encoder(train_dataset, args.task_label_key)
    _, val_dist = build_label_encoder(val_dataset, args.task_label_key)
    label_encoder, _ = build_label_encoder(all_datasets, args.task_label_key)

    num_class = len(label_encoder.names)
    logger.info(f"Number of classes: {num_class}")
    logger.info(f"Train label distribution: {train_dist}")
    logger.info(f"Validation label distribution: {val_dist}")

    model_config = {
        "encoder_grad": args.encoder_grad,
        "num_classes": num_class,
    }
    if args.model_type in ["SentenceDGConvGraphClassifier", 
                           "SentenceDGATGraphClassifier", 
                           "SentenceDGSAGEGraphClassifier"]:
        model_config["num_layers"] = 3
        model_config["hidden_channels"] = 64

    if args.model_type == "UniGraphGGraphClassifier" and args.unigraph_checkpoint is not None:
        model_config["embedding_choice"] = "graph"  # or "node" or "combined"
        # optionally, load pretrained UniGraph weights
        model_config["checkpoint_path"] = args.unigraph_checkpoint

    model_cls = model_mapper(args.model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    model = model_cls(encoder_model=args.model_name, **model_config).to(DEVICE)
    tokenizer = model.encoder.tokenizer

    # NEW: optionally load link predictor weights into the classifier
    if getattr(args, "link_predictor_checkpoint", None):
        print(f"Initializing from link predictor: {args.link_predictor_checkpoint}")
        try:
            # Map encoder + GCN into the classifier
            if hasattr(model, "load_from_link_predictor"):
                model.load_from_link_predictor(
                    args.link_predictor_checkpoint,
                    device=DEVICE,
                    map_encoder=True,
                    map_gcn=True,
                    strict_shapes=True,
                    verbose=True,
                )
            else:
                print("Model has no load_from_link_predictor method; skipping.")
        except Exception as e:
            print(f"Failed to load link predictor weights: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if args.continue_training and args.checkpoint_dir:
        final_ckpt = os.path.join(args.checkpoint_dir, "final_checkpoint.pth")
        if os.path.exists(final_ckpt):
            logger.info(f"Loading model from checkpoint: {final_ckpt}")
            checkpoint = torch.load(final_ckpt, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logger.info("Loaded model state dict from checkpoint.")

    # DataLoaders with custom collate to form batched graphs
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_graph_batch_pyg(
            batch, tokenizer, DEVICE, label_encoder, args.task_label_key
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_graph_batch_pyg(
            batch, tokenizer, DEVICE, label_encoder, args.task_label_key
        ),
    )

    logger.info("Starting training the model")
    for epoch in range(1, args.num_train_epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device=DEVICE)
        logger.info(f"Epoch {epoch}/{args.num_train_epochs}, Average Loss: {avg_loss:.4f}")

        if args.do_eval and epoch % args.do_eval == 0:
            train_res = evaluate_graph_classification_loader(model, train_loader, device=DEVICE)
            val_res = evaluate_graph_classification_loader(model, val_loader, device=DEVICE)
            print(f"Epoch {epoch} - Train Results: {train_res}")
            print(f"Epoch {epoch} - Val Results: {val_res}")

    logger.info(f"checkpoint_dir: {args.checkpoint_dir}")

    evaluate_graph_classification_loader(model, val_loader, device=DEVICE)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_graph_batch_pyg(
            batch, tokenizer, DEVICE, label_encoder, args.task_label_key
        ),
    )
    test_res = evaluate_graph_classification_loader(model, test_loader, device=DEVICE)
    print(f"Test Results: {test_res}")

    if args.checkpoint_dir is not None:
        final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pth")
        save_checkpoint(
            model, {
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "num_train_epochs": args.num_train_epochs,
                    "batch_size": args.train_batch_size,
                    "model_name": args.model_name,
                    "model_type": args.model_type,
                    "loss_function": "CrossEntropyLoss",
                },
                "model_config": model_config,
                "args": args,
                "label_encoder": label_encoder,
            },
            optimizer=optimizer,
            epoch=args.num_train_epochs,
            filename=final_checkpoint_path
        )
        logger.info(f"Model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
