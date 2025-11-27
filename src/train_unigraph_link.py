# import argparse
import logging
import os
# import time
# import random
from typing import Dict, Any, List

# import evaluate
import torch
# import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
# from torch.utils.data import ConcatDataset
# from transformers import DataCollatorWithPadding
# from transformers import default_data_collator

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch
# from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling

from utils_cli import parse_args
# from utils_torch import trainer_by_step, trainer_by_epochs
from pipeline import get_graph_dataset

# from unigraph import UniGraph
from unigraph import UniGraphLinkPredictor

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")



def train_graph_parse_args():
    train_parser = parse_args("Train unigraph link prediction model")

    train_parser.add_argument("--continue_training", action="store_true",
                              help="Continue training from a checkpoint")

    # train_parser.add_argument("--lm_type", type=str, default="bert-base-uncased")
    train_parser.add_argument("--hidden_size", type=int, default=768)
    train_parser.add_argument("--num_heads", type=int, default=8)
    train_parser.add_argument("--num_layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lam", type=float, default=0.1, help="Latent regularization weight")
    train_parser.add_argument("--log_interval", type=int, default=10)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--mask_rate", type=float, default=0.15, help="Masking rate for MLM")
    train_parser.add_argument("--train_mode", type=str, default="pretrain",
                              choices=["pretrain", "finetune", "both"],
                              help="Training mode: pretrain (MLM + link), finetune (link only), both")
    train_parser.add_argument("--continue_from_checkpoint_dir", type=str, default=None,
                              help="Directory to load checkpoint from to continue training")
    train_parser.add_argument("--filter_key", type=str, default="model_name",
                                help="Key to filter documents in the dataset")
    train_parser.add_argument("--filter_value", type=str, default=None,
                                help="Value to filter documents in the dataset")
    train_parser.set_defaults(encoder_grad=True)

    args = train_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # if args.continue_from_checkpoint_dir is not None:
    #     if not os.path.exists(args.continue_from_checkpoint_dir):
    #         raise ValueError(f"Continue from checkpoint directory {args.continue_from_checkpoint_dir} does not exist.")
    #     args.continue_from_checkpoint_dir = os.path.join(args.continue_from_checkpoint_dir, args.dataset_name)
    #     if not os.path.exists(args.continue_from_checkpoint_dir):
    #         raise ValueError(f"Continue from checkpoint directory {args.continue_from_checkpoint_dir} does not exist.")

    return args

def prepare_batch_for_unigraph(documents: List[PyGData], tokenizer, device,
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
        graph = PyGData(
            x=None,  # Node features will be obtained from LM
            edge_index=doc["edge_index"],
            edge_pairs=doc["edge_pairs"],
            node_labels=doc["node_labels"],
            labels=doc["labels"],
            num_nodes=len(doc["sentences"])
        )
        graphs.append(graph)

    batch = Batch.from_data_list(graphs).to(device)


    tokenized = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128  # Hard limit for BERT-based models
    )
    
    # Create masked inputs for MLM
    input_ids = tokenized.input_ids.to(device)
    # attention_mask = tokenized.attention_mask.to(device)
    # token_type_ids = tokenized.token_type_ids.to(device)
    
    # Create masked input ids
    # masked_input_ids = input_ids.clone()
    # rand = torch.rand(input_ids.shape).to(device)
    # mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)
    # selection = torch.flatten(mask_arr.nonzero()).tolist()
    # masked_input_ids[selection] = tokenizer.mask_token_id
    # masked_input_ids = masked_input_ids.to(device)

    # Create masked input ids (BERT-style 15% masking)
    masked_input_ids = input_ids.clone()

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

        if args.train_mode == "finetune":
            link_loss.backward()
        elif args.train_mode == "both":
            total_loss = pretrain_loss + link_loss
            total_loss.backward()
        else:
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
    all_probs = []
    all_preds = []

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
            all_labels.extend(link_labels.cpu().numpy().tolist())
            all_probs.extend(link_probs.cpu().numpy().tolist())
            all_preds.extend(predictions.cpu().numpy().tolist())


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
    args = train_graph_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    device = DEVICE

    if args.seed is not None:
        set_seed(args.seed)

    # filter the dataset with a custom filter function if provided
    def filter_function(doc):
        # write a filter function to filter out documents with model=='human'
        return doc.get(args.filter_key, '') != args.filter_value
        # return doc.get('model', '') != 'human'  # Example filter condition

    print("start loading the dataset", "path:", args.dataset_path)
    train_dataset = get_graph_dataset(data_name=args.dataset_name, 
                                      model_name=args.model_name, 
                                      path=args.dataset_path, split="train", 
                                      label_key=args.label_key,
                                      padding="max_length",
                                      filter_function=filter_function)
    print("finished loading the dataset")
    logger.info(f"Number of training samples: {len(train_dataset)}")

    test_dataset = get_graph_dataset(data_name=args.dataset_name,
                                    model_name=args.model_name, 
                                    path=args.dataset_path, split="test", 
                                    label_key=args.label_key,
                                    padding="max_length",
                                    filter_function=filter_function)
    logger.info(f"Number of test samples: {len(test_dataset)}")

    # make a dataloader
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.train_batch_size, 
                              shuffle=True, 
                              collate_fn=lambda batch: batch)

    # Initialize model
    model = UniGraphLinkPredictor(
        lm_type=args.model_name,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lam=args.lam
    ).to(device)

    # checkpoint_loaded = False
    # if checkpoint is not empty and exists, load from checkpoint
    if args.continue_from_checkpoint_dir and os.path.exists(f"{args.continue_from_checkpoint_dir}/unigraph_link_predictor.pth"):
        print(f"Loading model from {args.continue_from_checkpoint_dir}")
        model.load_state_dict(torch.load(f"{args.continue_from_checkpoint_dir}/unigraph_link_predictor.pth", 
                                         map_location=device))
        # checkpoint_loaded = True
    elif args.checkpoint_dir and os.path.exists(f"{args.checkpoint_dir}/unigraph_link_predictor.pth"):
        print(f"Loading model from {args.checkpoint_dir}")
        model.load_state_dict(torch.load(f"{args.checkpoint_dir}/unigraph_link_predictor.pth", 
                                         map_location=device))
        # checkpoint_loaded = True
    
    # Initialize optimizer
    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Training loop
    print("Starting UniGraph Link Prediction Pretraining...")

    for epoch in range(args.num_train_epochs):
        pretrain_loss, pretrain_latent_loss = train_pretrain(
            args, model, train_loader, pretrain_optimizer, epoch
        )

        print(f"Epoch {epoch + 1}/{args.num_train_epochs} completed")
        print(f"Pretrain Loss: {pretrain_loss:.4f}")
        print(f"Latent Loss: {pretrain_latent_loss:.4f}")
        print("-" * 50)
    
    print("Pretraining completed!")

    # evaluation on the training set itself (for demonstration)
    print("Evaluating on training set...")
    eval_results = evaluate_link_prediction(model, train_loader)
    print("Evaluation results:", eval_results)

    test_eval_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, 
                                  shuffle=False, collate_fn=lambda batch: batch)
    print("Evaluating on test set...")
    test_results = evaluate_link_prediction(model, test_eval_loader)
    print("Test set results:", test_results)

    if args.checkpoint_dir is not None and args.num_train_epochs > 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'unigraph_link_predictor.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
