# import argparse
import logging
import os
# import time
import random
# from typing import Tuple

# import evaluate
import torch
# import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
# from torch.utils.data import DataLoader
# from torch.utils.data import ConcatDataset
# from transformers import DataCollatorWithPadding
# from transformers import default_data_collator

# from sklearn.metrics import precision_recall_fscore_support, classification_report

from torch_geometric.data import Data as PyGData
from torch_geometric.data import DataLoader

from utils_cli import parse_args
from utils_torch import trainer_by_step, trainer_by_epochs
from pipeline import get_graph_dataset

from graph_model import SentenceGraphLinkPredictor
from graph_model import SentenceLinkPredictor
from graph_model import SentenceDGCNN
from graph_model import SentenceGATLinkPredictor
from graph_model import SentenceGraphSAGELinkPredictor
from graph_model import SentenceDGATLinkPredictor
from graph_model import SentenceDGSAGELinkPredictor
from graph_model import evaluate_link_prediction
from graph_model import save_checkpoint
from graph_model import contrastive_criterion

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")

def model_mapper(model_name):
    """Map model names to their respective classes."""
    model_map = {
        "SentenceGraphLinkPredictor": SentenceGraphLinkPredictor,
        "SentenceLinkPredictor": SentenceLinkPredictor,
        "SentenceDGCNN": SentenceDGCNN,
        "SentenceGATLinkPredictor": SentenceGATLinkPredictor,
        "SentenceGraphSAGELinkPredictor": SentenceGraphSAGELinkPredictor,
        "SentenceDGATLinkPredictor": SentenceDGATLinkPredictor,
        "SentenceDGSAGELinkPredictor": SentenceDGSAGELinkPredictor,
        # Add other models here as needed
    }
    return model_map.get(model_name, None)

def loss_mapper(loss_name):
    """Map loss names to their respective functions."""
    loss_map = {
        "contrastive": contrastive_criterion,
        "bce": nn.BCEWithLogitsLoss(),
        # Add other losses here as needed
    }
    return loss_map.get(loss_name, nn.BCEWithLogitsLoss())

def train_graph_parse_args():
    train_parser = parse_args("Train text classification models from CSV dataset")

    train_parser.add_argument("--continue_training", action="store_true",
                              help="Continue training from a checkpoint")
    train_parser.set_defaults(continue_training=False)
    train_parser.add_argument("--model_type", type=str, default="SentenceGraphLinkPredictor",
                              help="Type of model to train (e.g., SentenceGraphLinkPredictor, \
                                    SentenceLinkPredictor)")
    train_parser.add_argument("--loss_function", type=str, default="BCEWithLogitsLoss",
                              help="Loss function to use (e.g., contrastive, BCEWithLogitsLoss)")
    train_parser.add_argument("--encoder_grad", action="store_true",
                              help="Whether to allow gradients for the encoder model")
    train_parser.add_argument("--disable_encoder_grad", action="store_false",
                              dest="encoder_grad",
                              help="Disable gradients for the encoder model")
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

    return args


def main():
    args = train_graph_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

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

    # print(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    # print(train_dataset[0].keys())
    # print(f"First sample: {train_dataset[0]}")

    model_config = {
        "encoder_grad": args.encoder_grad,
    }

    if args.model_type in ["SentenceDGCNN", "SentenceDGATLinkPredictor", "SentenceDSAGELinkPredictor"]:
        model_config["num_layers"] = 3
        model_config["hidden_channels"] = 64


    # model = SentenceGraphLinkPredictor(encoder_model=args.model_name, **model_config).to(DEVICE)
    model = model_mapper(args.model_type)(encoder_model=args.model_name, **model_config).to(DEVICE)
    tokenizer = model.encoder.tokenizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = loss_mapper(args.loss_function)  # Use mapped loss function


    # Create a dataset
    # train_dataset = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Training loop
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        model.train()

        # shuffle the dataset
        # shuffle list train_dataset
        random.shuffle(train_dataset)

        for graph_doc in train_dataset:
            # print(f"Processing graph document: {graph_doc.keys()}")
            edge_index = graph_doc['edge_index'].to(DEVICE)
            edge_pairs = graph_doc['edge_pairs'].to(DEVICE)
            labels = graph_doc['labels'].to(DEVICE)
            sentences = graph_doc['sentences']

            # Tokenize sentences
            tokenized_inputs = tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)

            # Forward pass
            logits = model(tokenized_inputs, edge_index, edge_pairs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            average_loss = total_loss / len(train_dataset)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Average Loss: {average_loss:.4f}")
        # use logging to log the loss
        logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Average Loss: {average_loss:.4f}")

    evaluate_link_prediction(model, train_dataset, device=DEVICE, k=1)

    hyperparameters_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "batch_size": args.train_batch_size,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "loss_function": args.loss_function,
    }

    all_config = {
        "hyperparameters": hyperparameters_config,
        "model_config": model_config,
        "args": args,
    }

    if args.checkpoint_dir is not None:
        # torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))
        save_checkpoint(model, all_config, optimizer=optimizer,
                        epoch=args.num_train_epochs, loss=total_loss,
                        filename=os.path.join(args.checkpoint_dir, "final_checkpoint.pth"))
        logger.info(f"Model saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
