import argparse
import logging
import os
# import time
# import random
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

from utils_cli import parse_args
# from utils_torch import trainer_by_step, trainer_by_epochs
from pipeline import get_graph_dataset

from graph_model import SentenceGraphLinkPredictor
from graph_model import SentenceLinkPredictor
from graph_model import SentenceDGCNN
from graph_model import SentenceGATLinkPredictor
from graph_model import SentenceGraphSAGELinkPredictor
from graph_model import SentenceDGATLinkPredictor
from graph_model import SentenceDGSAGELinkPredictor
from graph_model import evaluate_link_prediction
# from graph_model import save_checkpoint

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

def eval_graph_parse_args():
    eval_parser = parse_args("Evaluate text classification models from CSV dataset")

    eval_parser.add_argument("--model_type", type=str, default="SentenceGraphLinkPredictor",
                              help="Type of model to train (e.g., SentenceGraphLinkPredictor, \
                                    SentenceLinkPredictor)")
    eval_parser.add_argument("--filter_key", type=str, default="model_name",
                                help="Key to filter documents in the dataset")
    eval_parser.add_argument("--filter_value", type=str, default=None,
                                help="Value to filter documents in the dataset")

    args = eval_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def main():
    args = eval_graph_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    if args.seed is not None:
        set_seed(args.seed)

    model_config = {}

    # load final_checkpoint to get model config
    if args.checkpoint_dir is None:
        raise ValueError("Checkpoint directory is not specified.")
    # check final_checkpoint.pth
    model_checkpoint_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pth")
    
    checkpoint = None
    if model_checkpoint_path:
        # load the config from the checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE)
        checkpoint_model_config = checkpoint.get('config', {}).get('model_config', {})
        # print(f"Checkpoint model config: {checkpoint_model_config}")
        model_config.update(checkpoint_model_config)
        checkpoint_args = checkpoint.get('config', {}).get('args', {})
        print(f"Updating args from checkpoint: {checkpoint_args}")
        # update args with checkpoint args
        # args = checkpoint_args
        if isinstance(checkpoint_args, argparse.Namespace):
            checkpoint_args = vars(checkpoint_args)  # Convert Namespace to a dictionary

        for key, value in checkpoint_args.items():
            # if key != 'dataset_path' or key != 'checkpoint_dir' :
            no_edit_args = ['dataset_path', 'checkpoint_dir', 
                            'filter_key', 'filter_value']
            if key not in no_edit_args:
                setattr(args, key, value)
        print(f"Updated args: {args}")

    # filter the dataset with a custom filter function if provided
    def filter_function(doc):
        # write a filter function to filter out documents with model=='human'
        return doc.get(args.filter_key, '') != args.filter_value

    print("start loading the dataset", "path:", args.dataset_path)
    dataset = get_graph_dataset(data_name=args.dataset_name, 
                                model_name=args.model_name, 
                                path=args.dataset_path, split="test", 
                                label_key=args.label_key,
                                padding="max_length",
                                filter_function=filter_function)
    print("finished loading the dataset")

    print(f"Number of samples: {len(dataset)}")
    # print(train_dataset[0].keys())
    # print(f"First sample: {train_dataset[0]}")

    default_model_config = {
        "encoder_grad": True,
    }

    if args.model_type == "SentenceDGCNN":
        default_model_config["num_layers"] = 3
        default_model_config["hidden_channels"] = 64

    # model = SentenceGraphLinkPredictor(encoder_model=args.model_name, **model_config).to(DEVICE)
    model = model_mapper(args.model_type)
    if model is None:
        raise ValueError(f"Model type '{args.model_type}' is not recognized.")
    model = model(encoder_model=args.model_name, **model_config).to(DEVICE)

    # # load model from checkpoint
    # if args.checkpoint_dir is None:
    #     raise ValueError("Checkpoint directory is not specified.")

    # model_checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    # load model from checkpoint if it exists
    if os.path.exists(model_checkpoint_path):
        print(f"Loading model from checkpoint: {model_checkpoint_path}")
        # model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {model_checkpoint_path}")
    
    evaluate_link_prediction(model, dataset, device=DEVICE, k=1, 
                             use_probabilities=True)

if __name__ == "__main__":
    main()

