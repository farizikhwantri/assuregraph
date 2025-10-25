import argparse

def parse_args(description="Train text classification models"):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/bart-base",
        help="A name of the pre-trained model.",
    )

    # dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mnli",
        help="A name of GLUE dataset.",
    )

    parser.add_argument(
        "--label_key",
        type=str,
        default="label",
        help="A keyname to access the label in the dataset.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/reqnli/task-1-data-nli.csv",
        help="The local dataset path to load",
    )

    parser.add_argument(
        "--num_labels", 
        type=int, default=None,
        help="Number of labels in the dataset")
    
    parser.add_argument(
        "--fast_tokenizer",
        action="store_true",
        dest="fast_tokenizer",
        help="Whether to use a fast tokenizer (default: True)",
    )
    parser.set_defaults(fast_tokenizer=True)

    parser.add_argument(
        "--slow_tokenizer",
        action="store_false",
        dest="fast_tokenizer",
        help="set to use slow tokenizer",
    )


    parser.add_argument(
        "--max_seq_length", 
        type=int, default=128, 
        help="Maximum sequence length")

    # training arguments
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-05,
        help="Fixed learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of epochs to train the model.",
    )

    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=0,
        help="Total number of steps to train the model.",
    )

    # add gradient accumulation steps
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )

    # evaluation arguments
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )


    # other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="A seed for reproducible training pipeline.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="A path to store the final checkpoint.",
    )

    return parser

