"""Main module for training models."""
import os
import sys
import tensorflow as tf

from .training_utils import Preprocessing, parse_arguments

import wandb
from wandb.keras import WandbCallback

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(83)
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

def main():
    """Parse command line arguments and train model."""
    args = parse_arguments()

    # Hyperparameters
    batch_size = args.batch_size  # Default: 16
    num_epochs = args.num_epochs  # Default: 200
    init_lr = args.init_lr  # Default: 1e-4
    lr_warmup_steps = args.lr_warmup_steps  # Default: 2e3
    dropout_rate = args.dropout_rate  # Default: 0.1
    norm_type = args.norm_type  # Default: 'reZero'
    dataset = args.dataset  # Default: 'physionet2012'
    num_layers = args.num_layers  # Default: 1
    proj_dim = args.proj_dim  # Default: 32
    num_heads = args.num_heads  # Default: 2
    equivariance = args.equivariance  # Default: False
    no_time = args.no_time  # Default: False
    ax_attn = args.ax_attn  # Default: False
    train_time_enc = args.train_time_enc  # Default: False

    # Load data
    transformation = Preprocessing(
        dataset=dataset, epochs=num_epochs, batch_size=batch_size)

    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = \
        transformation._prepare_dataset_for_training()
    
    # Initialize the model
    
    
if __name__ == "__main__":
    main()
    