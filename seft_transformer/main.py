"""Main module for training models."""
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax, device_put
from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn

import wandb
import numpy as np
from training_utils import Preprocessing, parse_arguments

tf.random.set_seed(83)
print("GPUs Available:", 'yes'
      if jax.default_backend()=='gpu' 
      else 'no')

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
    
    train_data = tfds.as_numpy(train_iter)
    
    for x, y in train_data:
        x = device_put(np.squeeze(x[2], axis=-1))
        print(x.shape)
        print(type(x))
        model = nn.Dense(features=32)
        sys.exit()
    
    # Initialize the model
    
    
if __name__ == "__main__":
    main()
    