"""Main module for training models."""
import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['WANDB_SILENT'] = 'true'
import tensorflow as tf
from tensorflow import keras

from .training_utils import Preprocessing
from .models import (
    TimeSeriesTransformer,
    ConvCNP
)

from .callbacks import (
    WarmUpScheduler,
    LearningRateLogger,
    ReduceLRBacktrack
)

import wandb
from wandb.keras import WandbCallback
#wandb.init(project="master_thesis_kaan", entity="borgwardt")

tf.random.set_seed(87)
checkpoint_filepath = './checkpoints/cp.ckpt'
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Embedding Translational Equivariance to SeFT')
    parser.add_argument('--batch_size', type=int, default=16,
                        metavar="16", help='batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        metavar="200", help='number of epochs')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        metavar="1e-4", help='initial learning rate')
    parser.add_argument('--lr_warmup_steps', type=float, default=5e3,
                        metavar="5e3", help='learning rate warmup steps')
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        metavar="0.2", help='dropout rate')
    parser.add_argument('--norm_type', type=str, default='reZero',
                        metavar="reZero", help='normalization type')
    parser.add_argument('--dataset', type=str, default='physionet2012',
                        metavar='physionet2012', help='dataset name')
    parser.add_argument('--num_layers', type=int, default='1',
                        metavar='1', help='number of layers')
    parser.add_argument('--proj_dim', type=int, default='64',
                        metavar='64', help='projection dimension')
    parser.add_argument('--num_heads', type=int, default='2',
                        metavar='2', help='number of heads')
    parser.add_argument('--equivariance', default=False, 
                        action='store_true')
    parser.add_argument('--no_time', default=False,
                        action='store_true')
    parser.add_argument('--uni_mod', default=False,
                        action='store_true')
    parser.add_argument('--train_time_enc', default=False,
                        action='store_true')
    parser.add_argument('--time_weight', type=float, default=0.0,
                        metavar="0.0", help='weight of time encoding')
    parser.add_argument('--mod_weight', type=float, default=1.0,
                        metavar="1.0", help='weight of modality encoding')
    parser.add_argument('--points_per_hour', type=float, default=2.0,
                        metavar="2", help='points per hour for the grid')
    parser.add_argument('--kernel_size', type=int, default=5,
                        metavar="5", help='kernel size of conv layers')
    parser.add_argument('--dilation_rate', type=int, default=1,
                        metavar="1", help='dilation rate of conv layers')
    parser.add_argument('--filter_size', type=int, default=128, 
                        metavar="128", help='filter size of first conv layer')
    parser.add_argument('--drop_rate_conv', type=float, default=0.3, 
                        metavar="0.3", help='dropout rate of conv layers')
    parser.add_argument('--drop_rate_dense', type=float, default=0.2, 
                        metavar="0.2", help='dropout rate of dense layers')
    parser.add_argument('--model_name', type=str, default='ConvCNP',
                        metavar="convCNP", help='name of learning model')
    return parser.parse_args()

def main():
    """Parse command line arguments and train model."""
    args = parse_arguments()

    # Add hyperparameters to wandb config
    #wandb.config.update(args)

    ## Hyperparameters
    batch_size = args.batch_size  # Default: 16
    num_epochs = args.num_epochs  # Default: 200
    init_lr = args.init_lr  # Default: 1e-4
    lr_warmup_steps = args.lr_warmup_steps  # Default: 2e3
    drop_rate = args.drop_rate  # Default: 0.1
    norm_type = args.norm_type  # Default: 'reZero'
    dataset = args.dataset  # Default: 'physionet2012'
    num_layers = args.num_layers  # Default: 1
    proj_dim = args.proj_dim  # Default: 32
    num_heads = args.num_heads  # Default: 2
    equivariance = args.equivariance  # Default: False
    no_time = args.no_time  # Default: False
    uni_mod = args.uni_mod  # Default: False
    train_time_enc = args.train_time_enc  # Default: False
    time_weight = args.time_weight  # Default: 0.0
    mod_weight = args.mod_weight  # Default: 1.0
    points_per_hour = args.points_per_hour  # Default: 2
    kernel_size = args.kernel_size  # Default: 5
    dilation_rate = args.dilation_rate  # Default: 2
    filter_size = args.filter_size  # Default: 64
    drop_rate_conv = args.drop_rate_conv  # Default: 0.2
    drop_rate_dense = args.drop_rate_dense  # Default: 0.2
    model_name = args.model_name  # Default: ConvCNP

    ## Load data
    transformation = Preprocessing(
        dataset=dataset, epochs=num_epochs, batch_size=batch_size)

    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = \
        transformation._prepare_dataset_for_training()

    ## Initialize the model
    if model_name=='Transformer':
        model = TimeSeriesTransformer(
            proj_dim=proj_dim, num_head=num_heads,
            enc_dim=proj_dim, pos_ff_dim=proj_dim,
            pred_ff_dim=proj_dim/4, drop_rate=dropout_rate,
            norm_type=norm_type, dataset=dataset,
            equivar=equivariance, num_layers=num_layers,
            no_time=no_time, uni_mod=uni_mod,
            train_time_enc=train_time_enc,
            time_weight=time_weight, mod_weight=mod_weight
        )
    else:
         model = ConvCNP(
            points_per_hour=points_per_hour,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate, 
            filter_size=filter_size, 
            drop_rate_conv=drop_rate_conv, 
            drop_rate_dense=drop_rate_dense,
            dataset=dataset
        )

    ## Optimizer function
    opt = keras.optimizers.Adam(
        learning_rate=init_lr
    )

    ## Loss function
    if dataset == 'physionet2019':
        loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=False,
            name="loss",
            reduction=tf.keras.losses.Reduction.SUM
        )
    else:
        loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=False,
            name="loss"
        )

    ## Compile the model
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                 keras.metrics.AUC(curve='PR', name='auprc'),
                 keras.metrics.AUC(curve='ROC', name='auroc')]
    )

    # Callback for logging the learning rate for inspection
    lr_logger_callback = LearningRateLogger()

    # Callback for warmup scheduler
    lr_warmup_callback = WarmUpScheduler(
        final_lr=init_lr,
        warmup_steps=lr_warmup_steps
    )

    ## Callback for saving the weights of the best model
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    lr_schedule_callback = ReduceLRBacktrack(
        best_path=checkpoint_filepath,
        monitor='val_loss', 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-8
    )

    # Callback for early stopping when val_auprc does not improve anymore
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_auprc',
        mode='max',
        patience=12,
        restore_best_weights=True
    )

    # Fit the model to the input data
    print("\n------- Training and Validation -------")
    model.fit(
        train_iter,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_iter,
        validation_steps=val_steps,
        verbose=1,
        callbacks=[lr_schedule_callback,
                   lr_warmup_callback,
                   lr_logger_callback,
                   model_checkpoint_callback,
                   #WandbCallback(),
                   early_stopping_callback]
    )

    print("\n------- Test -------")
    # Fit the model to the input data
    model.evaluate(
        test_iter,
        steps=test_steps,
        verbose=1
    )
    print("\n")

if __name__ == "__main__":
    main()
    