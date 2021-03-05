"""Main module for training models."""
import argparse
import os
import sys
import tensorflow as tf
import pdb
from tensorflow import keras

from .training_utils import Preprocessing
from .models import TimeSeriesTransformer, TimeSeriesTransformer_v2
from .callbacks import WarmUpScheduler, LearningRateLogger

import wandb
from wandb.keras import WandbCallback
wandb.init(project="master_thesis_kaan", entity="borgwardt")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

checkpoint_filepath = './checkpoints/cp.ckpt'
log_dir = "./logs"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(83)
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
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        metavar="0.5", help='decay rate of learning rate')
    parser.add_argument('--lr_warmup_steps', type=float, default=2e3,
                        metavar="2e3", help='learning rate warmup steps')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        metavar="0.1", help='dropout rate')
    parser.add_argument('--norm_type', type=str, default='reZero',
                        metavar="reZero", help='normalization type')
    parser.add_argument('--dataset', type=str, default='physionet2012',
                        metavar='physionet2012', help='dataset name')
    parser.add_argument('--num_layers', type=int, default='1',
                        metavar='1', help='number of layers')
    parser.add_argument('--proj_dim', type=int, default='32',
                        metavar='32', help='projection dimension')
    parser.add_argument('--num_heads', type=int, default='2',
                        metavar='2', help='number of heads')
    parser.add_argument('--equivariance', default=False, 
                        action='store_true')
    parser.add_argument('--no_time', default=False,
                        action='store_true')
    parser.add_argument('--uni_mod', default=False,
                        action='store_true')
    parser.add_argument('--no_mod', default=False,
                        action='store_true')
    return parser.parse_args()

def main():
    """Parse command line arguments and train model."""
    args = parse_arguments()

    # Add hyperparameters to wandb config
    wandb.config.update(args)

    # Hyperparameters
    batch_size = args.batch_size  # Default: 16
    num_epochs = args.num_epochs  # Default: 200
    init_lr = args.init_lr  # Default: 1e-4
    lr_decay_rate = args.lr_decay_rate  # Default: 0.5
    lr_warmup_steps = args.lr_warmup_steps  # Default: 2e3
    dropout_rate = args.dropout_rate  # Default: 0.1
    norm_type = args.norm_type  # Default: 'reZero'
    dataset = args.dataset  # Default: 'physionet2012'
    num_layers = args.num_layers  # Default: 1
    proj_dim = args.proj_dim  # Default: 32
    num_heads = args.num_heads  # Default: 2
    equivariance = args.equivariance  # Default: False
    no_time = args.no_time  # Default: False
    uni_mod = args.uni_mod  # Default: False
    no_mod = args.no_mod  # Default: False

    # Load data
    transformation = Preprocessing(
        dataset=dataset, epochs=num_epochs, batch_size=batch_size)

    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = \
        transformation._prepare_dataset_for_training()
    
    # Initialize the model
    if no_mod:
        model = TimeSeriesTransformer_v2(
            proj_dim=proj_dim, num_head=num_heads,
            enc_dim=proj_dim, pos_ff_dim=proj_dim,
            pred_ff_dim=proj_dim/4, drop_rate=dropout_rate,
            norm_type=norm_type, equivar=equivariance,
            no_time=no_time, num_layers=num_layers
        )
    else:
        model = TimeSeriesTransformer(
            proj_dim=proj_dim, num_head=num_heads,
            enc_dim=proj_dim, pos_ff_dim=proj_dim,
            pred_ff_dim=proj_dim/4, drop_rate=dropout_rate,
            norm_type=norm_type, dataset=dataset,
            equivar=equivariance, num_layers=num_layers,
            no_time=no_time, uni_mod=uni_mod
        )

    # Experiment logs folder
    experiment_log = os.path.join(
        log_dir,
        dataset,
        'ex_batchSize_' + str(batch_size) +
        '_projDim_' + str(proj_dim) +
        '_transEq_' + str(equivariance) +
        '_numHead_' + str(num_heads) +
        '_dropRate_' + str(dropout_rate) +
        '_normType_' + norm_type +
        '_uniMod_' + str(uni_mod) +
        '_timeEnc_' + str(not no_time) +
        '_noMod_' + str(no_mod) +
        '_numLayer_' + str(num_layers)
    )

    # File to log variables e.g. learning rate
    file_writer = tf.summary.create_file_writer(experiment_log + "/variables")
    file_writer.set_as_default()

    # Optimizer function
    opt = keras.optimizers.Adam(
        learning_rate=init_lr
    )

    # Loss function
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

    # Compile the model
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

    # Callback for reducing the learning rate when loss get stuck in a plateau
    lr_schedule_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=lr_decay_rate,
        patience=5,
        min_lr=1e-8
    )

    # Callback for early stopping when val_loss does not improve anymore
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True
    )

    # Callback for saving the weights of the best model
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Callback for Tensorboard logging
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=experiment_log,
        update_freq='epoch',
        profile_batch='100,110'
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
        callbacks=[#tensorboard_callback,
                   #model_checkpoint_callback,
                   early_stopping_callback,
                   WandbCallback(),
                   lr_schedule_callback,
                   lr_warmup_callback,
                   lr_logger_callback]
    )

    model.save(os.path.join(wandb.run.dir, "model.h5"))

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
    