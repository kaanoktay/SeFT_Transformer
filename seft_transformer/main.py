"""Main module for training models."""
import argparse
import os
import tensorflow as tf
import pdb
from tensorflow import keras

from .training_utils import Preprocessing
from .models import TimeSeriesTransformer
from .callbacks import WarmUpScheduler, LearningRateLogger


tf.executing_eagerly()
checkpoint_filepath = './checkpoints/cp.ckpt'
log_dir = "./logs"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(0)
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Embedding Translational Equivariance to SeFT')
    parser.add_argument('--batch_size', type=int, default=16,
                        metavar="16", help='batch size')
    parser.add_argument('--num_epochs', type=int, default=500,
                        metavar="500", help='number of epochs')
    parser.add_argument('--init_learning_rate', type=float, default=1e-4,
                        metavar="1e-4", help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.25,
                        metavar="0.25", help='decay rate of learning rate')
    parser.add_argument('--lr_warmup_steps', type=float, default=1e3,
                        metavar="1e3", help='learning rate warmup steps')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        metavar="0.1", help='dropout rate')
    parser.add_argument('--skip_connection', type=str, default="layer_norm",
                        metavar="layer_norm", help='type of skip connection')                        
    return parser.parse_args()

def main():
    """Parse command line arguments and train model."""
    args = parse_arguments()

    # Hyperparameters
    batch_size = args.batch_size  # Default: 16
    num_epochs = args.num_epochs  # Default: 500
    init_learning_rate = args.init_learning_rate  # Default: 1e-4
    lr_decay_rate = args.lr_decay_rate  # Default: 0.25
    lr_warmup_steps = args.lr_warmup_steps # Default: 1e3
    dropout_rate = args.dropout_rate # Default: 0.1
    skip_connection_type = args.skip_connection # Default: "layer_norm"

    # Experiment logs folder
    experiment_log = os.path.join(
        log_dir,
        "ex_initLr_" + str(init_learning_rate) +
        "_dropRate_" + str(dropout_rate) +
        "_warmSteps_" + str(lr_warmup_steps)
    )

    # File to log variables e.g. learning rate
    file_writer = tf.summary.create_file_writer(experiment_log + "/variables")
    file_writer.set_as_default()

    # Load data (epochs don't matter because we iterate over the dataset
    # indefinitely)
    transformation = Preprocessing(
        dataset='physionet2012', epochs=num_epochs, batch_size=batch_size)
    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = \
        transformation._prepare_dataset_for_training()

    # Initialize the model
    model = TimeSeriesTransformer(
        proj_dim=128, num_head=4, enc_dim=128, 
        pos_ff_dim=128, pred_ff_dim=32, drop_rate=dropout_rate
    )

    # Optimizer function
    opt = keras.optimizers.Adam(
        learning_rate=init_learning_rate
    )

    # Loss function
    loss_fn = keras.losses.BinaryCrossentropy(
        from_logits=False,
        name="loss"
    )

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(curve="PR", name="auprc"),
                 keras.metrics.AUC(curve="ROC", name="auroc")]
    )

    # Callback for logging the learning rate for inspection
    lr_logger_callback = LearningRateLogger()

    # Callback for warmup scheduler
    lr_warmup_callback = WarmUpScheduler(
        final_lr=init_learning_rate,
        warmup_steps=lr_warmup_steps
    )

    # Callback for reducing the learning rate when loss get stuck in a plateau
    lr_schedule_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        mode='min',
        factor=lr_decay_rate,
        patience=6,
        min_lr=1e-8
    )

    # Callback for early stopping when val_loss does not improve anymore
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
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
        profile_batch='350, 355'
    )

    # Fit the model to the input data
    print("\n------- Training and Validation -------")
    model.fit(
        train_iter,
        epochs=num_epochs,
        # TODO(Max): Are you sure about the -1 ?
        steps_per_epoch=steps_per_epoch-1,
        validation_data=val_iter,
        validation_steps=val_steps-1,
        verbose=1,
        callbacks=[tensorboard_callback,
                   model_checkpoint_callback,
                   lr_schedule_callback,
                   lr_warmup_callback,
                   lr_logger_callback]
    )

    print("\n------- Test -------")
    # Fit the model to the input data
    model.evaluate(
        test_iter,
        steps=test_steps-1,
        verbose=1
    )
    print("\n")

if __name__ == "__main__":
    main()
