"""Main module for training models."""
import argparse
import os
import tensorflow as tf
import pdb
from tensorflow import keras

from .training_utils import Preprocessing
from .models import TimeSeriesTransformer
from .misc import WarmUpScheduler, LearningRateLogger

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
    parser.add_argument('--num_epochs', type=int, default=100,
                        metavar="100", help='number of epochs')
    parser.add_argument('--init_learning_rate', type=float, default=1e-4,
                        metavar="1e-4", help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        metavar="0.2", help='decay rate of learning rate')
    return parser.parse_args()


def main():
    """Parse command line arguments and train model."""
    args = parse_arguments()

    # Hyperparameters
    batch_size = args.batch_size  # Default: 16
    num_epochs = args.num_epochs  # Default: 10
    init_learning_rate = args.init_learning_rate  # Default: 1e-3
    lr_decay_rate = args.lr_decay_rate  # Default: 0.2

    # Experiment logs folder
    experiment_log = os.path.join(
        log_dir, 
        "ex_bS_" + str(batch_size) + 
        "_nE_" + str(num_epochs) +
        "_iLr_" + str(init_learning_rate) +
        "_lrD_" + str(lr_decay_rate)
    )

    file_writer = tf.summary.create_file_writer(experiment_log + "/metrics")
    file_writer.set_as_default()

    # Load data (epochs don't matter because we iterate over the dataset
    # indefinitely)
    transformation = Preprocessing(
        dataset='physionet2012', epochs=num_epochs, batch_size=batch_size)
    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = \
        transformation._prepare_dataset_for_training()

    # Initialize the model
    model = TimeSeriesTransformer(
        proj_dim=128, num_head=4, enc_dim=128, pos_ff_dim=128, pred_ff_dim=32
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

    # Callback for loggin the learning rate for inspection
    lr_logger_callback = LearningRateLogger()

    # Callback for reducing the learning rate when loss get stuck in a plateau
    lr_schedule_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        mode='min',
        factor=lr_decay_rate,
        patience=2,
        min_lr=1e-7
    )

    # Callback for early stopping when val_loss does not improve anymore
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=8,
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
        profile_batch='180, 185'
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
        callbacks=[model_checkpoint_callback,
                   lr_schedule_callback,
                   tensorboard_callback,
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
