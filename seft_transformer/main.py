"""Main module for training models."""
import argparse
import os
import tensorflow as tf
from tensorflow import keras

from .training_utils import Preprocessing

from .models import TimeSeriesTransformer

tf.executing_eagerly()
checkpoint_filepath = './checkpoints/cp.ckpt'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(0)
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Embedding Translational Equivariance to SeFT')
    parser.add_argument('--batch_size', type=int, default=16,
                        metavar="16", help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        metavar="10", help='number of epochs')
    parser.add_argument('--init_learning_rate', type=float,
                        default=0.001, metavar="0.001", help='initial learning rate')
    parser.add_argument('--lr_decay_patience', type=int, default=2, metavar="2",
                        help='number of unimproving epochs after which learning rate decays')
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
    lr_decay_patience = args.lr_decay_patience  # Default: 2
    lr_decay_rate = args.lr_decay_rate  # Default: 0.2

    # Load data (epochs don't matter because we iterate over the dataset
    # indefinitely)
    transformation = Preprocessing(
        dataset='physionet2012', epochs=num_epochs, batch_size=batch_size)
    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = \
        transformation._prepare_dataset_for_training()

    # Initialize the model
    model = TimeSeriesTransformer(
        proj_dim=128, num_head=4, enc_dim=128, pos_ff_dim=128, pred_ff_dim=32)

    # Optimizer function
    opt = keras.optimizers.Adam(
        learning_rate=init_learning_rate
    )

    # Loss function
    loss_fn = keras.losses.BinaryCrossentropy(
        from_logits=False,
        name="Loss"
    )

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(curve="PR", name="auprc"),
                 keras.metrics.AUC(curve="ROC", name="auroc")]
    )

    # Callback for reducing the learning rate when the model get stuck in a plateau
    lr_schedule_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_auprc',
        mode='max',
        factor=lr_decay_rate,
        patience=lr_decay_patience,
        min_lr=0.0001
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_auprc',
        mode='max',
        patience=5,
        restore_best_weights=True
    )

    # Callback for saving the weights of the best model
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_auprc',
        mode='max',
        save_best_only=True
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
                   early_stopping_callback]
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
