"""Utility functions for training and evaluation."""
import math
import argparse
from collections.abc import Sequence

import tensorflow as tf
import tensorflow_datasets as tfds
import medical_ts_datasets
from normalization import Normalizer

get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types
make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator


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
    parser.add_argument('--lr_warmup_steps', type=float, default=2e3,
                        metavar="2e3", help='learning rate warmup steps')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        metavar="0.2", help='dropout rate')
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
    parser.add_argument('--ax_attn', default=False,
                        action='store_true')
    parser.add_argument('--train_time_enc', default=False,
                        action='store_true')
    return parser.parse_args()


class PaddedToSegments(tf.keras.layers.Layer):
    """Convert a padded tensor with mask to a stacked tensor with segments."""

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

    def call(self, inputs, mask):
        valid_observations = tf.where(mask)
        collected_values = tf.gather_nd(inputs, valid_observations)
        return collected_values, valid_observations[:, 0]


def positive_instances(*args):
    if len(args) == 2:
        data, label = args
    if len(args) == 3:
        data, label, sample_weights = args

    return tf.math.equal(tf.reduce_max(label), 1)


def negative_instances(*args):
    if len(args) == 2:
        data, label = args
    if len(args) == 3:
        data, label, sample_weights = args
    return tf.math.equal(tf.reduce_max(label), 0)


def get_padding_values(input_dataset_types, label_padding=-100):
    """Get a tensor of padding values fitting input_dataset_types.

    Here we pad everything with 0. and the labels with `label_padding`. This
    allows us to be able to recognize them later during the evaluation, even
    when the values have already been padded into batches.

    Args:
        tensor_shapes: Nested structure of tensor shapes.

    Returns:
        Nested structure of padding values where all are 0 except the one
        corresponding to tensor_shapes[1], which is padded according to the
        `label_padding` value.

    """
    def map_to_zero(dtypes):
        if isinstance(dtypes, Sequence):
            return tuple((map_to_zero(d) for d in dtypes))
        return tf.cast(0., dtypes)

    def map_to_label_padding(dtypes):
        if isinstance(dtypes, Sequence):
            return tuple((map_to_zero(d) for d in dtypes))
        return tf.cast(label_padding, dtypes)

    if len(input_dataset_types) == 2:
        data_type, label_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type)
        )

    if len(input_dataset_types) == 3:
        data_type, label_type, sample_weight_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type),
            map_to_zero(sample_weight_type)
        )


def build_training_iterator(dataset_name, epochs, batch_size, prepro_fn,
                            balance=False, class_balance=None):
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.TRAIN,
        as_supervised=True,
        with_info=True
    )
    n_samples = dataset_info.splits['train'].num_examples
    steps_per_epoch = int(math.floor(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(
            prepro_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if balance:
        majority_class = max(
            range(len(class_balance)), key=lambda i: class_balance[i])
        minority_class = min(
            range(len(class_balance)), key=lambda i: class_balance[i])

        n_majority = class_balance[majority_class] * n_samples
        n_minority = class_balance[minority_class] * n_samples
        # Generate two separate datasets using filter
        pos_data = (dataset
                    .filter(positive_instances)
                    .shuffle(
                        int(class_balance[1] * n_samples),
                        reshuffle_each_iteration=True)
                    .repeat()
                    )
        neg_data = (dataset
                    .filter(negative_instances)
                    .shuffle(
                        int(class_balance[0] * n_samples),
                        reshuffle_each_iteration=True)
                    .repeat()
                    )
        # And sample from them
        dataset = tf.data.experimental.sample_from_datasets(
            [pos_data, neg_data], weights=[0.5, 0.5])
        # One epoch should at least contain all negative examples or max
        # each instance of the minority class 3 times
        steps_per_epoch = min(
            math.ceil(2 * n_majority / batch_size),
            math.ceil(3 * 2 * n_minority / batch_size)
        )
    else:
        # Shuffle repeat and batch
        dataset = dataset.shuffle(n_samples, reshuffle_each_iteration=True)
        dataset = dataset.repeat(epochs)

    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=True
    )
    return batched_dataset.prefetch(tf.data.experimental.AUTOTUNE), steps_per_epoch


def build_validation_iterator(dataset_name, batch_size, prepro_fn):
    """Build a validation iterator for a tensorflow datasets dataset.

    Args:
        dataset_name: Name of the tensoflow datasets dataset. To be used with
            tfds.load().
        epochs: Number of epochs to run
        batch_size: Batch size
        prepro_fn: Optional preprocessing function that should be applied to
            prior to batching.

    Returns:
        A tensorflow dataset which iterates through the validation dataset
           epoch times.

    """
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.VALIDATION,
        as_supervised=True,
        with_info=True
    )
    n_samples = dataset_info.splits['validation'].num_examples
    steps_per_epoch = int(math.ceil(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(
            prepro_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=False
    )
    return batched_dataset, steps_per_epoch


def build_test_iterator(dataset_name, batch_size, prepro_fn):
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.TEST,
        as_supervised=True,
        with_info=True
    )
    n_samples = dataset_info.splits['test'].num_examples
    steps = int(math.floor(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(
            prepro_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=False
    )
    return batched_dataset, steps


class Preprocessing(object):
    """Preprocessing object.

    Groups all routines used for preprocessing data.
    """

    def __init__(self, dataset, epochs, batch_size, balance_dataset=True):
        self.dataset = dataset
        self.normalizer = Normalizer(dataset)
        self.balance_dataset = balance_dataset
        self.n_epochs = epochs
        self.batch_size = batch_size

    def normalize_and_preprocess(self):
        """Normalize input data and apply model specific preprocessing fn."""
        
        def flatten_to_set(ts, labels):
            # Convert normalized ts to flattened set representation
            demo, X, Y, measurements, lengths = ts
            X = tf.expand_dims(X, -1)
            measurement_positions = tf.cast(tf.where(measurements), tf.int32)
            X_indices = measurement_positions[:, 0]
            Y_indices = measurement_positions[:, 1]

            gathered_X = tf.gather(X, X_indices)
            gathered_Y = tf.gather_nd(Y, measurement_positions)
            gathered_Y = tf.expand_dims(gathered_Y, axis=-1)

            length = tf.shape(X_indices)[0]
            return (demo, gathered_X, gathered_Y, Y_indices, length), labels

        def combined_fn(ts, labels):
            normalized_ts, labels = self.normalizer.get_normalization_fn()(ts, labels)
            return flatten_to_set(normalized_ts, labels)

        return combined_fn

    def _prepare_dataset_for_training(self):
        if self.balance_dataset:
            class_balance = [
                self.normalizer._class_balance[str(i)] for i in range(2)]
        else:
            class_balance = None
        train_iterator, train_steps = build_training_iterator(
            self.dataset,
            self.n_epochs,
            self.batch_size,
            self.normalize_and_preprocess(),
            balance=self.balance_dataset,
            class_balance=class_balance
        )
        # Repeat epochs + 1 times as we run an additional validation step at
        # the end of training after recovering the model.
        val_iterator, val_steps = build_validation_iterator(
            self.dataset,
            self.batch_size,
            self.normalize_and_preprocess()
        )

        test_iterator, test_steps = build_test_iterator(
            self.dataset,
            self.batch_size,
            self.normalize_and_preprocess()
        )
        return train_iterator, train_steps, val_iterator, val_steps, test_iterator, test_steps

"""
class Physionet2019UtilityScore(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)

    def compute_prediction_utility(labels, predictions, dt_early=-12,
                                   dt_optimal=-6, dt_late=3.0, max_u_tp=1,
                                   min_u_fn=-2, u_fp=-0.05, u_tn=0,
                                   check_errors=True):
        Compute utility score of physionet 2019 challenge.

        # Does the patient eventually have sepsis?
        if tf.math.reduce_any(labels==1):
            is_septic = True
            t_sepsis = np.argmax(labels) - dt_optimal
        else:
            is_septic = False
            t_sepsis = float('inf')

        n = len(labels)

        # Define slopes and intercept points for utility functions of the form
        # u = m * t + b.
        m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
        b_1 = -m_1 * dt_early
        m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
        b_2 = -m_2 * dt_late
        m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
        b_3 = -m_3 * dt_optimal

        # Compare predicted and true conditions.
        u = np.zeros(n)
        for t in range(n):
            if t <= t_sepsis + dt_late:
                # TP
                if is_septic and predictions[t]:
                    if t <= t_sepsis + dt_optimal:
                        u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                    elif t <= t_sepsis + dt_late:
                        u[t] = m_2 * (t - t_sepsis) + b_2
                # FP
                elif not is_septic and predictions[t]:
                    u[t] = u_fp
                # FN
                elif is_septic and not predictions[t]:
                    if t <= t_sepsis + dt_optimal:
                        u[t] = 0
                    elif t <= t_sepsis + dt_late:
                        u[t] = m_3 * (t - t_sepsis) + b_3
                # TN
                elif not is_septic and not predictions[t]:
                    u[t] = u_tn

        # Find total utility for patient.
        return np.sum(u)

    def update_state(self, y_true, y_pred, sample_weight=None):

        Compute physionet 2019 Sepsis eary detection utility.
        Args:
            y_true:
            y_score:
        Returns:
        
        dt_early = -12
        dt_optimal = -6
        dt_late = 3.0

        utilities = []
        best_utilities = []
        inaction_utilities = []

        for labels, observed_predictions in zip(y_true, y_score):
            observed_predictions = np.round(observed_predictions)
            num_rows = len(labels)
            best_predictions = np.zeros(num_rows)
            inaction_predictions = np.zeros(num_rows)

            if np.any(labels):
                t_sepsis = np.argmax(labels) - dt_optimal
                pred_begin = int(max(0, t_sepsis + dt_early))
                pred_end = int(min(t_sepsis + dt_late + 1, num_rows))
                best_predictions[pred_begin:pred_end] = 1

            utilities.append(
                compute_prediction_utility(labels, observed_predictions))
            best_utilities.append(
                compute_prediction_utility(labels, best_predictions))
            inaction_utilities.append(
                compute_prediction_utility(labels, inaction_predictions))

        unnormalized_observed_utility = sum(utilities)
        unnormalized_best_utility = sum(best_utilities)
        unnormalized_inaction_utility = sum(inaction_utilities)
        normalized_observed_utility = (
            (unnormalized_observed_utility - unnormalized_inaction_utility)
            / (unnormalized_best_utility - unnormalized_inaction_utility)
        )
        return normalized_observed_utility


        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
    
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0)
"""