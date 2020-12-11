"""Utility functions for training and evaluation."""
import math

from collections.abc import Sequence

import tensorflow as tf
import tensorflow_datasets as tfds
import medical_ts_datasets
from .normalization import Normalizer

get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types
make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator


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

        def combined_fn(ts, labels):
            normalized_ts, labels = self.normalizer.get_normalization_fn()(ts, labels)
            return normalized_ts, labels

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
