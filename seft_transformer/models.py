"""Implementation of the model."""
from tensorflow import keras
import tensorflow as tf
from einops import rearrange
import sys

from .training_utils import PaddedToSegments

from .layers import (
    AxialAttentionEncoderLayer,
    ClassPredictionLayer,
    InputEmbedding
)


class TimeSeriesTransformer(keras.Model):
    """Time Series Transformer model."""

    def __init__(self, proj_dim=128, num_head=4, enc_dim=128, 
                 pos_ff_dim=128, pred_ff_dim=32, drop_rate=0.2, 
                 norm_type='reZero', dataset='physionet2012',
                 equivar=False, num_layers=1, no_time=False):
        super(TimeSeriesTransformer, self).__init__()
        if dataset=='physionet2019':
            self.causal_mask = True
        else:
            self.causal_mask = False
        self.input_embedding = InputEmbedding(
            enc_dim=enc_dim, equivar=equivar, no_time=no_time
        )
        self.transformer_encoder = AxialAttentionEncoderLayer(
            proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
            ff_dim=pos_ff_dim, drop_rate=drop_rate, norm_type=norm_type,
            causal_mask=self.causal_mask, equivar=equivar
        )
        self.class_prediction = ClassPredictionLayer(
            ff_dim=pred_ff_dim, drop_rate=drop_rate,
            causal_mask=self.causal_mask
        )
        self.equivar = equivar
        self.to_segments = PaddedToSegments()
    
    def train_step(self, data):
        x, y = data
        sample_weight = None

        with tf.GradientTape() as tape:
            if self.causal_mask:
                # Forward pass
                y_pred, count = self(x, training=True)
                # Calculate the sample weight
                mask = tf.cast(
                    tf.sequence_mask(count),
                    dtype='float32'
                )
                sample_weight = mask / \
                    tf.reduce_sum(tf.cast(count, dtype='float32'))
                # Compute the loss value
                loss = self.compiled_loss(y, y_pred, sample_weight)
            else:
                # Forward pass
                y_pred = self(x, training=True)
                # Compute the loss value
                loss = self.compiled_loss(y, y_pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        if self.causal_mask:
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
        else:
            self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        sample_weight = None

        if self.causal_mask:
            # Forward pass
            y_pred, count = self(x, training=False)
            # Calculate the sample weight
            mask = tf.cast(
                tf.sequence_mask(count),
                dtype='float32'
            )
            sample_weight = mask / \
                tf.reduce_sum(tf.cast(count, dtype='float32'))
            # Compute the loss value
            self.compiled_loss(y, y_pred, sample_weight)
        else:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            self.compiled_loss(y, y_pred)

        # Update metrics
        if self.causal_mask:
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
        else:
            self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        """Apply model to data.

        Input shapes:
          data: a tuple of measurements, timestamps, masks etc.
        Output shapes:
          return: prediction
        """
        # Get inputs
        time = tf.squeeze(inputs[1])  # (b, t)
        inp = tf.squeeze(inputs[2])   # (b, t)
        mod = inputs[3]  # (b, t)
        count = inputs[4]  # (b)
        mask = tf.sequence_mask(count)  # (b, t)

        # Get input, time and modality sets
        time_set, segment_ids = self.to_segments(time, mask)
        inp_set, _ = self.to_segments(inp, mask)
        mod_set, _ = self.to_segments(mod, mask)
        
        # Get variables for segment ids
        # Batch
        batch_segment_ids, _ = tf.unique(segment_ids)
        n_batch_segments = batch_segment_ids.shape[0]
        # Time
        time_segment_ids, _ = tf.unique(time_set)
        n_time_segments = time_segment_ids.shape[0]
        # Modality
        mod_segment_ids, _ = tf.unique(mod_set)
        n_mod_segments = mod_segment_ids.shape[0]

        # Add an extra dimension for embedding
        inp_set = rearrange(inp_set, 'n -> n 1')
        time_set = rearrange(time_set, 'n -> n 1')
        mod_set = rearrange(mod_set, 'n -> n 1')

        # If no_time --> no time information used for inp_enc
        # If equivar --> time_enc will be calculated on fly
        # If none    --> time_enc in inp_enc
        inp_enc = self.input_embedding(
            inp_set, time_set, mod_set)
        # Calculate attention
        attn = self.transformer_encoder(
            inp_enc, pos_enc)
        # Make prediction: if causal_mask (b, t, 1) else (n, 1)
        pred = self.class_prediction(attn, mask)

        if self.causal_mask:
            return pred, count
        else:
            return pred
