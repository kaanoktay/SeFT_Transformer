"""Implementation of the model."""
from tensorflow import keras
import tensorflow as tf
from einops import rearrange
import sys

from .layers import (
    AxialAttentionEncoderLayer,
    ClassPredictionLayer,
    InputEmbedding,
    AxialAttentionEncoderLayer_v2,
    ClassPredictionLayer_v2,
    InputEmbedding_v2
)


class TimeSeriesTransformer(keras.Model):
    """Time Series Transformer model."""

    def __init__(self, proj_dim=128, num_head=4, enc_dim=128, 
                 pos_ff_dim=128, pred_ff_dim=32, drop_rate=0.2, 
                 norm_type='reZero', dataset='physionet2012',
                 equivar=False, num_layers=1, no_time=False,
                 uni_mod=False):
        super(TimeSeriesTransformer, self).__init__()
        if dataset=='physionet2019':
            self.causal_mask = True
        else:
            self.causal_mask = False
        self.input_embedding = InputEmbedding(
            enc_dim=enc_dim, equivar=equivar,
            no_time=no_time, uni_mod=uni_mod
        )
        self.transformer_encoder = AxialAttentionEncoderLayer(
            proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
            ff_dim=pos_ff_dim, drop_rate=drop_rate, norm_type=norm_type,
            causal_mask=self.causal_mask, equivar=equivar,
            uni_mod=uni_mod
        )
        self.class_prediction = ClassPredictionLayer(
            ff_dim=pred_ff_dim, drop_rate=drop_rate,
            causal_mask=self.causal_mask
        )
    
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
        time = inputs[1]  # (b, t)
        inp = inputs[2]  # (b, t, m)
        mask = inputs[3]  # (b, t, m)
        count = inputs[4]  # (b, 1)
        # Expand input dimensions if necessary
        if len(inp.shape) == 3:
            inp = rearrange(inp, 'b t m -> b t m 1')
        # Encode inputs
        inp_enc, pos_enc = self.input_embedding(
            inp, time, mask)
        # Calculate attention
        attn = self.transformer_encoder(
            inp_enc, pos_enc, mask)
        # Make prediction: if causal_mask (b, t, 1) else (b, 1)
        pred = self.class_prediction(attn, mask)
        if self.causal_mask:
            return pred, count
        else:
            return pred


class TimeSeriesTransformer_v2(keras.Model):
    """Time Series Transformer model v2."""

    def __init__(self, proj_dim=128, num_head=4, enc_dim=128, 
                 pos_ff_dim=128, pred_ff_dim=32, drop_rate=0.2, 
                 norm_type='reZero', equivar=False, no_time=False):
        super(TimeSeriesTransformer_v2, self).__init__()

        self.input_embedding = InputEmbedding_v2(
                enc_dim=enc_dim, equivar=equivar, no_time=no_time
            )
        self.transformer_encoder = AxialAttentionEncoderLayer_v2(
            proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
            ff_dim=pos_ff_dim, drop_rate=drop_rate, norm_type=norm_type,
            equivar=equivar
        )
        self.class_prediction = ClassPredictionLayer_v2(
            ff_dim=pred_ff_dim, drop_rate=drop_rate
        )

    def call(self, inputs):
        """Apply model to data.

        Input shapes:
          data: a tuple of measurements, timestamps, masks etc.
        Output shapes:
          return: prediction
        """
        # Get inputs
        time = inputs[1]  # (b, t)
        inp = inputs[2]  # (b, t, m)
        mask = inputs[3]  # (b, t, m)
        count = inputs[4]  # (b, 1)
        # Transform mask
        mask = tf.cast(mask, tf.float32)
        mask = tf.math.reduce_sum(mask, axis=-1)
        mask = tf.cast(mask, tf.bool) # (b, t)
        # Encode inputs
        inp_enc, pos_enc = self.input_embedding(
            inp, time)  # (b, t, d), (b, t, t, d) or None
        # Calculate attention
        attn = self.transformer_encoder(
            inp_enc, pos_enc, mask)
        # Prediction
        pred = self.class_prediction(attn, mask)
        return pred  # (b, 1)