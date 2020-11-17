"""Implementation of the model."""
from tensorflow import keras
from einops import rearrange

from .layers import (
    AxialAttentionEncoderLayer,
    ClassPredictionLayer,
    InputEmbedding
)


class TimeSeriesTransformer(keras.Model):
    """Time Series Transformer model."""

    def __init__(self, proj_dim=128, num_head=4, enc_dim=128, pos_ff_dim=128,
                 pred_ff_dim=32):
        super().__init__()
        self.input_embedding = InputEmbedding(
            enc_dim=enc_dim
        )
        self.transformer_encoder = AxialAttentionEncoderLayer(
            proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
            ff_dim=pos_ff_dim
        )
        self.class_prediction = ClassPredictionLayer(ff_dim=pred_ff_dim)

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
        # Expand input dimensions if necessary
        if len(inp.shape) == 3:
            inp = rearrange(inp, 'b t m -> b t m 1')
        # Encode inputs
        enc_inp = self.input_embedding(inp, time, mask)  # (b, t, m, d)
        # Calculate attention
        attn = self.transformer_encoder(enc_inp, mask)  # (b, t, m, d)
        # Make prediction
        pred = self.class_prediction(attn)  # (b, 1)
        return pred
