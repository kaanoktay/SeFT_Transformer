"""Implementation of Layers used by Keras models."""
from tensorflow.keras import layers
import tensorflow as tf
from einops import rearrange, reduce

from .blocks import (
    AxialMultiHeadAttentionBlock,
    PosEncodingBlock,
    InpEncodingBlock,
    PosFeedforwardBlock
)


class InputEmbedding(layers.Layer):
    def __init__(self, enc_dim=128):
        super(InputEmbedding, self).__init__()
        self.pos_encoding = PosEncodingBlock(enc_dim=enc_dim)
        self.inp_encoding = InpEncodingBlock(enc_dim=enc_dim)

    def call(self, inp, time, mask):
        """
        Input shapes:
          inp:  (b, t, m, i)
          time: (b, t)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        pos_enc = self.pos_encoding(time, mask)  # (b, t, m, d)
        inp_enc = self.inp_encoding(inp)  # (b, t, m, d)
        tot_enc = inp_enc + pos_enc  # (b, t, m, d)
        return tot_enc


class AxialAttentionEncoderLayer(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128, num_head=4, ff_dim=128):
        super(AxialAttentionEncoderLayer, self).__init__()
        self.axAttention = AxialMultiHeadAttentionBlock(
            proj_dim=proj_dim,
            enc_dim=enc_dim,
            num_head=num_head
        )
        self.posFeedforward = PosFeedforwardBlock(
            enc_dim=enc_dim,
            ff_dim=ff_dim
        )
        self.layerNorm1 = layers.LayerNormalization()
        self.layerNorm2 = layers.LayerNormalization()

    def call(self, inp, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        # Calculate attention and apply layer norm
        attn = self.axAttention(inp, mask)  # (b, t, m, d)
        attn_out = self.layerNorm1(inp + attn)  # (b, t, m, d)
        # Calculate positionwise feedforward and apply layer norm
        ffn = self.posFeedforward(attn_out)  # (b, t, m, d)
        ffn_out = self.layerNorm2(attn_out + ffn)  # (b, t, m, d)
        return ffn_out


class ClassPredictionLayer(layers.Layer):
    """Layer for predicting a class output from a series."""

    def __init__(self, ff_dim=32):
        super(ClassPredictionLayer, self).__init__()
        self.ff_dim = ff_dim

    def build(self, input_shape):
        # Dense layer to aggregate different modalities
        self.denseMod = layers.Dense(1)
        # Dense layers to predict classes
        self.densePred1 = layers.Dense(self.ff_dim, activation='relu')
        self.densePred2 = layers.Dense(1, activation='sigmoid')

    def call(self, inp, mask, length):
        """Predict class.

        Input shapes:
          inp:  (b, t, m, d)
          mask: (b, t, m)
          length: (b)
        Output shapes:
          return: (b, 1)
        """
        # TODO(Max): I'm not sure if this two stage aggregation is ideal.
        # Mask the padded values
        if mask is not None:
            mask = rearrange(mask, 'b t m -> b t m 1')
            inp = tf.where(mask, inp, 0)
        #out = tf.boolean_mask(inp, mask)  # (b, t_r, d, 1)
        # Calculate mean over the timesteps
        out = reduce(inp, 'b t m d -> b d m', 'sum')
        length = tf.cast(length, dtype="float32")
        out = out / rearrange(length, 'b -> b 1 1')
        # Aggregate the modalities
        out = self.denseMod(out)  # (b, d, 1)
        out = rearrange(out, 'b d 1 -> b d')  # (b, d)
        # Map to the class
        pred = self.densePred2(self.densePred1(out))  # (b, 1)
        return pred
