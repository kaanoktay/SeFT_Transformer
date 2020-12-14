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


class ReZero(layers.Layer):
    def __init__(self):
        super(ReZero, self).__init__()
        self.re_weight = tf.Variable(
            initial_value=0.0,
            trainable=True
        )

    def call(self, x1, x2):
        return x1 + self.re_weight * x2


class AxialAttentionEncoderLayer(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128, num_head=4,
                 ff_dim=128, drop_rate=0.1, norm_type="reZero"):
        super(AxialAttentionEncoderLayer, self).__init__()
        self.axAttention = AxialMultiHeadAttentionBlock(
            proj_dim=proj_dim, enc_dim=enc_dim,
            num_head=num_head, drop_rate=drop_rate
        )
        self.posFeedforward = PosFeedforwardBlock(
            enc_dim=enc_dim, ff_dim=ff_dim,
            drop_rate=drop_rate
        )
        
        if norm_type == 'layerNorm':
            def get_residual():
                def residual(x1, x2):
                    return x1 + x2
                return residual
            def get_norm():
                return layers.LayerNormalization()
        elif norm_type == 'reZero':
            def get_residual():
                return ReZero()
            def get_norm():
                return layers.Layer()
        else:
            raise ValueError('Invalid normalization: {}'.format(norm_type))

        self.norm1 = get_norm()
        self.norm2 = get_norm()
        self.residual1 = get_residual()
        self.residual2 = get_residual()

    def call(self, inp, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        # Calculate attention and apply residual + normalization
        attn = self.axAttention(inp, mask)  # (b, t, m, d)
        attn = self.norm1(self.residual1(inp, attn))  # (b, t, m, d)
        # Calculate positionwise feedforward and apply residual + normalization
        attn_ffn = self.posFeedforward(attn)  # (b, t, m, d)
        attn_ffn = self.norm2(self.residual2(attn, attn_ffn))  # (b, t, m, d)
        return attn_ffn


class ClassPredictionLayer(layers.Layer):
    """Layer for predicting a class output from a series."""

    def __init__(self, ff_dim=32, drop_rate=0.1):
        super(ClassPredictionLayer, self).__init__()
        self.ff_dim = ff_dim
        self.dropout = layers.Dropout(drop_rate)

    def build(self, input_shape):
        # Dense layer to aggregate different modalities
        self.denseMod = layers.Dense(1)
        # Dense layers to predict classes
        self.densePred1 = layers.Dense(self.ff_dim, activation='relu')
        self.densePred2 = layers.Dense(1, activation='sigmoid')

    def call(self, inp, mask):
        """Predict class.

        Input shapes:
          inp:  (b, t, m, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, 1)
        """
        # Mask the padded values with 0's
        if mask is not None:
            mask = rearrange(mask, 'b t m -> b t m 1')
            inp = tf.where(mask, inp, 0)
        # Calculate sum over the timesteps and modalities
        out = reduce(inp, 'b t m d -> b d', 'sum')
        # Normalize the sum
        mask = tf.cast(mask, dtype='float32')
        norm = reduce(mask, 'b t m 1-> b 1', 'sum')
        out = out / norm
        # Predict the class
        pred = self.densePred2(self.densePred1(self.dropout(out)))  # (b, 1)
        return pred
