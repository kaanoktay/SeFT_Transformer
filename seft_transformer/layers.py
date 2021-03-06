"""Implementation of Layers used by Keras models."""
from tensorflow.keras import layers
import tensorflow as tf
from einops import rearrange, reduce
import sys

from .blocks import (
    AxialMultiHeadAttentionBlock,
    PosEncodingBlock,
    InpEncodingBlock,
    ModEncodingBlock,
    PosFeedforwardBlock,
)


class InputEmbedding(layers.Layer):
    def __init__(self, enc_dim=128, equivar=False, 
                 no_time=False):
        super(InputEmbedding, self).__init__()

        self.pos_encoding = PosEncodingBlock(
            enc_dim=enc_dim, equivar=equivar
        )
        self.inp_encoding = InpEncodingBlock(
            enc_dim=enc_dim
        )
        self.mod_encoding = ModEncodingBlock(
            enc_dim=enc_dim
        )
        self.equivar = equivar
        self.no_time = no_time

    def call(self, inp, time, mask):
        """
        Input shapes:
          inp:  (b, t, m, i)
          time: (b, t)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d), (b, t, t, d) if equivar
                  (b, t, m, d), None         else
        """
        pos_enc = self.pos_encoding(time)
        inp_enc = self.inp_encoding(inp)
        mod_enc = self.mod_encoding(inp)
        if self.no_time:
            return inp_enc + mod_enc, None
        else:
            if self.equivar:
                return inp_enc + mod_enc, pos_enc
            else:
                return inp_enc + mod_enc + pos_enc, None


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
    def __init__(self, proj_dim=128, enc_dim=128, 
                 num_head=4, ff_dim=128, drop_rate=0.2, 
                 norm_type="reZero", causal_mask=False,
                 equivar=False):
        super(AxialAttentionEncoderLayer, self).__init__()

        self.axAttention = AxialMultiHeadAttentionBlock(
            proj_dim=proj_dim, enc_dim=enc_dim,
            num_head=num_head, drop_rate=drop_rate,
            causal_mask=causal_mask, equivar=equivar,
        )
        self.posFeedforward = PosFeedforwardBlock(
            enc_dim=enc_dim, ff_dim=ff_dim, drop_rate=drop_rate
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

    def call(self, inp, pos, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          pos:  (b, t, t, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        # Calculate attention and apply residual + normalization
        attn = self.axAttention(inp, pos, mask)  # (b, t, m, d)
        attn = self.norm1(self.residual1(inp, attn))  # (b, t, m, d)
        # Calculate positionwise feedforward and apply residual + normalization
        attn_ffn = self.posFeedforward(attn)  # (b, t, m, d)
        attn_ffn = self.norm2(self.residual2(attn, attn_ffn))  # (b, t, m, d)
        return attn_ffn


class ClassPredictionLayer(layers.Layer):
    """Layer for predicting a class output from a series."""

    def __init__(self, ff_dim=32, drop_rate=0.2, causal_mask=False):
        super(ClassPredictionLayer, self).__init__()
        self.ff_dim = ff_dim
        self.dropout = layers.Dropout(drop_rate)
        self.causal_mask = causal_mask

    def build(self, input_shape):
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
        out = self.densePred1(self.dropout(inp)) # (b, t, m, f)
        # Calculate sum over the modalities
        if self.causal_mask:
            out = reduce(out, 'b t m f -> b t f', 'sum')
        # Calculate sum over the timesteps and modalities
        else:
            out = reduce(out, 'b t m f -> b f', 'sum')
        # Calculate number of measured samples and normalize the sum
        mask = tf.cast(mask, dtype='float32')
        if self.causal_mask:
            norm = reduce(mask, 'b t m 1-> b t 1', 'sum')
            out = out / norm  # (b, t, f)
            out = tf.where(tf.math.is_nan(out), 0.0, out)
        else:
            norm = reduce(mask, 'b t m 1-> b 1', 'sum')
            out = out / norm  # (b, f)
        # Predict the class
        # Project to an intermediate dimension
        pred = self.densePred2(out) # (b, 1)
        return pred  # if causal_mask (b, t, 1) else (b, 1)