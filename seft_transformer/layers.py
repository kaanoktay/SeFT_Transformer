"""Implementation of Layers used by Keras models."""
from tensorflow.keras import layers
import tensorflow as tf
from einops import rearrange, reduce
import sys

from .blocks import (
    MultiHeadAttentionBlock,
    PosEncodingBlock,
    InpEncodingBlock,
    ModEncodingBlock,
    PosFeedforwardBlock,
)


class InputEmbedding(layers.Layer):
    def __init__(self, enc_dim=128, equivar=False, no_time=False,
                 train_time_enc=False):
        super().__init__()

        self.pos_encoding = PosEncodingBlock(
            enc_dim=enc_dim, train_time_enc=train_time_enc
        )
        self.inp_encoding = InpEncodingBlock(
            enc_dim=enc_dim
        )
        self.mod_encoding = ModEncodingBlock(
            enc_dim=enc_dim
        )
        self.equivar = equivar
        self.no_time = no_time

    def call(self, inp, time, mod):
        """
        Input shapes:
          inp:  (n,)
          time: (n,)
          mod:  (n,)
        Output shapes:
          return: (n, d)
        """
        # Add an extra dimension for encodings
        inp = tf.expand_dims(inp, axis=-1)
        time = tf.expand_dims(time, axis=-1)
        mod = tf.expand_dims(mod, axis=-1)
    
        # Calculate encodings
        inp_enc = self.inp_encoding(inp)  # (n, d)
        mod_enc = self.mod_encoding(mod)  # (n, d)
        if self.no_time or self.equivar:
            return inp_enc + mod_enc  # (n, d)
        else:
            pos_enc = self.pos_encoding(time)  # (n, d)
            return inp_enc + mod_enc + pos_enc  # (n, d)


class ReZero(layers.Layer):
    def __init__(self):
        super().__init__()

        self.re_weight = tf.Variable(
            initial_value=0.0,
            trainable=True
        )

    def call(self, x1, x2):
        return x1 + self.re_weight * x2


class Attention(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128, 
                 num_head=4, ff_dim=128, drop_rate=0.2, 
                 norm_type="reZero", causal_mask=False,
                 equivar=False, ax_attn=False):
        super().__init__()

        self.axAttention = MultiHeadAttentionBlock(
            proj_dim=proj_dim, enc_dim=enc_dim,
            num_head=num_head, drop_rate=drop_rate,
            causal_mask=causal_mask, equivar=equivar,
            ax_attn=ax_attn
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

    def call(self, inp, pos, mod, batch_seg):
        """
        Input shapes:
          inp:  (n, d)
          pos:  (n, 1)
          mod:  (n, 1)
        Output shapes:
          return: (n, d)
        """
        # Calculate attention
        attn = self.axAttention(inp, pos, mod, batch_seg)   # (n, d)

        # Apply residual + normalization
        attn = self.norm1(self.residual1(inp, attn))  # (n, d)

        # Calculate positionwise feedforward
        attn_ffn = self.posFeedforward(attn)  # (n, d)
        
        # Apply residual + normalization
        attn_ffn = self.norm2(self.residual2(attn, attn_ffn))  # (n, d)

        return attn_ffn


class MultiLayerAttention(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128,
                 num_head=4, ff_dim=128, drop_rate=0.2, 
                 norm_type="reZero", causal_mask=False,
                 equivar=False, num_layers=1, ax_attn=False):
        super().__init__()

        self.layers = []

        for i in range(num_layers):
            self.layers.append(
                Attention(
                    proj_dim=proj_dim, enc_dim=enc_dim,
                    num_head=num_head, ff_dim=ff_dim, 
                    drop_rate=drop_rate, norm_type=norm_type, 
                    causal_mask=causal_mask, equivar=equivar,
                    ax_attn=ax_attn
                )
            )

    def call(self, inp, pos, mod, batch_seg):
        """
        Input shapes:
          inp:  (n, d)
          pos:  (n, 1)
          mod:  (n, 1)
        Output shapes:
          return: (n, d)
        """
        attn = inp
        
        for layer in self.layers:
            attn = layer(attn, pos, mod, batch_seg)

        return attn


class SumWhileLoop(layers.Layer):
    """Layer for summing over all dimensions."""

    def __init__(self):
        super().__init__()

    def call(self, inp, batch_seg):
        """
        Input shapes:
          inp:  (n, d)
        Output shapes:
          return: (b, d)
        """
        # Get variables for batch segment ids
        batch_seg_ids, _ = tf.unique(batch_seg)
        n_batch_seg = tf.shape(batch_seg_ids)[0]

        batch_out_arr = tf.TensorArray(
            inp.dtype, size=n_batch_seg, infer_shape=False)

        def loop_cond(i, out):
            return i < n_batch_seg

        def loop_body(i, out):
            curr_seg = batch_seg_ids[i]
            curr_ind = tf.cast(tf.where(tf.equal(batch_seg, curr_seg)),
                               dtype=tf.int32)
            curr_inp = tf.gather_nd(inp, curr_ind)
            
            inp_sum = tf.math.reduce_mean(
                curr_inp, axis=0, keepdims=True)

            out = out.write(i, inp_sum)

            return i+1, out

        i_end, batch_out_arr = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=(tf.constant(0), batch_out_arr)
        )

        # Concat batch outputs
        out = batch_out_arr.concat()
        out.set_shape([None, inp.shape[1]])

        return out


class ClassPrediction(layers.Layer):
    """Layer for predicting a class output from a series."""

    def __init__(self, ff_dim=32, drop_rate=0.2, causal_mask=False):
        super().__init__()
        self.dropout = layers.Dropout(drop_rate)
        self.causal_mask = causal_mask

        # Sum while loop to get classes
        self.sum_while_loop = SumWhileLoop()

        # Dense layers to predict classes
        self.densePred1 = layers.Dense(ff_dim, activation='relu')
        self.densePred2 = layers.Dense(1, activation='sigmoid')

    def call(self, inp, batch_seg):
        """Predict class.
        
        Input shapes:
          inp:  (n, d)
        Output shapes:
          return: (b, 1)
        """

        # Calculate sum within each batch
        out = self.sum_while_loop(inp, batch_seg) # (b, d)

        # Project to intermediate dimension
        out = self.densePred1(self.dropout(out)) # (b, f)

        # Predict class 
        pred = self.densePred2(out) # (b, 1)

        return pred  # (b, 1)