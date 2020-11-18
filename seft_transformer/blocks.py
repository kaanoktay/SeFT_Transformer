import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange
import numpy as np


class PosEncodingBlock(layers.Layer):
    """Layer for the computation of positional encodings."""

    def __init__(self, enc_dim=128):
        super(PosEncodingBlock, self).__init__()
        self.f = tf.math.exp(
            tf.range(start=0, limit=enc_dim, delta=2, dtype="float32")
            * -(tf.math.log(10000.0) / enc_dim)
        )

    def call(self, time, mask):
        """
        Input shapes:
          time: (b, t)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)  -> if mask is None: (b, t, 1, d)
        """
        # Calculate sine and cosine components
        angles = tf.einsum('bt,f->btf', time, self.f)  # (b, t, d/2)
        sin_enc = tf.math.sin(angles)  # sin encodings (b, t, d/2)
        cos_enc = tf.math.cos(angles)  # cos encodings (b, t, d/2)
        # Construct positional encodings
        pos_enc = rearrange([sin_enc, cos_enc],  'z b t k -> b t (k z)')
        pos_enc = rearrange(pos_enc, 'b t d -> b t 1 d')
        # Replace the invalid positions with zeros
        if mask is not None:
            mask = rearrange(mask, 'b t m -> b t m 1')
            pos_enc = tf.where(mask, pos_enc, 0)  # (b, t, m, d)
        return pos_enc


class InpEncodingBlock(layers.Layer):
    """Layer for the computation of value embeddings."""

    def __init__(self, enc_dim=128):
        super(InpEncodingBlock, self).__init__()
        self.enc_dim = enc_dim

    def build(self, input_shape):
        # Input shapes
        input_dim = input_shape[-1]
        num_mod = input_shape[-2]
        # Weight and bias initializers
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        # Weight matrix and bias: input data encoding
        self.W = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, input_dim, self.enc_dim), dtype="float32"),
            trainable=True
        )
        self.B = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.enc_dim), dtype="float32"),
            trainable=True
        )

    def call(self, inp):
        """
        Input shapes:
          inp: (b, t, m, i)
        Output shapes:
          return: (b, t, m, d)
        """
        # Calculate input data encodings
        inp_enc = tf.einsum('...i,...id->...d', inp,
                            self.W) + self.B  # (b, t, m, d)
        return inp_enc


class SeqAttentionBlock(layers.Layer):
    def __init__(self, proj_dim=128, num_head=4):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_head = num_head
        self.embed_dim = proj_dim // num_head

        self.query_dense = layers.Dense(proj_dim)
        self.key_dense = layers.Dense(proj_dim)
        self.value_dense = layers.Dense(proj_dim)

    def call(self, inp, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, p)
        """
        # Project query, key and value
        q = self.query_dense(inp)  # (b, t, m, p)
        k = self.key_dense(inp)    # (b, t, m, p)
        v = self.value_dense(inp)  # (b, t, m, p)
        # Separate heads
        q = rearrange(q, 'b t m (h e) -> b m h t e',
                      h=self.num_head)  # (b, m, h, t, e)
        k = rearrange(k, 'b t m (h e) -> b m h t e',
                      h=self.num_head)  # (b, m, h, t, e)
        v = rearrange(v, 'b t m (h e) -> b m h t e',
                      h=self.num_head)  # (b, m, h, t, e)
        # Calculate attention
        score = tf.einsum('...ij,...kj->...ik', q, k) / \
            np.sqrt(self.embed_dim)  # (b, m, h, t, t)
        if mask is not None:
            mask = rearrange(mask, 'b t m -> b m 1 1 t')
            score = tf.where(mask, score, -np.inf)
        weight = tf.nn.softmax(score)  # (b, m, h, t, t)
        if mask is not None:
            weight = tf.where(mask, weight, 0)
        # Check if there is any NaN in weights
        tf.debugging.check_numerics(
            tensor=weight,
            message="Check the values after softmax"
        )
        output = tf.einsum('...ij,...jk->...ik', weight, v)  # (b, m, h, t, e)
        # Concatenate heads
        concat_output = rearrange(
            output, 'b m h t e -> b t m (h e)')  # (b, t, m, p)
        return concat_output


class ModAttentionBlock(layers.Layer):
    def __init__(self, proj_dim=128, num_head=4):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_head = num_head
        self.embed_dim = proj_dim // num_head

        self.query_dense = layers.Dense(proj_dim)
        self.key_dense = layers.Dense(proj_dim)
        self.value_dense = layers.Dense(proj_dim)

    def call(self, inp, mask=None):
        """
        Input shapes:
          inp:  (b, t, m, p)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, p)
        """
        # Create query, key and value
        q = self.query_dense(inp)  # (b, t, m, p)
        k = self.key_dense(inp)   # (b, t, m, p)
        v = self.value_dense(inp)  # (b, t, m, p)
        # Separate heads
        q = rearrange(q, 'b t m (h e) -> b t h m e',
                      h=self.num_head)  # (b, t, h, m, e)
        k = rearrange(k, 'b t m (h e) -> b t h m e',
                      h=self.num_head)  # (b, t, h, m, e)
        v = rearrange(v, 'b t m (h e) -> b t h m e',
                      h=self.num_head)  # (b, t, h, m, e)
        # Calculate attention
        score = tf.einsum('...ij,...kj->...ik', q, k) / \
            np.sqrt(self.embed_dim)  # (b, t, h, m, m)
        if mask is not None:
            mask = rearrange(mask, 'b t m -> b t 1 1 m')
            score = tf.where(mask, score, -np.inf)
        weight = tf.nn.softmax(score)  # (b, t, h, m, m)
        if mask is not None:
            weight = tf.where(mask, weight, 0)
        # Check if there is any NaN in weights
        tf.debugging.check_numerics(
            tensor=weight,
            message="Check the values after softmax"
        )
        output = tf.einsum('...ij,...jk->...ik', weight, v)  # (b, t, h, m, e)
        # Concatenate heads
        concat_output = rearrange(
            output, 'b t h m e -> b t m (h e)')  # (b, t, m, p)
        return concat_output


class AxialMultiHeadAttentionBlock(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128, num_head=4):
        super().__init__()
        self.seqAttention = SeqAttentionBlock(
            proj_dim=proj_dim, num_head=num_head)
        self.modAttention = ModAttentionBlock(
            proj_dim=proj_dim, num_head=num_head)
        self.enc_dim = enc_dim

    def build(self, input_shape):
        # Input shapes
        input_dim = input_shape[-1]
        num_mod = input_shape[-2]
        # Weight and bias initializers
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        # Weight matrix and bias: map proj_dim to data_dim (=enc_dim)
        self.W = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, input_dim, self.enc_dim), dtype="float32"),
            trainable=True
        )
        self.B = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.enc_dim), dtype="float32"),
            trainable=True
        )

    def call(self, inp, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        # Attention over timestamps
        out = self.seqAttention(inp=inp, mask=mask)  # (b, t, m, p)
        # Attention over modalities
        out = self.modAttention(inp=out, mask=mask)   # (b, t, m, p)
        # Linear projection to encoding dimension
        out = tf.einsum('...d,...dp->...p', out, self.W) + \
            self.B  # (b, t, m, d)
        return out


class PosFeedforwardBlock(layers.Layer):
    def __init__(self, enc_dim=128, ff_dim=128):
        super().__init__()
        self.enc_dim = enc_dim
        self.ff_dim = ff_dim
        self.relu = layers.ReLU()

    def build(self, input_shape):
        # Input shapes
        input_dim = input_shape[-1]
        num_mod = input_shape[-2]
        # Weight and bias initializers
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        # Weight matrix and bias: 1st feedforward layer
        self.W1 = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, input_dim, self.ff_dim), dtype="float32"),
            trainable=True
        )
        self.B1 = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.ff_dim), dtype="float32"),
            trainable=True
        )
        # Weight matrix and bias: 2nd feedforward layer
        self.W2 = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, self.ff_dim, self.enc_dim), dtype="float32"),
            trainable=True
        )
        self.B2 = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.enc_dim), dtype="float32"),
            trainable=True
        )

    def call(self, inp):
        """
        Input shapes:
          inp:  (b, t, m, p)
        Output shapes:
          return: (b, t, m, d)
        """
        # Positionwise feedforward network
        out = tf.einsum('...p,...pf->...f', inp, self.W1) + \
            self.B1  # (b, t, m, f)
        out = self.relu(out)
        out = tf.einsum('...f,...fd->...d', out, self.W2) + \
            self.B2  # (b, t, m, d)
        return out
