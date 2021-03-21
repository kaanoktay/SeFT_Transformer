import tensorflow as tf
from tensorflow.linalg import LinearOperatorLowerTriangular
from tensorflow.keras import layers
from einops import rearrange, repeat
import numpy as np
import sys


class PosEncodingBlock(layers.Layer):
    """Positional encodings layer."""

    def __init__(self, enc_dim=128, equivar=False):
        super().__init__()
        f = tf.math.exp(
            tf.range(start=0, limit=enc_dim, delta=2, dtype="float32")
            * -(tf.math.log(10000.0) / enc_dim)
        )
        self.f = tf.Variable(f, trainable=True)
        self.equivar = equivar

    def call(self, time):
        """
        Input shapes:
          time: (b, t)
        Output shapes:
          return: (b, t, t, d) if equivar
                  (b, t, 1, d) else
        """
        if self.equivar:
            rel_time = rearrange(time, 'b t -> b t 1') - \
                rearrange(time, 'b t -> b 1 t')  # relative time (b, t, t)
            # Calculate sine and cosine components
            angles = tf.einsum(
                'btl,f->btlf', rel_time, self.f)  # (b, t, t, d/2)
            sin_enc = tf.math.sin(angles)  # sin encodings (b, t, t, d/2)
            cos_enc = tf.math.cos(angles)  # cos encodings (b, t, t, d/2)
            # Construct positional encodings
            pos_enc = rearrange(
                [sin_enc, cos_enc],  'z b t l k -> b t l (k z)')
            return pos_enc  # (b, t, t, d)
        else:
            # Calculate sine and cosine components
            angles = tf.einsum(
                'bt,f->btf', time, self.f)  # (b, t, d/2)
            sin_enc = tf.math.sin(angles)  # sin encodings (b, t, d/2)
            cos_enc = tf.math.cos(angles)  # cos encodings (b, t, d/2)
            # Construct positional encodings
            pos_enc = rearrange(
                [sin_enc, cos_enc],  'z b t k -> b t 1 (k z)')
            return pos_enc  # (b, t, 1, d)


class InpEncodingBlock(layers.Layer):
    """Input encodings layer."""

    def __init__(self, enc_dim=128):
        super().__init__()
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
                shape=(num_mod, self.enc_dim, input_dim), dtype="float32"),
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
        inp_enc = tf.linalg.matvec(self.W, inp) + self.B  # (b, t, m, d)
        return inp_enc


class ModEncodingBlock(layers.Layer):
    """Modality encodings layer."""

    def __init__(self, enc_dim=128):
        super().__init__()
        self.enc_dim = enc_dim

    def build(self, input_shape):
        # Input shapes
        num_mod = input_shape[-2]
        # Embedding layer
        self.embedding_layer = layers.Embedding(
            num_mod,
            self.enc_dim,
            input_length=num_mod
        )
        # Modality --> integers
        self.mods = rearrange(tf.range(num_mod), 'm -> 1 1 m')

    def call(self, inp):
        """
        Input shapes:
          inp: (b, t, m, i)
        Output shapes:
          return: (1, 1, m, d)
        """
        # Calculate modality data encodings
        mod_enc = self.embedding_layer(self.mods)  # (1, 1, m, d)
        return mod_enc


class SeqAttentionBlock(layers.Layer):
    """Sequential attention layer."""

    def __init__(self, proj_dim=128, num_head=4, drop_rate=0.2, 
                 causal_mask=False, equivar=False):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_head = num_head
        self.embed_dim = proj_dim // num_head
        self.dropout = layers.Dropout(drop_rate)
        self.causal_mask = causal_mask
        self.equivar = equivar

    def build(self, input_shape):
        # Input shapes
        input_dim = input_shape[-1]
        num_mod = input_shape[-2]
        # Weight and bias initializers
        w_init = tf.keras.initializers.glorot_uniform()
        b_init = tf.zeros_initializer()
        # Weight matrix and bias: query
        self.Wq = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, self.proj_dim, input_dim), dtype="float32"),
            trainable=True
        )
        self.Bq = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.proj_dim), dtype="float32"),
            trainable=True
        )
        # Weight matrix and bias: key
        self.Wk = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, self.proj_dim, input_dim), dtype="float32"),
            trainable=True
        )
        self.Bk = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.proj_dim), dtype="float32"),
            trainable=True
        )
        # Weight matrix and bias: value
        self.Wv = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, self.proj_dim, input_dim), dtype="float32"),
            trainable=True
        )
        self.Bv = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.proj_dim), dtype="float32"),
            trainable=True
        )
        if self.equivar:
            ## Scalable approach
            # Dense layers: time encodings
            self.time_dense = layers.Dense(self.proj_dim)
            # Weight matrix and bias: query/key for time
            self.Wt = tf.Variable(
                initial_value=w_init(
                    shape=(num_mod, self.proj_dim, input_dim), dtype="float32"),
                trainable=True
            )
            self.Bt = tf.Variable(
                initial_value=b_init(
                    shape=(num_mod, self.proj_dim), dtype="float32"),
                trainable=True
            )

    def call(self, inp, pos, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          pos:  (b, t, t, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, p)
        """
        # Project query, key and value
        q = tf.linalg.matvec(self.Wq, inp) + self.Bq
        k = tf.linalg.matvec(self.Wk, inp) + self.Bk
        v = tf.linalg.matvec(self.Wv, inp) + self.Bv
        # Separate heads
        q = rearrange(q, 'b t m (h e) -> b m h t e',
                      h=self.num_head)  # (b, m, h, t, e)
        k = rearrange(k, 'b t m (h e) -> b m h t e',
                      h=self.num_head)  # (b, m, h, t, e)
        v = rearrange(v, 'b t m (h e) -> b m h t e',
                      h=self.num_head)  # (b, m, h, t, e)
        # Calculate attention scores
        score = tf.einsum('...ij,...kj->...ik', q, k)
        score = score / (self.embed_dim**0.5)  # (b, m, h, t, t)

        if self.equivar:
            # Project query/key for time
            q_t = tf.linalg.matvec(self.Wt, inp) + self.Bt
            q_t = rearrange(q_t, 'b t m (h e) -> b m h t e',
                            h=self.num_head)  # (b, m, h, t, e)
            t = self.time_dense(pos)
            t = rearrange(t, 'b t l (h e) -> b 1 h t l e',
                          h=self.num_head)  # (b, 1, h, t, t, e)
            score = score + tf.linalg.matvec(t, q_t)/(self.embed_dim**0.5)

        # Apply mask and causal mask if needed
        causal_mask = None
        if self.causal_mask:
            t = tf.shape(score)[-1]
            causal_mask = LinearOperatorLowerTriangular(
                tf.ones([t, t], dtype='bool')
            ).to_dense()  # (t, t)
            score = tf.where(causal_mask, score, -np.inf)

        if mask is not None:
            mask = rearrange(mask, 'b t m -> b m 1 1 t')
            score = tf.where(mask, score, -np.inf)
        
        # Calculate attention weights
        weight = tf.nn.softmax(score)  # (b, m, h, t, t)
        if mask is not None:
            weight = tf.where(mask, weight, 0)
        if causal_mask is not None:
            weight = tf.where(causal_mask, weight, 0)
        # Check if there is any NaN in weights
        tf.debugging.check_numerics(
            tensor=weight,
            message="Check the values after softmax")
        # Apply dropout
        weight = self.dropout(weight)
        # Calculate attention output
        out = tf.einsum('...ij,...jk->...ik', weight, v)  # (b, m, h, t, e)
        # Concatenate heads
        concat_out = rearrange(
            out, 'b m h t e -> b t m (h e)')  # (b, t, m, p)
        return concat_out


class ModAttentionBlock(layers.Layer):
    """Modality attention layer."""

    def __init__(self, proj_dim=128, num_head=4, drop_rate=0.2):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_head = num_head
        self.embed_dim = proj_dim // num_head
        self.dropout = layers.Dropout(drop_rate)

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
        # Calculate attention scores
        score = tf.einsum('...ij,...kj->...ik', q, k)
        score = score / (self.embed_dim**0.5)  # (b, t, h, m, m)
        if mask is not None:
            mask = rearrange(mask, 'b t m -> b t 1 1 m')
            score = tf.where(mask, score, -np.inf)
        # Calculate attention weights
        weight = tf.nn.softmax(score)  # (b, t, h, m, m)
        if mask is not None:
            weight = tf.where(mask, weight, 0)
        # Check if there is any NaN in weights
        tf.debugging.check_numerics(
            tensor=weight,
            message="Check the values after softmax"
        )
        # Apply dropout
        weight = self.dropout(weight)
        # Calculate attention outputs
        out = tf.einsum('...ij,...jk->...ik', weight, v)  # (b, t, h, m, e)
        # Concatenate heads
        concat_out = rearrange(
            out, 'b t h m e -> b t m (h e)')  # (b, t, m, p)
        return concat_out


class AxialMultiHeadAttentionBlock(layers.Layer):
    """Multi head attention layer."""

    def __init__(self, proj_dim=128, enc_dim=128, num_head=4,
                 drop_rate=0.2, causal_mask=False, equivar=False):
        super().__init__()
        self.seqAttention = SeqAttentionBlock(
            proj_dim=proj_dim, num_head=num_head, drop_rate=drop_rate,
            causal_mask=causal_mask, equivar=equivar
        )
        self.modAttention = ModAttentionBlock(
            proj_dim=proj_dim, num_head=num_head, drop_rate=drop_rate
        )
        self.enc_dim = enc_dim

    def build(self, input_shape):
        # Input shapes
        input_dim = input_shape[-1]
        num_mod = input_shape[-2]
        # Weight and bias initializers
        w_init = tf.keras.initializers.glorot_uniform()
        b_init = tf.zeros_initializer()
        # Weight matrix and bias: map proj_dim to data_dim (=enc_dim)
        self.W = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, self.enc_dim, input_dim), dtype="float32"),
            trainable=True
        )
        self.B = tf.Variable(
            initial_value=b_init(
                shape=(num_mod, self.enc_dim), dtype="float32"),
            trainable=True
        )

    def call(self, inp, pos, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          pos:  (b, t, t, d)
          mod:  (b, 1, m, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        # Attention over timestamps
        out = self.seqAttention(inp, pos, mask)  # (b, t, m, p)
        # Attention over modalities
        out = self.modAttention(out, mask)  # (b, t, m, p)
        # Linear projection to encoding dimension
        out = tf.linalg.matvec(self.W, out) + self.B  # (b, t, m, d)
        return out


class PosFeedforwardBlock(layers.Layer):
    """Positional feedforward network layer."""

    def __init__(self, enc_dim=128, ff_dim=128, drop_rate=0.2):
        super().__init__()
        self.enc_dim = enc_dim
        self.ff_dim = ff_dim
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(drop_rate)

    def build(self, input_shape):
        # Input shapes
        input_dim = input_shape[-1]
        num_mod = input_shape[-2]
        # Weight and bias initializers
        w_init = tf.keras.initializers.glorot_uniform()
        b_init = tf.zeros_initializer()
        # Weight matrix and bias: 1st feedforward layer
        self.W1 = tf.Variable(
            initial_value=w_init(
                shape=(num_mod, self.ff_dim, input_dim), dtype="float32"),
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
                shape=(num_mod, self.enc_dim, self.ff_dim), dtype="float32"),
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
          inp:  (b, t, m, d)
        Output shapes:
          return: (b, t, m, d)
        """
        # Positionwise feedforward network
        out = tf.linalg.matvec(self.W1, inp) + self.B1
        out = self.relu(out)
        out = self.dropout(out)
        out = tf.linalg.matvec(self.W2, out) + self.B2
        return out
