import tensorflow as tf
from tensorflow.linalg import LinearOperatorLowerTriangular
from tensorflow.keras import layers
from einops import rearrange, repeat
import numpy as np
import sys


class PosEncodingBlock(layers.Layer):
    """Positional encodings layer."""

    def __init__(self, enc_dim=128, equivar=False,
                 train_time_enc=False):
        super().__init__()
        f = tf.math.exp(
            tf.range(start=0, limit=enc_dim, delta=2, dtype="float32")
            * -(tf.math.log(10000.0) / enc_dim)
        )

        self.f = tf.Variable(f, trainable=train_time_enc)
        self.equivar = equivar

    def call(self, time):
        """
        Input shapes:
          time: (n, 1)
        Output shapes:
          return: (n, n, d) if equivar
                  (n, d) else
        """
        if self.equivar:
            rel_time = rearrange(time, 'n -> n 1') - \
                 rearrange(time, 'n -> 1 n')  #(n, n)
            # Calculate sine and cosine components
            angles = tf.einsum(
                'tl,f->tlf', rel_time, self.f)  # (n, n, d/2)
            sin_enc = tf.math.sin(angles)  # sin encodings (n, n, d/2)
            cos_enc = tf.math.cos(angles)  # cos encodings (n, n, d/2)
            # Construct positional encodings
            pos_enc = rearrange(
                [sin_enc, cos_enc],  'z t l k -> t l (k z)')
            return pos_enc  # (n, n, d)
        else:
            # Calculate sine and cosine components
            angles = tf.einsum(
                'n,f->nf', tf.squeeze(time), self.f)  # (n, d/2)
            sin_enc = tf.math.sin(angles)  # sin encodings (n, d/2)
            cos_enc = tf.math.cos(angles)  # cos encodings (n, d/2)
            # Construct positional encodings
            pos_enc = rearrange(
                [sin_enc, cos_enc],  'z n k -> n (k z)')  # (n, d)
            return pos_enc  # (n, d)


class InpEncodingBlock(layers.Layer):
    """Input encodings layer."""

    def __init__(self, enc_dim=128):
        super().__init__()
        self.enc_dim = enc_dim
        self.dense = layers.Dense(enc_dim)

    def call(self, inp):
        """
        Input shapes:
          inp: (n, 1)
        Output shapes:
          return: (n, d)
        """
        # Calculate input data encodings
        inp_enc = self.dense(inp)
        return inp_enc


class ModEncodingBlock(layers.Layer):
    """Modality encodings layer."""

    def __init__(self, enc_dim=128):
        super().__init__()
        self.enc_dim = enc_dim
        # Embedding layer
        self.embedding_layer = layers.Embedding(
            37,
            self.enc_dim
        )

    def call(self, mod):
        """
        Input shapes:
          mod: (n, 1)
        Output shapes:
          return: (n, d)
        """
        # Calculate modality data encodings
        mod_enc = self.embedding_layer(mod)  # (n, 1, d)
        return tf.squeeze(mod_enc, axis=1)  # (n, d)


class SeqAttentionBlock(layers.Layer):
    """Sequential attention layer."""

    def __init__(self, proj_dim=128, enc_dim=128, num_head=4, 
                 drop_rate=0.2, causal_mask=False, equivar=False):
        super().__init__()
        self.num_head = num_head
        self.embed_dim = proj_dim // num_head
        self.dropout = layers.Dropout(drop_rate)
        self.causal_mask = causal_mask
        self.equivar = equivar

        self.query_dense = layers.Dense(proj_dim)
        self.key_dense = layers.Dense(proj_dim)
        self.value_dense = layers.Dense(proj_dim)

        if equivar:
            self.time_dense = layers.Dense(proj_dim)
            self.qt_dense = layers.Dense(proj_dim)
            self.pos_encoding = PosEncodingBlock(enc_dim, equivar)

    def call(self, inp, pos):
        """
        Input shapes:
          inp:  (n_t, d)
          pos:  (n_t,)
        Output shapes:
          return: (n_t, p)
        """
        # Create query, key and value
        q = self.query_dense(inp)  # (n_t, p)
        k = self.key_dense(inp)    # (n_t, p)
        v = self.value_dense(inp)  # (n_t, p)
        # Separate heads
        q = rearrange(q, 'n (h e) -> h n e',
                      h=self.num_head)  # (h, n_t, e)
        k = rearrange(k, 'n (h e) -> h n e',
                      h=self.num_head)  # (h, n_t, e)
        v = rearrange(v, 'n (h e) -> h n e',
                      h=self.num_head)  # (h, n_t, e)
        # Calculate attention scores
        score = tf.einsum('...ij,...kj->...ik', q, k)
        score = score / (self.embed_dim**0.5)  # (h, n_t, e)

        if self.equivar:
            # Calculate relative positional encodings
            pos = self.pos_encoding(pos)  # (n_t, n_t, d)
            # Project query/key for time
            q_t = self.qt_dense(inp)  # (n_t, p)
            q_t = rearrange(q_t, 'n (h e) -> h n e',
                            h=self.num_head)  # (h, n_t, e)
            t = self.time_dense(pos)  
            t = rearrange(t, 'n1 n2 (h e) -> h n1 n2 e',
                          h=self.num_head)    # (h, n_t, n_t, e)
            score = score + tf.linalg.matvec(t, q_t)/(self.embed_dim**0.5)

        # Apply mask and causal mask if needed
        causal_mask = None

        if self.causal_mask:
            t = tf.shape(score)[-1]
            causal_mask = LinearOperatorLowerTriangular(
                tf.ones([t, t], dtype='bool')
            ).to_dense()  # (t, t)
            score = tf.where(causal_mask, score, -np.inf)

        # Calculate attention weights
        weight = tf.nn.softmax(score)  # (h, n_t, n_t)

        if causal_mask is not None:
            weight = tf.where(causal_mask, weight, 0)

        # Check if there is any NaN in weights
        tf.debugging.check_numerics(
            tensor=weight,
            message="Check the values after softmax")

        # Apply dropout
        weight = self.dropout(weight)  # (h, n_t, n_t)

        # Calculate attention output
        out = tf.einsum('...ij,...jk->...ik', weight, v)  # (h, n_t, e)

        # Concatenate heads
        concat_out = rearrange(
            out, 'h n e -> n (h e)')  # (n_t, p) 

        return concat_out


class ModAttentionBlock(layers.Layer):
    """Modality attention layer."""

    def __init__(self, proj_dim=128, num_head=4, drop_rate=0.2):
        super().__init__()

        self.num_head = num_head
        self.embed_dim = proj_dim // num_head
        self.dropout = layers.Dropout(drop_rate)

        self.query_dense = layers.Dense(proj_dim)
        self.key_dense = layers.Dense(proj_dim)
        self.value_dense = layers.Dense(proj_dim)

    def call(self, inp):
        """
        Input shapes:
          inp:  (n_m, p)
        Output shapes:
          return: (n_m, p)
        """
        # Create query, key and value
        q = self.query_dense(inp)  # (n_m, p)
        k = self.key_dense(inp)    # (n_m, p)
        v = self.value_dense(inp)  # (n_m, p)

        # Separate heads
        q = rearrange(q, 'n (h e) -> h n e',
                      h=self.num_head)  # (h, n_m, e)
        k = rearrange(k, 'n (h e) -> h n e',
                      h=self.num_head)  # (h, n_m, e)
        v = rearrange(v, 'n (h e) -> h n e',
                      h=self.num_head)  # (h, n_m, e)

        # Calculate attention scores
        score = tf.einsum('...ij,...kj->...ik', q, k)
        score = score / (self.embed_dim**0.5)  # (h, n_m, n_m)

        # Calculate attention weights
        weight = tf.nn.softmax(score)  # (h, n_m, n_m)

        # Check if there is any NaN in weights
        tf.debugging.check_numerics(
            tensor=weight,
            message="Check the values after softmax"
        )

        # Apply dropout
        weight = self.dropout(weight)  # (h, n_m, n_m)

        # Calculate attention outputs
        out = tf.einsum('...ij,...jk->...ik', weight, v)  # (h, n_m, e)

        # Concatenate heads
        concat_out = rearrange(
            out, 'h n e -> n (h e)')  # (n_m, p)

        return concat_out


class ModWhileLoopBlock(layers.Layer):
    """Multi head attention layer."""

    def __init__(self, proj_dim=128, enc_dim=128, num_head=4,
                 drop_rate=0.2, causal_mask=False, equivar=False):
        super().__init__()

        self.proj_dim = proj_dim

        self.seqAttention = SeqAttentionBlock(
            proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
            drop_rate=drop_rate, causal_mask=causal_mask, equivar=equivar
        )

    def call(self, inp, mod, pos):
        """
        Input shapes:
          inp:  (n, d)
          mod:  (n, 1)
          pos:  (n, 1)
        Output shapes:
          return: (n, p)
        """
        # Get variables for mod segment ids
        mod_seg_ids, _ = tf.unique(mod)
        n_mod_seg = tf.shape(mod_seg_ids)[0]

        mod_out_arr = tf.TensorArray(
            inp.dtype, size=n_mod_seg, infer_shape=False)
        mod_out_seg = tf.TensorArray(
            tf.int32, size=n_mod_seg, infer_shape=False)

        def loop_cond(i, out, out_seg):
            return i < n_mod_seg

        def loop_body(i, out, out_seg):
            curr_seg = mod_seg_ids[i]
            curr_ind = tf.cast(tf.where(tf.equal(mod, curr_seg)),
                               dtype=tf.int32)
            curr_inp = tf.gather_nd(inp, curr_ind)  # (n_i, d)
            curr_pos = tf.gather_nd(pos, curr_ind)  # (n_i, 1)
            
            # Sequence attention
            attn = self.seqAttention(curr_inp, curr_pos)  # (n_i, p)

            out_seg = out_seg.write(i, curr_ind)
            out = out.write(i, attn)

            return i+1, out, out_seg

        i_end, mod_out_arr, mod_out_seg = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=(tf.constant(0), mod_out_arr, mod_out_seg)
        )

        # Concat batch outputs
        out_arr = mod_out_arr.concat()
        out_seg = mod_out_seg.concat()
        out = tf.scatter_nd(out_seg, out_arr, tf.shape(inp))

        return out


class SeqWhileLoopBlock(layers.Layer):
    """Multi head attention layer."""

    def __init__(self, proj_dim=128, num_head=4, drop_rate=0.2):
        super().__init__()

        self.modAttention = ModAttentionBlock(
            proj_dim=proj_dim, num_head=num_head, drop_rate=drop_rate
        )

    def call(self, inp, pos):
        """
        Input shapes:
          inp:  (n, p)
          pos:  (n, 1)
        Output shapes:
          return: (n, p)
        """
        # Get variables for pos segment ids
        pos_seg_ids, _ = tf.unique(pos)
        n_pos_seg = tf.shape(pos_seg_ids)[0]

        pos_out_arr = tf.TensorArray(
            inp.dtype, size=n_pos_seg, infer_shape=False)
        pos_out_seg = tf.TensorArray(
            tf.int32, size=n_pos_seg, infer_shape=False)

        def loop_cond(i, out, out_seg):
            return i < n_pos_seg

        def loop_body(i, out, out_seg):
            curr_seg = pos_seg_ids[i]
            curr_ind = tf.cast(tf.where(tf.equal(pos, curr_seg)),
                               dtype=tf.int32)
            curr_inp = tf.gather_nd(inp, curr_ind)  # (n_i, p)
            
            # Modality attention
            attn = self.modAttention(curr_inp)  # (n_i, p)
            
            out_seg = out_seg.write(i, curr_ind)
            out = out.write(i, attn)

            return i+1, out, out_seg

        i_end, pos_out_arr, pos_out_seg = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=(tf.constant(0), pos_out_arr, pos_out_seg)
        )

        # Concat batch outputs
        out_arr = pos_out_arr.concat()
        out_seg = pos_out_seg.concat()
        out = tf.scatter_nd(out_seg, out_arr, tf.shape(inp))
        
        return out


class MultiHeadAttentionBlock(layers.Layer):
    """Multi head attention layer."""

    def __init__(self, proj_dim=128, enc_dim=128, num_head=4,
                 drop_rate=0.2, causal_mask=False, equivar=False,
                 ax_attn=False):
        super().__init__()

        self.ax_attn = ax_attn
        self.proj_dim = proj_dim
        self.dense = layers.Dense(enc_dim)

        if ax_attn:
            self.mod_while_loop = ModWhileLoopBlock(
                proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
                drop_rate=drop_rate, causal_mask=causal_mask, equivar=equivar
            )

            self.seq_while_loop = SeqWhileLoopBlock(
                proj_dim=proj_dim, num_head=num_head, drop_rate=drop_rate
            )
        else:
            self.attention = SeqAttentionBlock(
                proj_dim=proj_dim, enc_dim=enc_dim, num_head=num_head,
                drop_rate=drop_rate, causal_mask=causal_mask, equivar=equivar
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
        # Get variables for batch segment ids
        batch_seg_ids, _ = tf.unique(batch_seg)
        n_batch_seg = tf.shape(batch_seg_ids)[0]

        # Get variables for mod segment ids
        mod_seg_ids, _ = tf.unique(mod)
        n_mod_seg = tf.shape(mod_seg_ids)[0]

        # Get variables for pos segment ids
        pos_seg_ids, _ = tf.unique(pos)
        n_pos_seg = tf.shape(pos_seg_ids)[0]

        batch_out_arr = tf.TensorArray(
            inp.dtype, size=n_batch_seg, infer_shape=False)
        batch_out_seg = tf.TensorArray(
            tf.int32, size=n_batch_seg, infer_shape=False)
        
    
        def loop_cond(i, out, out_seg):
            return tf.math.less(i, n_batch_seg)

       
        def loop_body(i, out, out_seg):
            curr_seg = batch_seg_ids[i]
            curr_ind = tf.cast(
                tf.where(tf.equal(batch_seg, curr_seg)),
                dtype=tf.int32
            )

            curr_inp = tf.gather_nd(inp, curr_ind) # (n_i, d)
            curr_pos = tf.gather_nd(pos, curr_ind) # (n_i, 1)
            curr_mod = tf.gather_nd(mod, curr_ind) # (n_i, 1)
            
            if self.ax_attn:
                # Sequence attention
                pos_attn = self.mod_while_loop(
                    curr_inp, curr_mod, curr_pos)  # (n_i, p)
                # Modality attention
                attn = self.seq_while_loop(
                    pos_attn, curr_pos)  # (n_i, p)
            else:
                # Attention
                attn = self.attention(curr_inp, curr_pos)  # (n_i, p)

            out_seg = out_seg.write(i, curr_ind)
            out = out.write(i, attn)

            return tf.math.add(i, 1), out, out_seg

        i_end, batch_out_arr, batch_out_seg = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=(tf.constant(0), batch_out_arr, batch_out_seg)
        )

        # Concat batch outputs
        out = batch_out_arr.concat()
        out.set_shape([None, self.proj_dim])  # (n, p)
        
        # Linear projection to encoding dimension
        out = self.dense(out)  # (n, d)

        return out


class PosFeedforwardBlock(layers.Layer):
    """Positional feedforward network layer."""

    def __init__(self, enc_dim=128, ff_dim=128, drop_rate=0.2):
        super().__init__()
        self.enc_dim = enc_dim
        self.ff_dim = ff_dim
        self.dropout = layers.Dropout(drop_rate)
        self.dense1 = layers.Dense(ff_dim, activation='relu')
        self.dense2 = layers.Dense(enc_dim)

    def call(self, inp):
        """
        Input shapes:
          inp:  (n, d)
        Output shapes:
          return: (n, d)
        """
        # Positionwise feedforward network
        out = self.dense1(inp)   # (n, f)
        out = self.dropout(out)
        out = self.dense2(out)   # (n, d)
        
        return out  # (n, d)
