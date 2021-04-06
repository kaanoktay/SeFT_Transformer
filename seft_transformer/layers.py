"""Implementation of Layers used by Keras models."""
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from einops import rearrange, reduce
import sys

from .blocks import (
    AxialMultiHeadAttentionBlock,
    UniAxialMultiHeadAttentionBlock,
    PosEncodingBlock,
    InpEncodingBlock,
    UniInpEncodingBlock,
    ModEncodingBlock,
    PosFeedforwardBlock,
    UniPosFeedforwardBlock
)


class InputEmbedding(layers.Layer):
    def __init__(self, enc_dim=128, no_time=False, 
                 uni_mod=False, train_time_enc=False):
        super().__init__()

        self.pos_encoding = PosEncodingBlock(
            enc_dim=enc_dim, train_time_enc=train_time_enc
        )

        if uni_mod:
            self.inp_encoding = UniInpEncodingBlock(
                enc_dim=enc_dim)
        else:
            self.inp_encoding = InpEncodingBlock(
                enc_dim=enc_dim)

        self.mod_encoding = ModEncodingBlock(
            enc_dim=enc_dim
        )

        self.no_time = no_time

    def call(self, inp, time, mask):
        """
        Input shapes:
          inp:  (b, t, m, i)
          time: (b, t)
        Output shapes:
          return: (b, t, m, d), None          if no_time
                  (b, t, m, d), (b, t, 1, d)  else
        """
        pos_enc = self.pos_encoding(time)
        inp_enc = self.inp_encoding(inp)
        mod_enc = self.mod_encoding(inp)
        
        if self.no_time:
            return inp_enc + mod_enc, None
        else:
            return inp_enc + mod_enc, pos_enc


class ReZero(layers.Layer):
    def __init__(self):
        super(ReZero, self).__init__()

        self.re_weight = tf.Variable(
            initial_value=0.0,
            trainable=True
        )

    def call(self, x1, x2):
        return x1 + self.re_weight * x2


class AxialAttentionEncoder(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128, 
                 num_head=4, ff_dim=128, drop_rate=0.2, 
                 norm_type="reZero", causal_mask=False,
                 equivar=False, uni_mod=False):
        super().__init__()

        if uni_mod:
            self.axAttention = UniAxialMultiHeadAttentionBlock(
                proj_dim=proj_dim, enc_dim=enc_dim,
                num_head=num_head, drop_rate=drop_rate,
                causal_mask=causal_mask, equivar=equivar
            )
            self.posFeedforward = UniPosFeedforwardBlock(
                enc_dim=enc_dim, ff_dim=ff_dim,
                drop_rate=drop_rate
            )
        else:
            self.axAttention = AxialMultiHeadAttentionBlock(
                proj_dim=proj_dim, enc_dim=enc_dim,
                num_head=num_head, drop_rate=drop_rate,
                causal_mask=causal_mask, equivar=equivar
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


class MultiLayerAttention(layers.Layer):
    def __init__(self, proj_dim=128, enc_dim=128,
                 num_head=4, ff_dim=128, drop_rate=0.2, 
                 norm_type="reZero", causal_mask=False,
                 equivar=False, uni_mod=False, num_layers=1):
        super().__init__()

        self.num_layers = num_layers

        self.layer1 = AxialAttentionEncoder(
                          proj_dim=proj_dim, enc_dim=enc_dim,
                          num_head=num_head, ff_dim=ff_dim, 
                          drop_rate=drop_rate, norm_type=norm_type, 
                          causal_mask=causal_mask, equivar=equivar,
                          uni_mod=uni_mod)

        if num_layers >= 2: 
            self.layer2 = AxialAttentionEncoder(
                            proj_dim=proj_dim, enc_dim=enc_dim,
                            num_head=num_head, ff_dim=ff_dim, 
                            drop_rate=drop_rate, norm_type=norm_type, 
                            causal_mask=causal_mask, equivar=equivar,
                            uni_mod=uni_mod)
        if num_layers >= 3: 
            self.layer3 = AxialAttentionEncoder(
                            proj_dim=proj_dim, enc_dim=enc_dim,
                            num_head=num_head, ff_dim=ff_dim, 
                            drop_rate=drop_rate, norm_type=norm_type, 
                            causal_mask=causal_mask, equivar=equivar,
                            uni_mod=uni_mod)
        
        if num_layers >= 4: 
            self.layer4 = AxialAttentionEncoder(
                            proj_dim=proj_dim, enc_dim=enc_dim,
                            num_head=num_head, ff_dim=ff_dim, 
                            drop_rate=drop_rate, norm_type=norm_type, 
                            causal_mask=causal_mask, equivar=equivar,
                            uni_mod=uni_mod)
        
    def call(self, inp, pos, mask):
        """
        Input shapes:
          inp:  (b, t, m, d)
          pos:  (b, t, 1, d)
          mask: (b, t, m)
        Output shapes:
          return: (b, t, m, d)
        """
        
        attn = self.layer1(inp, pos, mask)

        if self.num_layers >= 2:
            attn = self.layer2(attn, pos, mask)
        if self.num_layers >= 3:
            attn = self.layer3(attn, pos, mask)
        if self.num_layers >= 4:
            attn = self.layer4(attn, pos, mask)

        return attn


class ClassPrediction(layers.Layer):
    """Layer for predicting a class output from a series."""

    def __init__(self, ff_dim=32, drop_rate=0.2, causal_mask=False):
        super().__init__()
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
        # Calculate sum over the modalities
        if self.causal_mask:
            out = reduce(inp, 'b t m f -> b t f', 'sum')
        # Calculate sum over the timesteps and modalities
        else:
            out = reduce(inp, 'b t m f -> b f', 'sum')
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
        pred = self.densePred2(self.densePred1(self.dropout(out))) # (b, 1)

        return pred  # if causal_mask (b, t, 1) else (b, 1)


class FuncRepresentation(keras.layers.Layer):
    def __init__(self, points_per_hour):
        super().__init__()

        self.sigma = tf.Variable(
            initial_value=2*(1/points_per_hour) * tf.ones(2), 
            dtype=tf.float32,
            trainable=True
        )

        num_points = int(50 * points_per_hour)
        grid = tf.linspace(-1.0, 49.0, num_points)

        self.num_points = num_points
        self.grid = grid

    def call(self, y, x, mask):
        batch_size = tf.shape(y)[0]
        num_mod = y.shape[1]

        grid = tf.repeat(
            self.grid[None,:,None], repeats=batch_size, axis=0
        )

        x = x[:,:,None]
        dist = (grid - tf.transpose(x, perm=[0, 2, 1])) ** 2
        
        repeated_dist = tf.repeat(
            dist[:,None,...], repeats=num_mod, axis=1
        )

        scales = self.sigma[None, None, None, None, :]
        wt = tf.exp(-0.5 * (tf.expand_dims(repeated_dist, -1) / (scales ** 2)))

        density = tf.cast(
            mask, dtype=tf.float32
        )

        y_out = tf.concat(
            [tf.expand_dims(density, -1), tf.expand_dims(y, -1)], axis=-1
        )

        y = tf.expand_dims(y_out, 2) * wt

        func = tf.reduce_sum(y, -2)

        density, conv = func[..., :1], func[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        func = tf.concat([density, normalized_conv], axis=-1)
        func = tf.transpose(func, perm=[0, 1, 3, 2])  
        func = tf.reshape(func, shape=[-1,num_mod*2, self.num_points])

        return tf.transpose(func, perm=[0, 2, 1])


class ConvNet(keras.layers.Layer):
    def __init__(self, kernel_size, dilation_rate, filter_size, 
                 drop_rate_conv, drop_rate_dense):
        super().__init__()

        self.dropout_conv = keras.layers.Dropout(
            rate = drop_rate_conv
        )

        self.dropout_dense_1 = keras.layers.Dropout(
            rate = drop_rate_dense*1.5
        )

        self.dropout_dense_2 = keras.layers.Dropout(
            rate = drop_rate_dense
        )

        self.conv_1 = keras.layers.Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            padding="same"
        )

        self.conv_2 = keras.layers.Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            padding="same"
        )

        self.conv_3 = keras.layers.Conv1D(
            filters=filter_size*2,
            kernel_size=kernel_size,
            padding="same"
        )

        self.conv_4 = keras.layers.Conv1D(
            filters=filter_size*2,
            kernel_size=kernel_size,
            padding="same"
        )

        self.conv_5 = keras.layers.Conv1D(
            filters=filter_size*4,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same"
        )

        self.conv_6 = keras.layers.Conv1D(
            filters=filter_size*4,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same"
        )

        self.dense_1 = keras.layers.Dense(512)

        self.dense_2 = keras.layers.Dense(64)

        self.dense_3 = keras.layers.Dense(1)

        self.pool = keras.layers.MaxPooling1D(
            pool_size=2
        )

        self.flatten = keras.layers.Flatten()

        self.relu = keras.layers.Activation(keras.activations.relu)
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)

    def call(self, x):
        # 1st conv layer
        z = self.relu(self.conv_1(x))

        # 2nd conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.pool(self.conv_2(z)))

        # 3rd conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.conv_3(z))

        # 4th conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.pool(self.conv_4(z)))

        # 5th conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.conv_5(z))

        # 6th conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.pool(self.conv_6(z)))

        # Flatten
        z = self.flatten(z)

        # First dense layer
        z = self.dropout_dense_1(z)
        z = self.relu(self.dense_1(z))

        # Second dense layer
        z = self.dropout_dense_2(z)
        z = self.relu(self.dense_2(z))

        # Last dense layer
        z = self.sigmoid(self.dense_3(z))

        return z


class CausalFuncRepresentation(keras.layers.Layer):
    def __init__(self, points_per_hour):
        super().__init__()

        self.sigma = tf.Variable(
            initial_value=2 * tf.ones(2), 
            dtype=tf.float32,
            trainable=True
        )

        num_points = int(338 * points_per_hour)
        grid = tf.linspace(0.0, 337.0, num_points)

        self.num_points = num_points
        self.grid = grid

    def call(self, y, x, mask):
        batch_size = tf.shape(y)[0]
        num_mod = y.shape[1]

        grid = tf.repeat(
            self.grid[None,:,None], repeats=batch_size, axis=0
        )

        x = x[:,:,None]
        dist = (grid - tf.transpose(x, perm=[0, 2, 1])) ** 2
        
        repeated_dist = tf.repeat(
            dist[:,None,...], repeats=num_mod, axis=1
        )

        scales = self.sigma[None, None, None, None, :]
        wt = tf.exp(-0.5 * (tf.expand_dims(repeated_dist, -1) / (scales ** 2)))

        density = tf.cast(
            mask, dtype=tf.float32
        )

        y_out = tf.concat(
            [tf.expand_dims(density, -1), tf.expand_dims(y, -1)], axis=-1
        )

        y = tf.expand_dims(y_out, 2) * wt

        func = tf.cumsum(y, -2)

        density, conv = func[..., :1], func[..., 1:]
        normalized_conv = conv / (density + 1e-8)

        func = tf.concat([density, normalized_conv], axis=-1)
        func = tf.transpose(func, perm=[0, 1, 4, 3, 2])
        func = tf.reshape(
            func,
            shape=[batch_size, num_mod*2, -1, self.num_points]
        )

        return tf.transpose(func, perm=[0, 2, 3, 1])


class CausalConvNet(keras.layers.Layer):
    def __init__(self, kernel_size, dilation_rate, filter_size, 
                 drop_rate_conv, drop_rate_dense):
        super().__init__()

        self.dropout_conv = layers.Dropout(
            rate = drop_rate_conv
        )

        self.dropout_dense_1 = layers.Dropout(
            rate = drop_rate_dense*1.5
        )

        self.dropout_dense_2 = layers.Dropout(
            rate = drop_rate_dense
        )

        self.conv_1 = layers.TimeDistributed(
            layers.Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.conv_2 = layers.TimeDistributed(
            layers.Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.conv_3 = layers.TimeDistributed(
            layers.Conv1D(
                filters=filter_size*2,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.conv_4 = layers.TimeDistributed(
            layers.Conv1D(
                filters=filter_size*2,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.conv_5 = layers.TimeDistributed(
            layers.Conv1D(
                filters=filter_size*4,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.conv_6 = layers.TimeDistributed(
            layers.Conv1D(
                filters=filter_size*4,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.dense_1 = layers.Dense(512)

        self.dense_2 = layers.Dense(64)

        self.dense_3 = layers.Dense(1)

        self.pool_1 = layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=2)
        )

        self.pool_2 = layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=2)
        )

        self.pool_3 = layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=2)
        )

        self.flatten = layers.TimeDistributed(
            layers.Flatten()
        )

        self.relu = layers.Activation(keras.activations.relu)
        self.sigmoid = layers.Activation(keras.activations.sigmoid)

    def call(self, x):
        # 1st conv layer
        z = self.relu(self.conv_1(x))

        # 2nd conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.pool_1(self.conv_2(z)))

        # 3rd conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.conv_3(z))

        # 4th conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.pool_2(self.conv_4(z)))

        # 5th conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.conv_5(z))

        # 6th conv layer
        z = self.dropout_conv(z)
        z = self.relu(self.pool_3(self.conv_6(z)))

        # Flatten
        z = self.flatten(z)

        # First dense layer
        z = self.dropout_dense_1(z)
        z = self.relu(self.dense_1(z))

        # Second dense layer
        z = self.dropout_dense_2(z)
        z = self.relu(self.dense_2(z))

        # Last dense layer
        z = self.sigmoid(self.dense_3(z))

        return z