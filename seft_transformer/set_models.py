import tensorflow as tf
from tensorflow import keras


class PaddedToSegments(keras.layers.Layer):
    """Convert a padded tensor with mask to a stacked tensor with segments."""

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

    def call(self, inputs, mask):
        valid_observations = tf.where(mask)
        collected_values = tf.gather_nd(inputs, valid_observations)
        return collected_values, valid_observations[:, 0]


class TimeSeriesTransformer(keras.Model):
    """Time Series Transformer model."""

    def __init__(self, proj_dim=128, num_head=4, enc_dim=128,
                 pos_ff_dim=128, pred_ff_dim=32, drop_rate=0.2,
                 norm_type='reZero', dataset='physionet2012',
                 equivar=False, num_layers=1, no_time=False):
        super(TimeSeriesTransformer, self).__init__()
        if dataset == 'physionet2019':
            self.causal_mask = True
        else:
            self.causal_mask = False
        self.to_segments = PaddedToSegments()
        self.W_k = keras.Dense(128, bias=False)
        self.W_q = keras.Dense(128, bias=False)
        self.W_v = keras.Dense(128, bias=False)

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
        time = inputs[1]  # (bs, n_set_elements)
        inp = inputs[2]  # (bs, n_set_elements)
        modality_index = inputs[3]  # (bs, n_set_elements)
        lengths = inputs[4]  # (bs)
        mask = tf.sequence_mask(lengths, name='mask')
        time_set, segment_ids = self.to_segments(time, mask)
        inp_set, _ = self.to_segments(inp, mask)
        modality_set, _ self.to_segments(modality_index, mask)
        inp_embedded_set = self.inp_embedding(inp_set)

        keys = self.W_k(inp_embedded_set)
        queries = self.W_q(inp_embedded_set)
        values = self.W_v(inp_embedded_set)

        unique_segment_ids, _ = tf.unique(segment_ids)
        n_segments = unique_segment_ids.shape[0]

        output_array = tf.TensorArray(
            inp_embedded_set.dtype, size=n_segments, infer_shape=False)
        output_segments = tf.TensorArray(
            segment_ids.dtype, size=n_segements, infer_shape=False)

        def loop_condition(i, out, out_seg):
            return i < n_segments

        def loop_body(i, out, out_seg):
            cur_segment = unique_segment_ids[i]
            cur_indices = tf.where(tf.equal(segment_ids, cur_segment))

            k = tf.gather_nd(keys, cur_indices)
            q = tf.gather_nd(queries, cur_indices)
            v = tf.gather_nd(values, cur_indices)
            q = rearrange(q, 'b t m (h e) -> b m h t e',
                          h=self.num_head)  # (b, m, h, t, e)
            k = rearrange(k, 'b t m (h e) -> b m h t e',
                          h=self.num_head)  # (b, m, h, t, e)
            v = rearrange(v, 'b t m (h e) -> b m h t e',
                          h=self.num_head)  # (b, m, h, t, e)
            score = tf.einsum('...ij,...kj->...ik', q, k)
            score = score / (self.embed_dim**0.5)  # (b, m, h, t, t)
            weight = tf.nn.softmax(score)  # (b, m, h, t, t)
            weight = self.dropout(weight)
            out = tf.einsum('...ij,...jk->...ik', weight, v)  # (b, m, h, t, e)
            concat_out = rearrange(
                out, 'b m h t e -> b t m (h e)')  # (b, t, m, p)
            out_seg = out_seg.write(i, cur_indices)
            out = out.write(i, concat_out)
            return i+1, out, out_seg

        i_end, output_array, output_segments = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=(tf.constant(0), output_array, output_segments)
        )
        output_tensor = output_tensor.concat()
        output_segments = output_segments.concat()
        out = tf.scatter_nd(output_segments, output_tensor, values.shape)

        # Expand input dimensions if necessary
        # Encode inputs
        inp_enc, pos_enc = self.input_embedding(
            inp, time, mask)

        determine_number
