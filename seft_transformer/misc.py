"""Miscellenaous functions."""
import tensorflow as tf

class WarmUpScheduler(tf.keras.callbacks.Callback):
    def __init__(self, final_lr, warmup_learning_rate=0.0, warmup_steps=0,
                 verbose=0):
        """Constructor for warmup learning rate scheduler.
        Args:
            learning_rate_base: base learning rate.
            warmup_learning_rate: Initial learning rate for warm up. (default:
                0.0)
            warmup_steps: Number of warmup steps. (default: 0)
            verbose: 0 -> quiet, 1 -> update messages. (default: {0})
        """
        super().__init__()
        self.final_lr = final_lr
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        # Count global steps from 1, allows us to set warmup_steps to zero to
        # skip warmup.
        self.global_step = 1
        self._increase_per_step = \
        self.learning_rates = []
    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
    def on_batch_begin(self, batch, logs=None):
        if self.global_step <= self.warmup_steps:
            increase = \
                (self.final_lr - self.warmup_learning_rate) / self.warmup_steps
            new_lr = self.warmup_learning_rate + (increase * self.global_step)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print(
                    f'Warmup - learning rate: '
                    f'{new_lr:.6f}/{self.final_lr:.6f}',
                    end=''
                )


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        tf.summary.scalar(
            name='learning rate', 
            data=self.model.optimizer.lr, 
            step=epoch)
