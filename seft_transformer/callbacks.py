"""Miscellenaous functions."""
import tensorflow as tf


class ReduceLRBacktrack(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, best_path, monitor='val_auprc', 
                 mode='max', factor=0.5, patience=5, 
                 min_lr=1e-8):
        super().__init__(
            monitor=monitor, mode=mode, factor=factor, 
            patience=patience, min_lr=min_lr
        )
        self.best_path = best_path

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if not self.monitor_op(current, self.best): # not new best
            if self.wait+1 >= self.patience: # going to reduce lr
                # load best model so far
                self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs) # actually reduce LR


class WarmUpScheduler(tf.keras.callbacks.Callback):
    def __init__(self, final_lr, init_lr=0.0, warmup_steps=0):
        """Constructor for warmup learning rate scheduler.
        Args:
            final_lr: base learning rate.
            init_lr: Initial learning rate for warm up. (default: 0.0)
            warmup_steps: Number of warmup steps. (default: 0)
        """
        super(WarmUpScheduler, self).__init__()
        self.final_lr = final_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        # Don't do warmup if warmup_steps==1, otherwise linear warmup
        self.increase = (final_lr - init_lr) / warmup_steps
        self.global_step = 1

    def on_train_batch_begin(self, batch, logs=None):
        if self.global_step <= self.warmup_steps:
            new_lr = self.init_lr + (self.increase * self.global_step)
            self.model.optimizer.lr = new_lr
            self.global_step += 1


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
    
    def on_epoch_begin(self, epoch, logs=None):
        tf.summary.scalar(
            name='learning rate', 
            data=self.model.optimizer.lr, 
            step=epoch
        )

class BatchPrinter(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BatchPrinter, self).__init__()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.first_batch = True

    """
    def on_train_batch_begin(self, batch, logs=None):
        if self.global_step == 1:
            tf.summary.scalar(
                name='learning rate in first epoch', 
                data=tf.keras.backend.get_value(self.model.optimizer.lr), 
                step=batch
            )

    def on_epoch_end(self, epoch, logs=None):
        self.global_step += 1

    class TimeEncWeightLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar(
            name='time enc weight',
            data=self.model.input_embedding.w_t,
            step=epoch
        )
    """
