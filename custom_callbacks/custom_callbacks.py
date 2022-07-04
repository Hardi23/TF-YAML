import wandb
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import Callback
from wandb.keras import WandbCallback


class WandbCallbackWrapper(WandbCallback):
    def __new__(cls, *args, **kwargs):
        return WandbCallback(monitor="val_loss", verbose=1, log_weights=True)


class LRLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"lr": backend.eval(self.model.optimizer.learning_rate)})
        return super().on_epoch_begin(epoch, logs)
