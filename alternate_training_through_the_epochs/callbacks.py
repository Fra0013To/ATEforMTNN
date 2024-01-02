"""
This module is dedicated to 'extended' versions of the keras callbacks, in order to make them suitable for an
Alternate Training through the Epochs (ATE) procedure.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging


class EarlyStoppingExtended(tf.keras.callbacks.EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 start_from_epoch=0,
                 starting_wait=None,
                 starting_best=None,
                 starting_best_weights=None,
                 starting_best_epoch=0):
        super(EarlyStoppingExtended, self).__init__(monitor=monitor,
                                                    min_delta=min_delta,
                                                    patience=patience,
                                                    verbose=verbose,
                                                    mode=mode,
                                                    baseline=baseline,
                                                    restore_best_weights=restore_best_weights,
                                                    start_from_epoch=start_from_epoch
                                                    )

        if starting_wait is None:
            self.starting_wait = self.wait
        else:
            self.starting_wait = starting_wait

        self.starting_best = starting_best
        self.starting_best_weights = starting_best_weights
        self.starting_best_epoch = starting_best_epoch

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = self.starting_wait
        self.stopped_epoch = 0
        if self.starting_best is None:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        else:
            self.best = self.starting_best
        self.best_weights = self.starting_best_weights
        self.best_epoch = self.starting_best_epoch


class ReduceLROnPlateauExtended(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 starting_wait=None,
                 starting_cooldown_counter=None,
                 starting_best=None,
                 **kwargs):
        super(ReduceLROnPlateauExtended, self).__init__(monitor=monitor,
                                                        factor=factor,
                                                        patience=patience,
                                                        verbose=verbose,
                                                        mode=mode,
                                                        min_delta=min_delta,
                                                        cooldown=cooldown,
                                                        min_lr=min_lr,
                                                        **kwargs)

        if starting_wait is None:
            self.starting_wait = self.wait
        else:
            self.starting_wait = starting_wait

        if starting_best is None:
            self.starting_best = self.best
        else:
            self.starting_best = starting_best

        if starting_cooldown_counter is None:
            self.starting_cooldown_counter = self.cooldown_counter
        else:
            self.starting_cooldown_counter = starting_cooldown_counter

    def _set_starters(self):
        """set the starting wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf

        if self.starting_best is not None:
            self.best = self.starting_best

        self.cooldown_counter = self.starting_cooldown_counter
        self.wait = self.starting_wait

    def on_train_begin(self, logs=None):
        self._set_starters()





