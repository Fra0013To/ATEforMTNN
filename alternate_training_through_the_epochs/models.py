import numpy as np
import tensorflow as tf
import alternate_training_through_the_epochs.callbacks as ate_callbacks

class AlternateTrainingEpochsModel(tf.keras.models.Model):
    """
    Alternate Training through the Epochs (ATE) model class (obtained as subclass of tf.keras.models.Model).
    It implements a NN model able to be trained with the ATE procedure described in https://arxiv.org/abs/2312.16340
    by Bellavia S., Della Santa F., Papini A..
    """
    def __init__(self, prefix_layername_shared='trunk', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix_layername_shared = prefix_layername_shared

    def ate_fit(self,
                x=None,
                y=None,
                batch_size=None,
                epochs=1,
                epochs_shared=1,
                epochs_taskspecific=1,
                verbose='auto',
                verbose_alternate=True,
                callbacks=[],
                validation_split=0.0,
                validation_data=None,
                shuffle=True,
                class_weight=None,
                sample_weight=None,
                initial_epoch=0,
                steps_per_epoch=None,
                validation_steps=None,
                validation_batch_size=None,
                validation_freq=1,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                replace_nonextended_original_callbacks=True
                ):
        """
        ATE procedure for the model.
        The method trains alternatively: i) the weights of the layers with name starting with the string stored in the
        attribute _prefix_layername_shared; ii) the weights of all the other layers.
        All the input arguments are the same of the fit method, except for the ones listed below.
        :param epochs_shared: number of consecutive epochs dedicated to the shared parameters (integer)
        :param epochs_taskspecific: number of consecutive epochs dedicated to the task-specific parameters (integer)
        :param verbose_alternate: argument for printing information about which kind of parameters are training (bool)
        :return history: dictionary containing the training history.
        """

        training_configs = {
            'x': x,
            'y': y,
            'batch_size': batch_size,
            'epochs': epochs,
            'verbose': verbose,
            'callbacks': callbacks,
            'validation_split': validation_split,
            'validation_data': validation_data,
            'shuffle': shuffle,
            'class_weight': class_weight,
            'sample_weight': sample_weight,
            'initial_epoch': initial_epoch,
            'steps_per_epoch': steps_per_epoch,
            'validation_steps': validation_steps,
            'validation_batch_size': validation_batch_size,
            'validation_freq': validation_freq,
            'max_queue_size': max_queue_size,
            'workers': workers,
            'use_multiprocessing': use_multiprocessing
            }

        # VALIDATION DATA ARE MANDATORY IF EarlyStopping AND/OR ReduceLROnPlatueau ARE/IS USED

        # ---------------------------------------------------------------------------------------------------------
        # REPLACING EarlyStopping AND ReduceLROnPlateau CALLBACKS WITH "EXTENDED" VERSIONS (IF NOT ALREADY DONE)

        mask_es = [cb.__class__.__name__ == 'EarlyStopping' for cb in callbacks]
        mask_rlrop = [cb.__class__.__name__ == 'ReduceLROnPlateau' for cb in callbacks]

        try:
            ii_es = np.argwhere(mask_es).flatten()[0]
        except IndexError:
            ii_es = None

        try:
            ii_rlprop = np.argwhere(mask_rlrop).flatten()[0]
        except IndexError:
            ii_rlprop = None

        if ii_es is not None:
            es_base = callbacks[ii_es]
            try:
                es_mode = es_base.mode
            except AttributeError:
                es_mode = 'auto'

            training_configs['callbacks'][ii_es] = ate_callbacks.EarlyStoppingExtended(
                monitor=es_base.monitor,
                min_delta=es_base.min_delta,
                patience=es_base.patience,
                verbose=es_base.verbose,
                mode=es_mode,
                baseline=es_base.baseline,
                restore_best_weights=es_base.restore_best_weights,
                start_from_epoch=es_base.start_from_epoch
            )

            if replace_nonextended_original_callbacks:
                callbacks[ii_es] = training_configs['callbacks'][ii_es]

        if ii_rlprop is not None:
            rlprop_base = callbacks[ii_rlprop]

            training_configs['callbacks'][ii_rlprop] = ate_callbacks.ReduceLROnPlateauExtended(
                monitor=rlprop_base.monitor,
                factor=rlprop_base.factor,
                patience=rlprop_base.patience,
                verbose=rlprop_base.verbose,
                mode=rlprop_base.mode,
                min_delta=rlprop_base.min_delta,
                cooldown=rlprop_base.cooldown,
                min_lr=rlprop_base.min_lr
            )

            if replace_nonextended_original_callbacks:
                callbacks[ii_rlprop] = training_configs['callbacks'][ii_rlprop]

        # ----------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------
        # PREPARING CALLBACKS THAT MONITOR VALUES DURING EPOCHS

        tnan = [cb for cb in training_configs['callbacks'] if cb.__class__.__name__ == 'TerminateOnNaN']
        if tnan:
            check_tnan = True
            tnan = tnan[0]
        else:
            check_tnan = False
            tnan = None

        es = [cb for cb in training_configs['callbacks'] if cb.__class__.__name__.startswith('EarlyStopping')]
        if es:
            check_es = True
            es = es[0]
        else:
            check_es = False
            es = None

        rlrop = [cb for cb in training_configs['callbacks'] if cb.__class__.__name__.startswith('ReduceLROnPlateau')]
        if rlrop:
            check_rlrop = True
            rlrop = rlrop[0]
        else:
            check_rlrop = False
            rlrop = None

        # ----------------------------------------------------------------------------------------------------------

        e = 0
        stop_training = False
        history = None

        EPOCHS = epochs
        SUBEPOCHS_shared = epochs_shared
        SUBEPOCHS_taskspecific = epochs_taskspecific

        while e < EPOCHS and not stop_training:

            if verbose_alternate:
                print(f'******** TRAINING TRUNK - STARTING EPOCH: {e + 1}/{EPOCHS} ********')

            for l in self.layers:
                if l.name.startswith(self._prefix_layername_shared):
                    l.trainable = True
                else:
                    l.trainable = False

            training_configs['initial_epoch'] = e
            training_configs['epochs'] = e + SUBEPOCHS_shared
            history_trunk = self.fit(**training_configs)

            if history is None:
                history = history_trunk.history.copy()
            else:
                history = {k: history[k] + history_trunk.history[k] for k in history.keys()}

            e += len(history_trunk.history['loss'])

            if check_tnan:
                stop_training = stop_training or tnan.model.stop_training

            if check_es:
                stop_training = stop_training or es.model.stop_training
                es.starting_wait = es.wait
                es.starting_best = es.best
                es.starting_best_weights = es.best_weights
                es.starting_best_epoch = es.best_epoch

            if stop_training:
                break

            if check_rlrop:
                rlrop.starting_wait = rlrop.wait
                rlrop.starting_cooldown_counter = rlrop.cooldown_counter
                rlrop.starting_best = rlrop.best

            if verbose_alternate:
                print(f'******** TRAINING BRANCHES - STARTING EPOCH: {e + 1}/{EPOCHS} ********')

            for l in self.layers:
                if not l.name.startswith(self._prefix_layername_shared):
                    l.trainable = True
                else:
                    l.trainable = False

            training_configs['initial_epoch'] = e
            training_configs['epochs'] = e + SUBEPOCHS_taskspecific
            history_branch = self.fit(**training_configs)

            history = {k: history[k] + history_branch.history[k] for k in history.keys()}

            e += len(history_branch.history['loss'])

            if check_tnan:
                stop_training = stop_training or tnan.model.stop_training

            if check_es:
                stop_training = stop_training or es.model.stop_training
                es.starting_wait = es.wait
                es.starting_best = es.best
                es.starting_best_weights = es.best_weights
                es.starting_best_epoch = es.best_epoch

            if check_rlrop:
                rlrop.starting_wait = rlrop.wait
                rlrop.starting_cooldown_counter = rlrop.cooldown_counter
                rlrop.starting_best = rlrop.best

        if check_tnan and check_es:
            if tnan.model.stop_training and es.restore_best_weights:
                self.set_weights(es.best_weights)

        return history

