import tensorflow as tf
import tensorflow.keras.backend as K


def binary_accuracy_from_logits(y_true, y_pred, threshold=0.5):
    y_pred = tf.convert_to_tensor(tf.keras.activations.sigmoid(y_pred))
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)

    return K.mean(tf.equal(y_true, y_pred), axis=-1)


class BinaryAccuracyFromLogits(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, name='binary_accuracy_from_logits', dtype=None, threshold=0.5):
        super(tf.keras.metrics.BinaryAccuracy, self).__init__(binary_accuracy_from_logits,
                                                              name,
                                                              dtype=dtype,
                                                              threshold=threshold
                                                              )


