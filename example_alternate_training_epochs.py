import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from alternate_training_through_the_epochs.models import AlternateTrainingEpochsModel as ATEModel

rs = 2024
np.random.seed(rs)
tf.random.set_seed(rs)
tf.keras.utils.set_random_seed(rs)

Xtrain = -1 + 2 * np.random.rand(750, 2)
Xval = -1 + 2 * np.random.rand(150, 2)
Xtest = -1 + 2 * np.random.rand(1000, 2)

# TASK1: INSIDE/OUTSIDE CIRCLE OF RADIUS 0.5
Ytrain_task1 = (np.linalg.norm(Xtrain, axis=1) <= 0.5).astype('int32')
Yval_task1 = (np.linalg.norm(Xval, axis=1) <= 0.5).astype('int32')
Ytest_task1 = (np.linalg.norm(Xtest, axis=1) <= 0.5).astype('int32')

# TASK2: POINTS WITH l1-NORM LESS THAN OR EQUAL TO 1
Ytrain_task2 = (np.linalg.norm(Xtrain, ord=1, axis=1) <= 1).astype('int32')
Yval_task2 = (np.linalg.norm(Xval, ord=1, axis=1) <= 1).astype('int32')
Ytest_task2 = (np.linalg.norm(Xtest, ord=1, axis=1) <= 1).astype('int32')

# --------------- BUILDING THE MODEL -------------------

# TRUNK
I = tf.keras.layers.Input(shape=(2,), name='trunk_00_input')
Dt_01 = tf.keras.layers.Dense(64, activation='relu', name='trunk_01')(I)
Dt_02 = tf.keras.layers.Dense(64, activation='relu', name='trunk_02')(Dt_01)
Dt_03 = tf.keras.layers.Dense(64, activation='relu', name='trunk_03')(Dt_02)
Dt_last = tf.keras.layers.Dense(64, activation='relu', name='trunk_last')(Dt_03)
# BRANCH TASK 1
Db1_01 = tf.keras.layers.Dense(64, activation='relu', name='branch_t1_01')(Dt_last)
Db1_02 = tf.keras.layers.Dense(64, activation='relu', name='branch_t1_02')(Db1_01)
Db1_out = tf.keras.layers.Dense(1, activation='sigmoid', name='branch_t1_out')(Db1_02)
# BRANCH TASK 2
Db2_01 = tf.keras.layers.Dense(64, activation='relu', name='branch_t2_01')(Dt_last)
Db2_02 = tf.keras.layers.Dense(64, activation='relu', name='branch_t2_02')(Db2_01)
Db2_out = tf.keras.layers.Dense(1, activation='sigmoid', name='branch_t2_out')(Db2_02)

model = ATEModel(inputs=[I], outputs=[Db1_out, Db2_out])

# ------------------------------------------------------


# --------------- TRAINING THE MODEL WITH ATE-SGD PROCEDURE -------------------

verbose_alternate = True

EPOCHS = 500
E0 = 1
Ets = 1

losses = {
    'branch_t1_out': 'binary_crossentropy',
    'branch_t2_out': 'binary_crossentropy'
}

metrics = {
    'branch_t1_out': ['binary_accuracy'],
    'branch_t2_out': ['binary_accuracy']
}

model.compile(
    optimizer='sgd',
    loss=losses,
    metrics=metrics
)

history = model.ate_fit(
    x=Xtrain,
    y=(Ytrain_task1, Ytrain_task2),
    validation_data=(Xval, (Yval_task1, Yval_task2)),
    batch_size=16,
    epochs=EPOCHS,
    epochs_shared=E0,
    epochs_taskspecific=Ets,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            verbose=True,
            restore_best_weights=True
        )
    ],
    verbose=True,
    verbose_alternate=verbose_alternate
)

Ypred = model.predict(Xtest)
test_evals = model.evaluate(Xtest, (Ytest_task1, Ytest_task2))
acc_task1 = np.round(test_evals[-2] * 100, decimals=2)
acc_task2 = np.round(test_evals[-1] * 100, decimals=2)

# ----------------------------------------------------------------------------


# --------------- VISUALIZING THE RESULTS ON THE TEST SET --------------------

Xplain, Yplain = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 100))
Z1 = np.sqrt(Xplain ** 2 + Yplain ** 2) <= 0.5
Z2 = (np.abs(Xplain) + np.abs(Yplain)) <= 1

pred_colors = {
    0: 'purple',
    1: 'orange'
}

fig, ax = plt.subplots(1, 2)
ax[0].contourf(Xplain, Yplain, Z1, alpha=0.15)
ax[0].scatter(Xtest[:, 0], Xtest[:, 1], c=[pred_colors[int(y)] for y in np.round(Ypred[0])],
              alpha=1e-2 + np.abs(0.5 - Ypred[0]))
ax[0].set_title(f'Task 1 - {acc_task1}% Accuracy')
ax[1].contourf(Xplain, Yplain, Z2, alpha=0.15)
ax[1].scatter(Xtest[:, 0], Xtest[:, 1], c=[pred_colors[int(y)] for y in np.round(Ypred[1])],
              alpha=1e-2 + np.abs(0.5 - Ypred[1]))
ax[1].set_title(f'Task 2 - {acc_task2}% Accuracy')
plt.suptitle('Test Set Predictions')

plt.show()

# -----------------------------------------------------------------------------


