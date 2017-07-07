'''
Train PredNet on Gridworld simulator
xiangyu.liu@hobot.cc
'''

import os
import numpy as np

SEED1 = 123
SEED2 = 1234

from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input, Dense, Flatten
from tensorflow.contrib.keras.python.keras.layers import TimeDistributed
from tensorflow.contrib.keras.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.contrib.keras.python.keras.callbacks import TensorBoard

from prednet import PredNet
from data_generator import SequenceGenerator
from settings import *


save_model = True  # if weights will be saved
T_board = True  # use Tensorboard


# Training parameters
USE_ROI_LOSS = False    # use roi (add loss weights in obstacle positions)
THRESHOLD = 0.5    # # re-threshold generated images
INTERVAL = 2    # sampling frequency in simulator
nb_epoch = 100
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Save model files
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_weights_Interval_' + str(INTERVAL) + '_USE_ROI_' + str(USE_ROI_LOSS) + '_thres_' + str(THRESHOLD) + '.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_model_Interval_' + str(INTERVAL) + '_USE_ROI_' + str(USE_ROI_LOSS) + '_thres_' + str(THRESHOLD) + '.json')

# Model parameters
nt = 10  # time step
n_channels, im_height, im_width = (1, MAP_SIZE, MAP_SIZE)
input_shape = (im_height, im_width, n_channels)
stack_sizes = (n_channels, 32, 64, 128, 256)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)  # layer weight (only assigned to ground-truth)
time_loss_weights = 1. / (nt - 1) * np.ones((nt, 1))    # temporal average weight
time_loss_weights[0] = 0    # step 0 with 0 loss

# Construct the model
prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True, unroll=True, use_roi_loss=USE_ROI_LOSS, threshold=THRESHOLD)
inputs = Input(shape=(nt,) + input_shape)   # (?, nt, im_size, im_size, channel)
errors = prednet(inputs)    # (?, nt, nb_layers)

# calculate weighted error by layer
errors_by_time = TimeDistributed(
    Dense(1, weights=[layer_loss_weights, np.zeros(1)],
          trainable=False), trainable=False)(errors)    # (?, nt, 1)
errors_by_time = Flatten()(errors_by_time)  # (?, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # (?, 1)
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()
train_generator = SequenceGenerator(nt, interval=INTERVAL, data_size=100, batch_size=batch_size, shuffle=True, seed=SEED1)
val_generator = SequenceGenerator(nt, interval=INTERVAL, data_size=100, batch_size=batch_size, N_seq=N_seq_val, seed=SEED2)

# start with lr of 0.001 and then drop to 0.0001 after 75 epochs
lr_schedule = lambda epoch: 0.001 if epoch < nb_epoch / 2 else 0.0001
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

if T_board:
    if not os.path.exists(TBOARD_DIR):
        os.mkdir(TBOARD_DIR)
    callbacks.append(TensorBoard(log_dir=TBOARD_DIR, histogram_freq=10, write_graph=True, write_images=True))

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

history = model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks,
                    validation_data=val_generator, validation_steps=N_seq_val)
