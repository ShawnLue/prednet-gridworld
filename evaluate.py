'''
Evaluate trained PredNet on simulator sequences.
Plots predictions.

xiangyu.liu@hobot.cc
'''

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.contrib.keras.python.keras.models import Model, model_from_json
from tensorflow.contrib.keras.python.keras.layers import Input
from prednet import PredNet
from data_generator import SequenceGenerator
from settings import *


# hyper-parameters for evaluation
INTERVAL = 2    # sampling frequency in simulator
THRESHOLD = 0.5 # re-threshold generated images
USE_ROI_LOSS = False    # use roi (large loss weights for obstacle positions)
batch_size = 5
nt = 10
dim = 1
EXTRAP = None   #

# Read the model files
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_weights_Interval_' + str(INTERVAL) + '_USE_ROI_' + str(USE_ROI_LOSS) + '_thres_' + str(THRESHOLD) + '.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_model_Interval_' + str(INTERVAL) + '_USE_ROI_' + str(USE_ROI_LOSS) + '_thres_' + str(THRESHOLD) + '.json')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = u'prediction'
layer_config['extrap_start_time'] = EXTRAP

test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)

input_shape = list(train_model.layers[0].batch_input_shape[1:])
del train_model
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(nt, interval=INTERVAL, dimension=dim, data_size=10, batch_size=batch_size, N_seq=10)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)


# Plot some predictions

aspect_ratio = float(X_hat.shape[2] / X_hat.shape[3])
plt.figure(figsize=(nt, 2 * aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0.1, hspace=0.1)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots_interval' +
                             str(INTERVAL) + '_EXTRAP' + str(EXTRAP) + 'USE_POI_' + str(USE_ROI_LOSS) + '/')
if not os.path.exists(plot_save_dir):
    os.mkdir(plot_save_dir)
for i in range(X_hat.shape[0]):
    for t in range(nt):
        plt.subplot(gs[t])
        # plt.imshow(X_test[i, t], interpolation='none', cmap=plt.get_cmap('gray'))
        plt.imshow(X_test[i, t].squeeze(), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off',
                        right='off', labelbottom='off', labelleft='off')
        if t == 0:
            plt.ylabel('Actual', fontsize=10)
        plt.subplot(gs[t + nt])
        # plt.imshow(X_hat[i, t], interpolation='none', cmap=plt.get_cmap('gray'))
        plt.imshow(X_hat[i, t].squeeze(), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off',
                        right='off', labelbottom='off', labelleft='off')
        if t == 0:
            plt.ylabel('Predicted', fontsize=10)
    plt.savefig(plot_save_dir + 'plot_' + str(i) + '.png')
    plt.clf()
