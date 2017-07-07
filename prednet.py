'''
https://github.com/coxlab/prednet/blob/master/prednet.py
'''

import numpy as np


from tensorflow.contrib.keras.python.keras.layers import Recurrent
from tensorflow.contrib.keras.python.keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import activations

import tensorflow as tf


class PredNet(Recurrent):
    '''PredNet architecture 
        Stacked convolutional LSTM inspired by predictive coding principles.

    # Arguments
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
        A_filt_sizes: filter sizes for the target (A) modules.
            Has length of 1 - len(stack_sizes).
            Ex. (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of the errors (E)
                from the layer below (followed by max-pooling)
        Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        A_activation: activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
                Nomenclature of 'all' is kept for backwards compatibility, but should not be confused with returning
                all of the layers of the model
            For returning the features of a particular layer, output_mode should be of the form unit_type + layer_number.
                For instance, to return the features of the LSTM "representational" units in the lowest layer,
                output_mode should be specificied as 'R0'.
                The possible unit types are 'R', 'Ahat', 'A', and 'E' corresponding to the 'representation',
                'prediction', 'target', and 'error' units respectively.
        extrap_start_time: time step for which model will start extrapolating.
            Starting at this time step, the prediction from the previous time step will be treated as the "actual"
        use_roi_loss: add loss on region of interests(domain specific)
        threshold: manually threshold
    '''
    def __init__(self, stack_sizes, R_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time=None, use_roi_loss=False, threshold=None, **kwargs):
        self.stack_sizes = stack_sizes  # output_dim for each layer
        self.nb_layers = len(stack_sizes)  # layer num
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes  # R state dim
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes  # A2E filter
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes  # Ahat2E filter
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes  # R_l+1, El 2 R_l filter

        self.extrap_start_time = extrap_start_time
        self.use_roi_loss = use_roi_loss
        self.threshold = threshold

        self.pixel_max = pixel_max
        self.error_activation = activations.get(error_activation)
        self.A_activation = activations.get(A_activation)
        self.LSTM_activation = activations.get(LSTM_activation)
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation)

        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None

        # self.dim_ordering = 'tf'
        self.channel_axis = -1
        self.row_axis = -3
        self.column_axis = -2

        super(PredNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5)]  # (batch_num, time_step, row, column, channel)

    def get_output_shape_for(self, input_shape):  # determine output shape
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.nb_layers,)
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_layers, )
        else:
            stack_str = 'R_stack_sizes' if self.output_layer_type == 'R' else 'stack_sizes'
            stack_mult = 2 if self.output_layer_type == 'E' else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            out_nb_row = input_shape[self.row_axis] / 2**self.output_layer_num  # upsampling or downsampling
            out_nb_col = input_shape[self.column_axis] / 2**self.output_layer_num
            out_shape = (-1, out_nb_row, out_nb_col, out_stack_size)

        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    def get_initial_states(self, x):
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]

        base_initial_state = K.zeros_like(x)  # (batch_samples, timesteps) + image_shape
        non_channel_axis = -2
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = K.sum(base_initial_state, axis=1)  # (samples, nb_channels)

        initial_states = []
        states_to_pass = ['r', 'c', 'e']
        nlayers_to_pass = {u: self.nb_layers for u in states_to_pass}
        if self.extrap_start_time is not None:
            # pass prediction in states so can use as actual for t+1 when extrapolating
            states_to_pass.append('ahat')
            nlayers_to_pass['ahat'] = 1
        for u in states_to_pass:    # ['r', 'c', 'e'] is the state
            for l in range(nlayers_to_pass[u]): # initialize all the state with zero
                ds_factor = 2 ** l  # why downsampling?
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = self.R_stack_sizes[l]
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]
                elif u == 'ahat':
                    stack_size = self.stack_sizes[l]
                output_size = nb_row * nb_col * stack_size  # flattened size

                reducer = K.zeros((input_shape[self.channel_axis], output_size))    # (nb_channels, output_size)
                initial_state = K.dot(base_initial_state, reducer)  # (samples, output_size)
                output_shp = [-1, nb_row, nb_col, stack_size]
                initial_state = K.reshape(initial_state, output_shp)
                initial_states += [initial_state]

        if self.extrap_start_time is not None:
            initial_states += [K.variable(0, 'int32')]  # the last state will correspond to the current timestep
        return initial_states

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'a', 'ahat']}

        for l in range(self.nb_layers):
            for c in ['i', 'f', 'c', 'o']:
                act = self.LSTM_activation if c == 'c' else self.LSTM_inner_activation
                self.conv_layers[c].append(
                    Convolution2D(self.R_stack_sizes[l],
                                  self.R_filt_sizes[l],
                                  padding='same',
                                  data_format="channels_last", activation=act))
            act = 'relu' if l == 0 else self.A_activation
            self.conv_layers['ahat'].append(Convolution2D(self.stack_sizes[l],
                                                          self.Ahat_filt_sizes[l],
                                                          padding='same',
                                                          data_format="channels_last",
                                                          activation=act))
            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(
                    Convolution2D(self.stack_sizes[l+1],
                                  self.A_filt_sizes[l],
                                  padding='same',
                                  data_format="channels_last",
                                  activation=self.A_activation))

        self.upsample = UpSampling2D(data_format="channels_last")    # upsampling
        self.pool = MaxPooling2D(data_format="channels_last")    # downsampling

        self._trainable_weights = []
        nb_row, nb_col = (input_shape[-3], input_shape[-2])
        # Super model
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                elif c == 'a':
                    nb_channels = 2 * self.stack_sizes[l]
                else:  # i, c, o, f
                    nb_channels = self.stack_sizes[l] * 2 + self.R_stack_sizes[l]
                    if l < self.nb_layers - 1:
                        nb_channels += self.R_stack_sizes[l+1]
                in_shape = (input_shape[0], nb_row // ds_factor, nb_col // ds_factor, nb_channels)  # up -> downsampling
                self.conv_layers[c][l].build(in_shape)
                self._trainable_weights += self.conv_layers[c][l].trainable_weights

        self.states = [None] * self.nb_layers*3 # ['r', 'c', 'e']
        if self.extrap_start_time is not None:
            self.t_extrap = K.variable(np.array(self.extrap_start_time), 'int32')
            self.states += [None] * 2

    def step(self, a, states):
        """

        :param a: ground-truth
        :param states:
         type: list
         index[:-2]: r, c, e (#: self.nb_layers)
         index[-2:] (if self.extrap_start_time is not None:): [frame_prediction, t+1]
        :return:
        """
        r_tm1 = states[:self.nb_layers]
        c_tm1 = states[self.nb_layers:2*self.nb_layers]
        e_tm1 = states[2*self.nb_layers:3*self.nb_layers]


        if self.extrap_start_time is not None:
            t = states[-1]
            # if past self.extrap_start_time, the previous prediction will be treated as the actual
            a = K.switch(t >= self.t_extrap, states[-2], a)
        c = []
        r = []
        e = []
        for l in reversed(range(self.nb_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_layers - 1:
                inputs.append(r_up)
            inputs = K.concatenate(inputs, axis=self.channel_axis)
            # print l, inputs.shape
            i = self.conv_layers['i'][l].call(inputs)
            f = self.conv_layers['f'][l].call(inputs)
            o = self.conv_layers['o'][l].call(inputs)
            _c = f * c_tm1[l] + i * self.conv_layers['c'][l].call(inputs)
            _r = o * self.LSTM_activation(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)   # upsampling

        for l in range(self.nb_layers):
            ahat = self.conv_layers['ahat'][l].call(r[l])
            if l == 0:
                ahat = K.minimum(ahat, self.pixel_max)
                frame_prediction = ahat
                ### threshold
                where = K.greater_equal(frame_prediction, K.constant(self.threshold))
                frame_prediction = tf.where(where, 0.5 * tf.ones_like(frame_prediction, dtype=tf.float32),
                                            tf.zeros_like(frame_prediction, dtype=tf.float32))
                ###
            # compute errors
            e_up = ahat - a
            e_down = a - ahat

            # ROI loss
            if l == 0 and self.use_roi_loss:
                e_up = tf.add(e_up, tf.multiply(e_up, a, name='multiply_up_err'), name='add_up_err')
                e_down = tf.add(e_down, tf.multiply(e_down, a, name='multiply_down_err'), name='add_down_err')
            #

            e_up = self.error_activation(e_up)
            e_down = self.error_activation(e_down)

            e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))

            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'R':
                    output = r[l]
                elif self.output_layer_type == 'E':
                    output = e[l]

            if l < self.nb_layers - 1:
                a = self.conv_layers['a'][l].call(e[l])
                a = self.pool.call(a)  # target for next layer (downsampling)

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction

            else:
                for l in range(self.nb_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    # TODO: where is all_error ?
                    all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                    # print l, e[l].shape, layer_error.shape, all_error.shape
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)
                # print output.shape

        states = r + c + e
        if self.extrap_start_time is not None:
            ###
            '''
            sess = tf.get_default_session()
            comparison = tf.greater_equal(frame_prediction, tf.constant(0.3))
            sess.run(comparison)
            conditional_op = tf.assign(frame_prediction, tf.where(comparison, 0.5 * tf.ones_like(frame_prediction), tf.zeros_like(frame_prediction)))
            sess.run(conditional_op)
            '''
            ###
            states += [frame_prediction, t + 1]
        return output, states

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'A_filt_sizes': self.A_filt_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'A_activation': self.A_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode,
                  'use_roi_loss': self.use_roi_loss,
                  'threshold': self.threshold}
        base_config = super(PredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
