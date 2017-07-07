'''
Data generator for Gridworld Simulator.

xiangyu.liu@hobot.cc
'''

import numpy as np
from settings import MAP_SIZE, MAX_PIXEL_VALUE
from simulator.Env.Env import Env
import os
import pygame as pg

from tensorflow.contrib.keras.python.keras.preprocessing.image import Iterator


# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):  # inherits keras.preprocessing.image.Iterator
    def __init__(self, nt, data_size, interval=0, dimension=1, batch_size=8, shuffle=False, seed=None, output_mode='error', N_seq=None):
        # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        # For pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pg.init()
        _ = pg.display.set_mode((MAP_SIZE, MAP_SIZE), 0, 8)

        self.data_size = data_size
        self.interval = interval
        self.map_list = []
        if seed:
            np.random.seed(seed)
        INPUT_DICT = {'ag': 0, 'ob': 0, 'ch': 1}
        for i in range(self.data_size):
            INPUT_DICT['ob'] = np.random.randint(3, 6)
            self.map_list.append(Env.random_scene(MAP_SIZE, INPUT_DICT))
        self.env = Env(MAP_SIZE, False, False)
        self.nt = nt
        assert dimension > 0, 'dimension must >= 1.'
        self.dimension = dimension
        self.batch_size = batch_size
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode
        self.im_shape = (self.env.map_size, self.env.map_size, self.dimension)

        if shuffle:
            self.map_list = np.random.permutation(self.map_list)

        if N_seq is not None and len(self.map_list) > N_seq:
            self.map_list = self.map_list[:N_seq]
        self.N_sequences = len(self.map_list)
        super(SequenceGenerator, self).__init__(len(self.map_list), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            _ = self.env.reset(self.map_list[idx], static=False, )
            for j in range(self.nt):
                for k in range(self.interval):
                    self.env.step([])
                batch_x[i][j] = self.preprocess(self.env.step([])[0])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return np.expand_dims(X, axis=-1).astype(np.float32) / MAX_PIXEL_VALUE

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, map_i in enumerate(self.map_list):
            _ = self.env.reset(map_i, static=False, )
            for j in range(self.nt):
                for k in range(self.interval):
                    self.env.step([])
                X_all[i][j] = self.preprocess(self.env.step([]))
        return X_all
