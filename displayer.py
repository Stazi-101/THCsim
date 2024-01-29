
import numpy as np
import pyqtgraph as pqg

class Displayer():

    def __init__(self):
        pass


    def draw_chunk_2d(self, config, ys):

        pqg.image(np.array(ys))

        input()


    def draw_chunk_3d(self, config, yss):

        pqg.image(np.swapaxes(np.array(yss), 1, 2))

        input()