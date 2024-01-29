
import numpy as np
import pyqtgraph as pqg

# This is a class since it may be intended to hold some display data
class Displayer():

    def __init__(self):
        pass

    # Draw array shaped (lat,long)
    def draw_chunk_2d(self, config, ys):

        pqg.image(np.swapaxes(np.array(ys), 0, 1))

        input()

    # Draw array shaped (t, lat, long)
    def draw_chunk_3d(self, config, yss):

        pqg.image(np.swapaxes(np.array(yss), 1, 2))

        input()