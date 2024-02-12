
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
        Ts, Ss, vs = [np.array(ys) for ys in yss]

        vs = np.swapaxes(vs, 0, 1)
        yssc = np.expand_dims(Ts, 3)
        yssc = np.repeat(yssc, 3, axis=3)
        #yssc = yssc*0
        yssc[:,:,:,1] = Ss
        yssc[:,:,:,2] = np.linalg.norm(vs, axis=0)

        pqg.image(np.swapaxes(np.array(yssc), 1, 2))

        print(np.amax(vs))

        input()
        #breakpoint()