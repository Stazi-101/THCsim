
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class Displayer():

    def __init__(self):
        pass

    def draw_chunk(self, config, ys):

        c = config
        # Load some config
        t_first = c['temporal_discretisation_finite']['t_first']
        t_final = c['temporal_discretisation_finite']['t_final']
        x_first = c['spatial_discretisation']['x_first']
        x_final = c['spatial_discretisation']['x_final']
        
        plt.figure(figsize=(5, 5))
        plt.imshow(
            ys,
            origin="lower",
            extent=(x_first, x_final, t_first, t_final),
            aspect=(x_final - x_first) / (t_final - t_first),
            cmap="inferno",
        )
        plt.xlabel("x")
        plt.ylabel("t", rotation=0)
        plt.clim(0, 1)
        plt.colorbar()
        plt.show()



    def draw_chunk_3d(self, config, ys):

        c = config
        # Load some config
        t_first = c['temporal_discretisation_finite']['t_first']
        t_final = c['temporal_discretisation_finite']['t_final']
        x_first = c['spatial_discretisation']['x_first']
        x_final = c['spatial_discretisation']['x_final']
        
        plt.figure(figsize=(5, 5))

        slider_t = Slider()
        plt.imshow(
            ys,
            origin="lower",
            extent=(x_first, x_final, t_first, t_final),
            aspect=(x_final - x_first) / (t_final - t_first),
            cmap="inferno",
        )
        plt.xlabel("x")
        plt.ylabel("t", rotation=0)
        plt.clim(0, 1)
        plt.colorbar()
        plt.show()
