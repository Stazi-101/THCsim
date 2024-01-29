
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



    def draw_chunk_2d(self, config, ys):

        c = config
        # Load some config
        lat_first = c['spatial_discretisation']['lat_first']
        lat_final = c['spatial_discretisation']['lat_final']
        lng_first = c['spatial_discretisation']['lng_first']
        lng_final = c['spatial_discretisation']['lng_final']
        
        plt.figure(figsize=(5, 5))
        #breakpoint()

        plt.imshow(
            ys,
            origin="lower",
            extent=(lng_first, lng_final, lat_first, lat_final),
            #aspect=(lng_final - lng_first) / (lat_final - lat_first),
            cmap="inferno",
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude", rotation=0)
        plt.clim(0, 1)
        plt.colorbar()
        plt.show()


    def draw_chunk_3d(self, config, yss):

        c = config
        # Load some config
        lat_first = c['spatial_discretisation']['lat_first']
        lat_final = c['spatial_discretisation']['lat_final']
        lng_first = c['spatial_discretisation']['lng_first']
        lng_final = c['spatial_discretisation']['lng_final']
        t_first = c['temporal_discretisation_finite']['t_first']
        t_final = c['temporal_discretisation_finite']['t_first']

        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left = 0.25, bottom=0.25)
        #breakpoint()

        im1 = ax.imshow(
            25*yss[0],
            origin="lower",
            #extent=(lng_first, lng_final, lat_first, lat_final),
            #aspect=(lng_final - lng_first) / (lat_final - lat_first),
            cmap="inferno",
        )
        fig.colorbar(im1)
        axt = fig.add_axes([.25, .2, .65, .03])
        slider_t = Slider(axt, 'Time', 0, 10, valinit=0)

        def update(val):
            im1.set_data(25*yss[int(slider_t.val)])
            fig.canvas.draw()

        slider_t.on_changed(update)

        
        #plt.xlabel("Longitude")
        #plt.ylabel("Latitude", rotation=0)
        #plt.clim(0, 1)
        #plt.colorbar()
        plt.show()