
import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Checkbox
import pickle as pkl


# This is a class since it may be intended to hold some display data
class Displayer():

    def __init__(self, config, args, scalar_fields, vector_fields, max_val=100):

        self.config=config
        self.args = args

        self.scalar_fields = scalar_fields
        self.vector_fields = vector_fields

        self.t_n = self.vector_fields['v'].shape[0]

        self.ba = self.args['boundary_aware_functions']



        

    @classmethod
    def fromPath(cls, config, path, max_val = 10):
        with open(path, 'rb') as file:
            yss, args = pkl.load(file)
            print(args.keys())
        return cls.fromSimOutput(config, yss, args, max_val=10)
        
    @classmethod
    def fromSimOutput(cls, config, yss, args, max_val=10):
        Ts, Ss, vs = [np.clip(np.array(ys), -100, 100) for ys in yss]

        scalar_fields = {'T': Ts,
                         'S': Ss}
        
        vector_fields = {'v': vs}
        
        return cls(config, args, scalar_fields, vector_fields, max_val=10)

    def get_yss(self):
        return self.data

    # Draw array shaped (t, lat, long)
    def slice(self, x=0, y=0, 
              c1=False, c2=False, c3=False, c4=False, c5=False, c6=False):

                        
        T = self.scalar_fields['T'][x]
        S = self.scalar_fields['S'][x]
        v = self.vector_fields['v'][x]

        r, g, b = [np.zeros(T.shape) for _ in range(3)]

        v = v/y

        if c1:
            r += v[0] + 0.5
        if c2:
            g += v[1] + 0.5
        if c3:
            print(np.sum(np.isnan(v)))
            b += np.linalg.norm(v, axis=0)
        if c4:
            r += T
        if c5:
            g += S
        if c6: 
            b += self.args['state']


        return np.stack((r,g,b), axis=2)
    
    def imshow(self):

        def plot_2d_slice(x=0, y=0, 
              c1=False, c2=False, c3=False, c4=False, c5=False, c6=False):
        # Get the correct slice of data wrt state
                     
            plt.imshow(self.slice(x=x, y=y, 
              c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6), 
              interpolation='nearest')
            plt.xticks([])
            plt.yticks([])

        # `interact` function call to create two sliders
        return interact(plot_2d_slice,
        x=IntSlider(min=0, max=self.t_n-1, step=1, value=0, description='X Dimension', continuous_update=True),
        y=IntSlider(min=2, max=100, step=2, value=16, description='Y Dimension'),
        c1 = Checkbox(value = False, description = 'v0'),
        c2 = Checkbox(value = False, description = 'v1'),
        c3 = Checkbox(value = True, description = 'vnorm'),
        c4 = Checkbox(value = True, description = 'T'),
        c5 = Checkbox(value = True, description = 'S'),
        c6 = Checkbox(value = False, description = 'state'),

        )


        







if __name__ == '__main__':

    do_pyqt = False
    if do_pyqt:
        import pickle as pkl
        import pyqtgraph as pqg
        import numpy as np
        with open('output/test3.npy', 'rb') as file:
            ddd = pkl.load(file)
        

        dis = Displayer.fromSimOutput({}, ddd)
        pqg.image(np.swapaxes(dis.get_yss(), 1, 2))
        input()


