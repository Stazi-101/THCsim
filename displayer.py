
import numpy as np



# This is a class since it may be intended to hold some display data
class Displayer():

    def __init__(self, config, scalar_fields, vector_fields):

        self.scalar_fields = scalar_fields
        self.vector_fields = vector_fields

        Ts = self.scalar_fields['T']
        Ss = self.scalar_fields['S']
        vs = self.vector_fields['v']
        
        vs = np.swapaxes(vs, 0, 1)
        yssc = np.expand_dims(Ts, 3)
        yssc = np.repeat(yssc, 3, axis=3)
        #yssc = yssc*0
        yssc[:,:,:,1] = Ss
        yssc[:,:,:,2] = np.linalg.norm(vs, axis=0)

        self.data = yssc
        
    @classmethod
    def fromSimOutput(cls, config, yss):
        Ts, Ss, vs = [np.clip(np.array(ys), -100, 100) for ys in yss]

        scalar_fields = {'T': Ts,
                         'S': Ss}
        
        vector_fields = {'v': vs}
        
        return cls(config, scalar_fields, vector_fields)


    # Draw array shaped (t, lat, long)
    def slice(self, x=0, y=0):

        return self.data[x]
        







if __name__ == '__main__':
    dis = Displayer()
    dis.draw_chunk_2d(None, np.zeros((10,10)))