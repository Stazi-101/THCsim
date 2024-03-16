import jax.numpy as jnp
import jax.scipy as jsp

CG_REPEATS = 2
DT = 0.01
MAXITER = 100

DTHETA = 1
DLAMBDA = 1

# Vector fields: list of ys -> list of dy/dts

def vf_flow_basic(t, ys, args):
    T, S, v = ys
    config = args['config']
    ba = args['boundary_aware_functions']

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * ba.laplacian(T) + ba.advection(v, T)
    dS = A_HH * ba.laplacian(S) + ba.advection(v, S)

    dv = A_MH * ba.laplacian(v) 
    dv = dv.at[0].add( -1/rho_0 * ba.ptheta(p)  + ba.advection(v,v[0]))
    dv = dv.at[1].add( -1/rho_0 * ba.plambda(p) + ba.advection(v,v[1]))

    return (dT, dS, dv)


def vf_flow_incompressible(t, ys, args):
    T, S, v = ys

    # If simulation has diverged, don't bother doing any more maths
    # (but do not crash so we can analyse the output)
    # doesnt work with jit
    #if jnp.any(jnp.isnan(v)):
    #    return T*0, S*0, v*0

    config = args['config']
    ba = args['boundary_aware_functions']

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * ba.laplacian(T) + ba.advection(v, T)
    dS = A_HH * ba.laplacian(S) + ba.advection(v, S)

    dv = A_MH * ba.laplacian(v)
    dv = dv.at[0].add( ba.advection(v,v[0]) )  # -1/rho_0 * ba.ptheta(p) 
    dv = dv.at[1].add( ba.advection(v,v[1]) )  # -1/rho_0 * ba.plambda(p)

    state = args['state']
    dv *= state

    dv = ba.project_divergencefree(dv)
    dv *= (ba.neighbours==4)
    dT *= state
    dS *= state

    return (dT, dS, dv)


def vf_flow_semicompressible(t, ys, args):
    T, S, v = ys
    config = args['config']
    ba = args['boundary_aware_functions']

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * ba.laplacian(T) + ba.advection(v, T)
    dS = A_HH * ba.laplacian(S) + ba.advection(v, S)

    dv = A_MH * ba.laplacian(v)
    dv = dv.at[0].add( ba.advection(v,v[0]) )
    dv = dv.at[1].add( ba.advection(v,v[1]) )

    dv = ba.project_divergencefree(dv)

    dv = dv.at[0].add( -1/rho_0 * ba.ptheta(p) )
    dv = dv.at[1].add( -1/rho_0 * ba.plambda(p) )

    return (dT, dS, dv)



# Initial conditions: coordinates -> list of ys
def ic_flow_basic(config, lat,lng):
    
    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7) 
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)

    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add a spot of vertical movement
    v = v.at[1].set( -10 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )
    # Force v to be divergence free

    state = 1 - 1 * ((jnp.square(lat+0.5) + jnp.square(lng+2))<0.7)
    ba = BoundaryAware(state)

    T*=state; S*=state; v*=state

    v = ba.project_divergencefree(v)

    return (T, S, v), {'state': state, 'boundary_aware_functions': ba }


def ic_flow_basic_noboundary(config, lat, lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7) 
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)

    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add a spot of vertical movement
    v = v.at[1].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )
    
    state = jnp.zeros((lat.shape[0], lat.shape[1]))
    ba = BoundaryAware(state)

    # Force v to be divergence free
    v = ba.project_divergencefree(v)

    return (T, S, v), {'state': state, 'boundary_aware_functions': ba}

def ic_flow_ts_only(config, lat, lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)
    # Initialise v is 0. Naturally divergence free :D
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))

    state = jnp.zeros((lat.shape[0], lat.shape[1]))
    ba = BoundaryAware(state)

    return (T, S, v), {'state': state, 'boundary_aware_functions': ba}

def ic_flow_funky(config, lat, lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)

    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add circles of differing velocities
    v = v.at[1].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )
    v = v.at[0].set( 1 * ((jnp.square(lat+0.8) + jnp.square(lng+0.2))<.1) )
    v = v.at[0].set(-1 * ((jnp.square(lat+0.3) + jnp.square(lng+0.8))<.1) )

    state = jnp.zeros((lat.shape[0], lat.shape[1]))
    ba = BoundaryAware(state)

    return (T, S, v), {'state': state, 'boundary_aware_functions': ba}

def ic_flow_vt_only(config, lat, lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.2)
    # Initialise salinity is 0
    S = 0 * lat
    
    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add circles of differing velocities
    v = v.at[0].set( 1 * ((jnp.square(lat) + jnp.square(lng))<0.8) )
    v = v.at[1].set( -1 * ((jnp.square(lat-0.5) + jnp.square(lng-0.5))<0.5) )

    state = jnp.zeros((lat.shape[0], lat.shape[1]))
    ba = BoundaryAware(state)

    return (T, S, v), {'state': state, 'boundary_aware_functions': ba}

def ic_flow_v_only(config, lat, lng):

    # Initialise T and S are 0
    T = 0 * lat
    S = 0 * lat
    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add moving spot
    v = v.at[0].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<0.5) )

    state = jnp.zeros((lat.shape[0], lat.shape[1]))
    ba = BoundaryAware(state)


    v = ba.project_divergencefree(v, state=state)

    return (T, S, v), {'state': state, 'boundary_aware_functions': ba}


# Functional definitions
# Note negative axes are used to enable function for both T,S (indexed [lat,lng]) and v (indexed [direction, lat, lng])

class BoundaryAware():

    def __init__(self, state):

        def neighbours(state):
            s_i_next = jnp.roll(state, shift=1, axis=-2)
            s_i_prev = jnp.roll(state, shift=-1,axis=-2)
            s_j_next = jnp.roll(state, shift=1 ,axis=-1)
            s_j_prev = jnp.roll(state, shift=-1,axis=-1)

            return s_i_next + s_i_prev + s_j_next + s_j_prev
        

        self.state = state
        self.neighbours = neighbours(state)
        self.inner_solid = neighbours(1-state)==4
    
    def laplacian(self,y):
        return self.laplacian_divgrad(y)

    def laplacian_divgrad(self, y):
        
        return self.divergence(self.gradient(y)) +  y*(-1+self.state)
        
    def laplacian_rolls(self, y):
        yb = y*self.state

        y_i_next = jnp.roll(yb, shift=1, axis=-2)
        y_i_prev = jnp.roll(yb, shift=-1,axis=-2)
        y_j_next = jnp.roll(yb, shift=1 ,axis=-1)
        y_j_prev = jnp.roll(yb, shift=-1,axis=-1)
        return  y*(1-self.state) + self.state*(y_j_next #
                                            + y_i_next
                                            - self.nbs * yb 
                                            + y_j_prev 
                                            + y_i_prev) / (DTHETA*DLAMBDA)
    
    def laplacian_ps(self, y):
        return self.ptheta(self.ptheta(y)) + self.plambda(self.plambda(y))


    def ptheta(self, y):
        y_i_next = jnp.roll(y,          shift=1, axis=-2)
        y_i_prev = jnp.roll(y,          shift=-1,axis=-2)

        f_i_next = jnp.roll(self.state, shift=1, axis=-2)
        f_i_prev = jnp.roll(self.state, shift=-1,axis=-2)

        return self.state * ( # there is only a valid derivative in fluid cells
                 f_i_next * f_i_prev     * (y_i_next - y_i_prev) / 2  # case where both neighbours are fluid
               + f_i_next * (1-f_i_prev) * (y_i_next - y)             # case where only the next neighbour is fluid
               + (1-f_i_next) * f_i_prev * (y - y_i_prev)             # case where only the previous neighbour is fluid
            )/DTHETA                                        # when neither neighbour is fluid, the previous two cases cancel out
             
    def plambda(self, y): 
        y_j_next = jnp.roll(y,          shift= 1,axis=-1)
        y_j_prev = jnp.roll(y,          shift=-1,axis=-1)

        f_j_next = jnp.roll(self.state, shift= 1,axis=-1)
        f_j_prev = jnp.roll(self.state, shift=-1,axis=-1)

        return self.state * (  # there is only a valid derivative in fluid cells
                 f_j_next * f_j_prev     * (y_j_next - y_j_prev) / 2  # case where both neighbours are fluid
               + f_j_next * (1-f_j_prev) * (y_j_next - y)             # case where only the next neighbour is fluid
               + (1-f_j_next) * f_j_prev * (y - y_j_prev)             # case where only the previous neighbour is fluid
            )/DTHETA                                        # when neither neighbour is fluid, the previous two cases cancel out

    def gradient(self, y):
        return jnp.array( (self.ptheta(y), self.plambda(y)))
    
    def divergence(self, y):
        return self.divergence_roll(y)

    def divergence_ps(self, y):
        return self.ptheta(y[0]) + self.plambda(y[1])

    def divergence_roll(self, y):
        y_i_next = jnp.roll(y[0], shift=1, axis=-2)
        y_i_prev = jnp.roll(y[0], shift=-1,axis=-2)
        y_j_next = jnp.roll(y[1], shift=1 ,axis=-1)
        y_j_prev = jnp.roll(y[1], shift=-1,axis=-1)
        return (y_i_next - y_i_prev + y_j_next - y_j_prev) #*self.state
    
    def advection(self, v, y):
        return self.state*(v[0] * self.ptheta(y) + v[1] * self.plambda(y) )
    
    def blur(self, y):
        return (jnp.roll(y,  1, axis=-2)
              + jnp.roll(y, -1, axis=-2)
              + jnp.roll(y,  1, axis=-1)
              + jnp.roll(y, -1, axis=-1)
              + y) / (self.neighbours + 1) *self.state

    # Incompressability 

    def project_divergencefree(self, v):

        v *= self.state

        iters = (150, 100, 50)
        blurs = (True, True, True)

        v_f = v
        # Repeats as conjugate gradient descent does not necessarily converge to correct solution after 1 run
        for i in range(len(iters)):

            # q is the exact "pressure" needed to maintain densities
            q, _ = jsp.sparse.linalg.cg(
                self.laplacian,
                -self.divergence(v_f), 
                maxiter= iters[i])
            
            v_f += self.gradient(q)
            if blurs[i]:
                v_f = self.blur(v_f)

        return v_f    
        

   
