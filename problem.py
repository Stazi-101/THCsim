import jax.numpy as jnp
import jax.scipy as jsp

CG_REPEATS = 2
DT = 0.01
MAXITER = 100

DTHETA = 10
DLAMBDA = 10

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

    #dv = A_MH * ba.laplacian(v)
    dv = v*0
    dv = dv.at[0].add( ba.advection(v,v[0]) )  # -1/rho_0 * ba.ptheta(p) 
    dv = dv.at[1].add( ba.advection(v,v[1])  )  # -1/rho_0 * ba.plambda(p)

    state = args['state']

    dv = ba.project_divergencefree(dv)
    #dv = ba.blur(dv)
    dv += A_MH * ba.laplacian(v)

    dv *= (ba.inner_fluid)
    dT *= ba.inner_fluid
    dS *= ba.inner_fluid

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

    state = 1 - 1 * ((jnp.square(lat+0.5) + jnp.square(lng+2))<0.7)
    ba = BoundaryAware(state)
    
    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7) 
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)

    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add a spot of vertical movement
    v = v.at[1].set( -10 * ((jnp.square(lat+0.5) + jnp.square(lng-1.5))<1) )

    for i in range(5):
        v = ba.blur(v)

    #v = v*0 + 1

    v *= ba.inner_fluid[jnp.newaxis]
    T *= ba.inner_fluid
    S *= ba.inner_fluid

    v = ba.project_divergencefree(v)

    R = config['spatial_discretisation']['earth_radius']
    overRsintheta = 1/(R*jnp.sin(lat))
    

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

    def __init__(self, fluid):

        def neighbours(state):
            s_i_next = jnp.roll(state, shift=1, axis=-2)
            s_i_prev = jnp.roll(state, shift=-1,axis=-2)
            s_j_next = jnp.roll(state, shift=1 ,axis=-1)
            s_j_prev = jnp.roll(state, shift=-1,axis=-1)

            return s_i_next + s_i_prev + s_j_next + s_j_prev
        

        self.fluid = fluid
        self.neighbours = neighbours(fluid)
        self.inner_solid = neighbours(1-fluid)==4
        self.inner_fluid = self.neighbours==4
    
    def laplacian_solvey(self,y):
        return self.divergence(self.gradient(y)) - y*(self.inner_solid)
        
    def laplacian(self, y):
        y_i_next = jnp.roll(y, shift=1, axis=-2)
        y_i_prev = jnp.roll(y, shift=-1,axis=-2)
        y_j_next = jnp.roll(y, shift=1 ,axis=-1)
        y_j_prev = jnp.roll(y, shift=-1,axis=-1)
        return (y_i_next + y_i_prev + y_j_next + y_j_prev - self.neighbours*y) / (DTHETA * DLAMBDA) * self.fluid 

    def ptheta(self, y):
        y_i_next = jnp.roll(y, shift=1, axis=-2)
        y_i_prev = jnp.roll(y, shift=-1,axis=-2)

        return self.fluid * (y_i_next - y_i_prev) / (2 * DTHETA)  
             
    def plambda(self, y): 
        y_j_next = jnp.roll(y,          shift= 1,axis=-1)
        y_j_prev = jnp.roll(y,          shift=-1,axis=-1)

        return self.fluid * (y_j_next - y_j_prev) / (2 * DLAMBDA)  

    def gradient(self, y):
        return jnp.array( (self.ptheta(y), self.plambda(y)))
    
    def divergence(self, y):
        y_i_next = jnp.roll(y[0], shift=1, axis=-2)
        y_i_prev = jnp.roll(y[0], shift=-1,axis=-2)
        y_j_next = jnp.roll(y[1], shift=1 ,axis=-1)
        y_j_prev = jnp.roll(y[1], shift=-1,axis=-1)
        return (y_i_next - y_i_prev + y_j_next - y_j_prev)
    
    def advection(self, v, y):
        return self.fluid*(v[0] * self.ptheta(y) + v[1] * self.plambda(y) )
        
    def blur(self, y):
        return (jnp.roll(y*self.fluid,  1, axis=-2)
            + jnp.roll(y*self.fluid, -1, axis=-2)
            + jnp.roll(y*self.fluid,  1, axis=-1)
            + jnp.roll(y*self.fluid, -1, axis=-1)
            + y) / (self.neighbours + 1 ) *self.fluid

    # Incompressability 

    def project_divergencefree(self, v):

        iters = (150, 100, 50)

        v_f = v
        # Repeats as conjugate gradient descent does not necessarily converge to correct solution after 1 run
        for i in range(len(iters)):

            # q is the exact "pressure" needed to maintain densities
            q, _ = jsp.sparse.linalg.cg(
                self.laplacian_solvey,
                -self.divergence(v_f), 
                maxiter= iters[i])
            
            v_f += self.gradient(q)

        return v_f    
        

   
