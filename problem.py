import jax.numpy as jnp
import jax.scipy as jsp

CG_REPEATS = 2
DT = 0.01
MAXITER = 100

# Vector fields: list of ys -> list of dy/dts
def vf_flow_zero(t, T, args):
    return T*0


def vf_flow_simplest(t, T, args):
    return laplacian(T)


def vf_flow_basic(t, ys, args):
    T, S, v = ys
    config = args['config']

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * laplacian(T) + advection(v, T)
    dS = A_HH * laplacian(S) + advection(v, S)

    dv = A_MH * laplacian(v) 
    dv = dv.at[0].add( -1/rho_0 * ptheta(p) + advection(v,v[0]))
    dv = dv.at[1].add( -1/rho_0 * plambda(p) + advection(v,v[1]))


    return (dT, dS, dv)

def vf_flow_incompressible(t, ys, args):
    T, S, v = ys
    config = args['config']

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * laplacian(T) + advection(v, T)
    dS = A_HH * laplacian(S) + advection(v, S)

    dv = A_MH * laplacian(v)
    dv = dv.at[0].add( -1/rho_0 * ptheta(p) + advection(v,v[0]))
    dv = dv.at[1].add( -1/rho_0 * plambda(p) + advection(v,v[1]))

    dv = project_divergencefree(dv, args['q_last'])

    return (dT, dS, dv)

def vf_flow_semicompressible(t, ys, args):
    T, S, v = ys
    config = args['config']

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * laplacian(T) + advection(v, T)
    dS = A_HH * laplacian(S) + advection(v, S)

    dv = A_MH * laplacian(v)
    dv = dv.at[0].add( advection(v,v[0]) )
    dv = dv.at[1].add( advection(v,v[1]) )

    dv = project_divergencefree(dv, args['q_last'])

    dv = dv.at[0].add( -1/rho_0 * ptheta(p) )
    dv = dv.at[1].add( -1/rho_0 * plambda(p) )

    return (dT, dS, dv)



# Initial conditions: coordinates -> list of ys
def ic_flow_basic(lat,lng):
    
    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7) 
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)

    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add a spot of vertical movement
    v = v.at[1].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )
    # Force v to be divergence free
    v = project_divergencefree(v)

    return (T, S, v), {}


def ic_flow_basic_noboundary(lat,lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7) 
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)

    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add a spot of vertical movement
    v = v.at[1].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )
    # Force v to be divergence free
    v = project_divergencefree(v)

    return (T, S, v), {}

def ic_flow_ts_only(lat,lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)
    # Salty spot
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)
    # Initialise v is 0. Naturally divergence free :D
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))

    return (T, S, v), {}

def ic_flow_funky(lat,lng):

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

    return (T, S, v), {}

def ic_flow_vt_only(lat,lng):

    # Hot spot in centre
    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.2)
    # Initialise salinity is 0
    S = 0 * lat
    
    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add circles of differing velocities
    v = v.at[0].set( 1 * ((jnp.square(lat) + jnp.square(lng))<0.8) )
    v = v.at[1].set( -1 * ((jnp.square(lat-0.5) + jnp.square(lng-0.5))<0.5) )

    return (T, S, v), {}

def ic_flow_v_only(lat,lng):

    # Initialise T and S are 0
    T = 0 * lat
    S = 0 * lat
    # Initialise v as 0
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    # Add moving spot
    v = v.at[0].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<0.5) )

    return (T, S, v), {}


# Functional definitions
# Note negative axes are used to enable function for both T,S (indexed [lat,lng]) and v (indexed [direction, lat, lng])
def ptheta(y):
    y_i_next = jnp.roll(y, shift=1, axis=-2)
    y_i_prev = jnp.roll(y, shift=-1,axis=-2)
    return (y_i_next - y_i_prev) / (2*0.01)

def plambda(y):
    y_j_next = jnp.roll(y, shift=1, axis=-1)
    y_j_prev = jnp.roll(y, shift=-1,axis=-1)
    return (y_j_next - y_j_prev) / (2*0.01)

def gradient(y):
    return jnp.array( (ptheta(y), plambda(y)))

def laplacian(y):

    y_i_next = jnp.roll(y, shift=1, axis=-2)
    y_i_prev = jnp.roll(y, shift=-1,axis=-2)
    y_j_next = jnp.roll(y, shift=1 ,axis=-1)
    y_j_prev = jnp.roll(y, shift=-1,axis=-1)
    return (y_j_next + y_i_next - 4 * y + y_j_prev + y_i_prev) / (DT**2)

def advection(v, y):
    if len(y.shape)>=3:
        raise RuntimeError('y too many dimensions')
    return v[0] * ptheta(y) + v[1] * plambda(y)

def divergence(v):
    return ptheta(v[0]) + plambda(v[1])


# Incompressability 

def project_divergencefree(v, q_guess=None):

    q, _ = jsp.sparse.linalg.cg(
            laplacian,
            -divergence(v), 
            x0=q_guess,
            maxiter= MAXITER)
    #q_total = q
    v_f = v + gradient(q)
    # Repeats as conjugate gradient descent does not necessarily converge to correct solution after 1 run
    for i in range(CG_REPEATS):
        # q is the exact "pressure" needed to maintain densities
        q, _ = jsp.sparse.linalg.cg(
            laplacian,
            -divergence(v_f), 
            maxiter= MAXITER)
        #q_total += q
        v_f += gradient(q)
    return v_f

