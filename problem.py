import jax.numpy as jnp
import jax.scipy as jsp


# Vector fields: list of ys -> list of dy/dts
def vf_flow_zero(t, T, args):
    return T*0


def vf_flow_simplest(t, T, args):
    return laplacian(T)


def vf_flow_basic(t, ys, args):
    T, S, v = ys
    config = args[0]

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * laplacian(T) + advection(v, T)
    dS = A_HH * laplacian(S) + advection(v, S)

    dv = A_MH * laplacian(v) #+ advection(v, v)
    dv = dv.at[0].add( -1/rho_0 * ptheta(p) + advection(v,v[0]))
    dv = dv.at[1].add( -1/rho_0 * plambda(p) + advection(v,v[1]))


    return (dT, dS, dv)

def vf_flow_incompressible(t, ys, args):
    T, S, v = ys
    config = args[0]

    A_HH = config['constants']['horizontal_diffusivity']
    A_MH = config['constants']['horizontal_viscosity']
    rho_0 = config['constants']['reference_density']

    p = 1 - T*0.01 - S*0.01

    dT = A_HH * laplacian(T) + advection(v, T)
    dS = A_HH * laplacian(S) + advection(v, S)

    dv = A_MH * laplacian(v) #+ advection(v, v)
    dv = dv.at[0].add( -1/rho_0 * ptheta(p) + advection(v,v[0]))
    dv = dv.at[1].add( -1/rho_0 * plambda(p) + advection(v,v[1]))

    dv = project_divergencefree(dv)

    return (dT, dS, dv)

# Initial conditions: coordinates -> list of ys
def ic_flow_basic(lat,lng):

    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    v = v.at[1].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )

    return T, S, v


def ic_flow_ts_only(lat,lng):

    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))

    return T, S, v

def ic_flow_funky(lat,lng):

    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)
    S = 1 * ((jnp.square(lat) + jnp.square(lng+1))<0.7)
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    v = v.at[1].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<1) )
    #v = v.at[0].set( 1 * ((jnp.square(lat+0.8) + jnp.square(lng+0.2))<.1) )
    #v = v.at[0].set(-1 * ((jnp.square(lat+0.3) + jnp.square(lng+0.8))<.1) )

    return T, S, v

def ic_flow_vt_only(lat,lng):

    T = 1 * ((jnp.square(lat) + jnp.square(lng))<0.2)
    S = 0 * lat
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    v = v.at[0].set( 1 * ((jnp.square(lat) + jnp.square(lng))<0.8) )
    #v = v.at[1].set( -1 * ((jnp.square(lat-0.5) + jnp.square(lng-0.5))<0.5) )

    return T, S, v

def ic_flow_v_only(lat,lng):

    T = 0 * lat
    S = 0 * lat
    v = jnp.zeros((2, lat.shape[0], lat.shape[1]))
    #v = v.at[0, 50:70, 50:70].set(1)
    v = v.at[0].set( 1 * ((jnp.square(lat+0.5) + jnp.square(lng+0.5))<0.5) )
    #v = v.at[1].set( -1 * ((jnp.square(lat-0.5) + jnp.square(lng-0.5))<0.5) )

    return T, S, v


# Functional definitions
def laplacian(y):

    y_i_next = jnp.roll(y, shift=1, axis=-2)
    y_i_prev = jnp.roll(y, shift=-1,axis=-2)
    y_j_next = jnp.roll(y, shift=1 ,axis=-1)
    y_j_prev = jnp.roll(y, shift=-1,axis=-1)
    return (y_j_next + y_i_next - 4 * y + y_j_prev + y_i_prev) / (0.01**2)

def ptheta(y):
    y_i_next = jnp.roll(y, shift=1, axis=-2)
    y_i_prev = jnp.roll(y, shift=-1,axis=-2)
    return (y_i_next - y_i_prev) / (2*0.01)

def plambda(y):
    y_j_next = jnp.roll(y, shift=1, axis=-1)
    y_j_prev = jnp.roll(y, shift=-1,axis=-1)
    return (y_j_next - y_j_prev) / (2*0.01)

def advection(v, y):
    if len(y.shape)>=3:
        raise RuntimeError('y too many dimensions')
    return v[0] * ptheta(y) + v[1] * plambda(y)

# Incompressability stuff
def divergence(v):
    return ptheta(v[0]) + plambda(v[1])

def gradient(y):
    return jnp.array( (ptheta(y), plambda(y)))

def project_divergencefree(v):
    # See test_nb for proof that this doesn't work correctly
    q, _ = jsp.sparse.linalg.cg(
        laplacian,
        -divergence(v), 
        maxiter= 1000)
    
    v_f = v + 0.1*gradient(q)
    
    return v_f

def aghhhh(v):

    q, _ = jsp.sparse.linalg.cg(
        laplacian,
        -divergence(v), 
        maxiter= 1000)
    return gradient(q)