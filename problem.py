import jax.numpy as jnp


def vf_flow_zero(t, T, args):
    return T*0


def vf_flow_simplest(t, T, args):
    return laplacian(T)


def ic_flow_basic(lat,lng):
    return 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)




def laplacian(y):

    y_i_next = jnp.roll(y, shift=1, axis=0)
    y_i_prev = jnp.roll(y, shift=-1,axis=0)
    y_j_next = jnp.roll(y, shift=1 ,axis=1)
    y_j_prev = jnp.roll(y, shift=-1,axis=1)
    return (y_j_next + y_i_next - 4 * y + y_j_prev + y_i_prev) / (0.01**2)
