from typing import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import Array, Float  # https://github.com/google/jaxtyping




jax.config.update("jax_enable_x64", True)

# Represents the interval [x0, x_final] discretised into n equally-spaced points.
class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    @classmethod
    def squish(cls, sd):
        return cls(sd.x0, sd.x_final, sd.vals[-1])



class Simulator():

    def __init__(self, config):
        print('Beep boop, sim init')
        s = self; c = config

        # No config for args because I don't know what they are for :)
        s.args = None

        # Set term as decided in config
        vf = {'vf_flow_simplest': vf_flow_simplest,
              'vf_flow_zero': vf_flow_zero,
              }[c['problem']['vector_field']]
        s.term = diffrax.ODETerm(vf)

        # Set initial condition as decided in config
        ic = {'ic_flow_basic': ic_flow_basic,
              }[c['problem']['initial_condition']]
        
        # Create spatial discretisation of the initial conditions
        lat_first = c['spatial_discretisation']['lat_first']
        lat_final = c['spatial_discretisation']['lat_final']
        lat_n     = c['spatial_discretisation']['lat_n']
        lng_first = c['spatial_discretisation']['lng_first']
        lng_final = c['spatial_discretisation']['lng_final']
        lng_n     = c['spatial_discretisation']['lng_n']

        lat,lng = jnp.mgrid[lat_first:lat_final:lat_n*1j,
                            lng_first:lng_final:lng_n*1j]
        
        s.y0 = ic(lat,lng)
        print('ic created')



        import displayer
        dis = displayer.Displayer()
        #breakpoint()
        #yss = jnp.repeat(jnp.expand_dims(s.y0, 2), 10, axis=2)
        #for i in range(10):
        #    yss = yss.at[:,:,i].mul(10-i)
        #breakpoint()
        #dis.draw_chunk_3d(config, yss)

        #dis.draw_chunk_2d(config, s.y0)
        #dis.draw_chunk_2d(config, laplacian(s.y0))


        

        # Set stepsize controller with stepsize options
        controller = {'diffrax_PIDController': diffrax.PIDController,
                      }[c['solver_options']['stepsize_controller']['type']] 
        s.stepsize_controller = controller(
            pcoeff = c['solver_options']['stepsize_controller']['pcoeff'],
            icoeff = c['solver_options']['stepsize_controller']['icoeff'],
            rtol = float(c['solver_options']['stepsize_controller']['rtol']),
            atol = float(c['solver_options']['stepsize_controller']['atol']),
            dtmax = c['solver_options']['stepsize_controller']['dtmax'])

        # Set solver as decided in config
        s.solver = {'diffrax_Tsit5': diffrax.Tsit5(),
                    }[c['solver_options']['type']]


    def simulate_continuous_chunks(self, config):
        print('Beep boop, simulating chunks forever. These don\'t get displayed as this is not finished')

        s = self; c = config

        # Load some config
        t_first        = c['temporal_discretisation_infinite']['t_first']
        t_each_solve   = c['temporal_discretisation_infinite']['t_each_solve']
        t_n_each_solve = c['temporal_discretisation_infinite']['t_n_each_solve']

        # Live calculation
        t_prev = t_first
        t_next = t_first + t_each_solve
        y = s.y0

        saved_ys = []

        i = 0
        while True:

            sol = diffrax.diffeqsolve(
                s.term,
                s.solver,
                t_prev,
                t_next,
                c['solver_options']['finite_diff_dt'],
                y,
                saveat=diffrax.SaveAt( ts = jnp.linspace(t_prev,t_next, t_n_each_solve+1)[1:]),
                stepsize_controller=s.stepsize_controller,
                max_steps=None,
            )

            t_prev = t_next
            t_next = t_next + t_each_solve

            saved_ys.append(sol.ys.vals)
            
            y = SpatialDiscretisation.squish(sol.ys)
            i += t_n_each_solve
            print("{} timesteps saved, now processing time {}".format(i,t_prev))

        print("Done :D")
        return saved_ys
    

    def simulate_chunk(self, config):
        print('Beep boop, simulating a chunk')

        s = self; c = config

        # Load some config
        t_first = c['temporal_discretisation_finite']['t_first']
        t_final = c['temporal_discretisation_finite']['t_final']
        t_n     = c['temporal_discretisation_finite']['t_n']

        # Run solver
        sol = diffrax.diffeqsolve(
            s.term,
            s.solver,
            t_first,
            t_final,
            c['solver_options']['finite_diff_dt'],
            s.y0,
            saveat=diffrax.SaveAt( ts = jnp.linspace(t_first,t_final, t_n+1)),
            stepsize_controller=s.stepsize_controller,
            max_steps=None,
        )

        print("Done :D")
        return sol.ys
    

    

def laplacian_old(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    Δy = (y_next - 2 * y.vals + y_prev) / (y.δx**2)
    # Dirichlet boundary condition
    Δy = Δy.at[0].set(0)
    Δy = Δy.at[-1].set(0)
    return SpatialDiscretisation(y.x0, y.x_final, Δy)

def laplacian(y):
    #return y*0

    y_i_next = jnp.roll(y, shift=1, axis=0)
    y_i_prev = jnp.roll(y, shift=-1,axis=0)
    y_j_next = jnp.roll(y, shift=1 ,axis=1)
    y_j_prev = jnp.roll(y, shift=-1,axis=1)
    return (y_j_next + y_i_next - 4 * y + y_j_prev + y_i_prev) / (0.01**2)
    # Dirichlet boundary condition
    #Δy = Δy.at[0].set(0)
    #Δy = Δy.at[-1].set(0)

# Problem

def vf_flow_zero(t, T, args):
    return T*0


def vf_flow_simplest(t, T, args):
    return laplacian(T)




def ic_flow_basic(lat,lng):
    return 1 * ((jnp.square(lat) + jnp.square(lng))<0.7)


if __name__ == '__main__':
    print('Running simulator.py has no effect')