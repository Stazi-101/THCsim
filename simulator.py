from typing import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
        vf = {'vf_heat_equation': vf_heat_equation,
              }[c['problem']['vector_field']]
        s.term = diffrax.ODETerm(vf)

        # Set initial condition as decided in config
        ic = {'ic_heat_square': ic_heat_square,
              }[c['problem']['initial_condition']]
        
        # Create spatial discretisation of the initial conditions
        s.y0 = SpatialDiscretisation.discretise_fn(
            c['spatial_discretisation']['x_first'],
            c['spatial_discretisation']['x_final'],
            c['spatial_discretisation']['x_n'], 
            ic)

        

        #
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


    def simulate_chunk(self, config):
        print('Beep boop, simulating a chunk')

        s = self; c = config

        # Load some config
        t_first = c['temporal_discretisation']['t_first']
        t_final = c['temporal_discretisation']['t_final']
        t_n     = c['temporal_discretisation']['t_n']
        t_n_run = c['temporal_discretisation']['t_n_solver_runs']

        x_n = c['spatial_discretisation']['x_n']
        
        # Live calculation
        dt0 = (t_final-t_first)/(t_n-1)*t_n_run
        t_prev = t_first
        t_next = t_prev + dt0
        y = s.y0

        ys = jnp.zeros((t_n,x_n))

        i = 0
        while t_prev < t_final:

            sol = diffrax.diffeqsolve(
                s.term,
                s.solver,
                t_prev,
                t_next,
                c['solver_options']['finite_diff_dt'],
                y,
                saveat=diffrax.SaveAt( ts = jnp.linspace(t_prev,t_next, t_n_run+1)[1:]),
                stepsize_controller=s.stepsize_controller,
                max_steps=None,
            )

            t_prev = t_next
            t_next = min(t_next + dt0, t_final)
            
            # Save data
            if i+t_n_run-1 >= t_n:
                break

            ys = ys.at[i:i+t_n_run].set(sol.ys.vals)
            
            y = SpatialDiscretisation.squish(sol.ys)
            print("{} / {}".format(i, t_n))
            i += t_n_run
            

        print("Done :D")
        return ys
    
    def draw(self, config, ys):

        c = config
        # Load some config
        t_first = c['temporal_discretisation']['t_first']
        t_final = c['temporal_discretisation']['t_final']
        

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



    def simulate_continuous(self, config):
        print('Beep boop, simulating continuously')
        pass






def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    Δy = (y_next - 2 * y.vals + y_prev) / (y.δx**2)
    # Dirichlet boundary condition
    Δy = Δy.at[0].set(0)
    Δy = Δy.at[-1].set(0)
    return SpatialDiscretisation(y.x0, y.x_final, Δy)

# Problem
def vector_field(t, y, args):
    return (1 - y) * laplacian(y)

def vf_heat_equation(t, y, args):
    return (1 - y) * laplacian(y)

def ic_heat_square(x):
    return x**2

if __name__ == '__main__':
    print('Running simulator.py has no effect')