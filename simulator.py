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

    def __init__(s, config):
        print('Beep boop, sim init')
        s.term = diffrax.ODETerm(vector_field)
        s.solver = diffrax.Tsit5()

        # Initial condition
        s.ic = lambda x: x**2

        # Spatial discretisation
        s.x0 = -1
        s.x_final = 1
        s.xn = 200 # Number of discrete values of x
        s.y0 = SpatialDiscretisation.discretise_fn(s.x0, s.x_final, s.xn, s.ic)

        # Temporal discretisation
        s.t0 = 0
        s.t_final = 1
        s.tn = 200 # Number of values of t where result is saved
        s.tnrun = 40 # Number of times diffeqsolve is run
        s.δt = 0.01 # Timestep in 1 step of solver?
        s.dt0 = (s.t_final-s.t0)/(s.tn-1)*s.tnrun


        s.args = None

        # Tolerances
        s.rtol = 1e-10
        s.atol = 1e-10
        s.stepsize_controller = diffrax.PIDController(
            pcoeff=0.3, icoeff=0.4, rtol=s.rtol, atol=s.atol, dtmax=0.001
        )

    def simulate_chunk(s, config, ts):
        print('Beep boop, simulating a chunk')

        # Live calculation
        s.tprev = s.t0
        s.tnext = s.t0 + s.dt0
        s.y = s.y0
        #state = solver.init(term, tprev, tnext, y0, args)


        s.ys = jnp.zeros((s.tn,s.xn))

        i = 0
        while s.tprev < s.t_final:

            s.sol = diffrax.diffeqsolve(
                s.term,
                s.solver,
                s.tprev,
                s.tnext,
                s.δt,
                s.y,
                saveat=diffrax.SaveAt( ts = jnp.linspace(s.tprev,s.tnext, s.tnrun+1)[1:]),
                stepsize_controller=s.stepsize_controller,
                max_steps=None,
            )

            s.tprev = s.tnext
            s.tnext = min(s.tnext + s.dt0, s.t_final)
            
            # Save data
            if i+s.tnrun-1 >= s.tn:
                break

            s.ys = s.ys.at[i:i+s.tnrun].set(s.sol.ys.vals)
            
            s.y = SpatialDiscretisation.squish(s.sol.ys)
            print("{} / {}".format(i, s.tn))
            i += s.tnrun
            

        print("Done :D")

        plt.figure(figsize=(5, 5))
        plt.imshow(
            s.ys,
            origin="lower",
            extent=(s.x0, s.x_final, s.t0, s.t_final),
            aspect=(s.x_final - s.x0) / (s.t_final - s.t0),
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



if __name__ == '__main__':
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()

    # Initial condition
    ic = lambda x: x**2

    # Spatial discretisation
    x0 = -1
    x_final = 1
    xn = 200 # Number of discrete values of x
    y0 = SpatialDiscretisation.discretise_fn(x0, x_final, xn, ic)

    # Temporal discretisation
    t0 = 0
    t_final = 1
    tn = 200 # Number of values of t where result is saved
    tnrun = 40 # Number of times diffeqsolve is run
    δt = 0.01 # Timestep in 1 step of solver?
    dt0 = (t_final-t0)/(tn-1)*tnrun


    args = None

    # Tolerances
    rtol = 1e-10
    atol = 1e-10
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
    )

    # Live calculation
    tprev = t0
    tnext = t0 + dt0
    y = y0
    #state = solver.init(term, tprev, tnext, y0, args)


    ys = jnp.zeros((tn,xn))

    i = 0
    while tprev < t_final:

        sol = diffrax.diffeqsolve(
            term,
            solver,
            tprev,
            tnext,
            δt,
            y,
            saveat=diffrax.SaveAt( ts = jnp.linspace(tprev,tnext, tnrun+1)[1:]),
            stepsize_controller=stepsize_controller,
            max_steps=None,
        )

        tprev = tnext
        tnext = min(tnext + dt0, t_final)
        
        # Save data
        if i+tnrun-1 >= tn:
            break

        ys = ys.at[i:i+tnrun].set(sol.ys.vals)
        
        y = SpatialDiscretisation.squish(sol.ys)
        print("{} / {}".format(i, tn))
        i += tnrun
        

    print("Done :D")

    plt.figure(figsize=(5, 5))
    plt.imshow(
        ys,
        origin="lower",
        extent=(x0, x_final, t0, t_final),
        aspect=(x_final - x0) / (t_final - t0),
        cmap="inferno",
    )
    plt.xlabel("x")
    plt.ylabel("t", rotation=0)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()
