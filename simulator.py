from typing import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping

import problem

jax.config.update("jax_enable_x64", True)

class Simulator():

    def __init__(self, config):
        print('Beep boop, sim init')
        s = self; c = config

        s.args = [config]

        # Set term as decided in config
        vf = {'vf_flow_simplest': problem.vf_flow_simplest,
              'vf_flow_zero': problem.vf_flow_zero,
              'vf_flow_basic': problem.vf_flow_basic,
              'vf_flow_incompressible': problem.vf_flow_incompressible
              }[c['problem']['vector_field']]
        s.term = diffrax.ODETerm(vf)

        # Set initial condition as decided in config
        ic = {'ic_flow_basic': problem.ic_flow_basic,
              'ic_flow_basic_divfree': problem.ic_flow_basic_divfree,
              'ic_flow_v_only': problem.ic_flow_v_only,
              'ic_flow_vt_only': problem.ic_flow_vt_only,
              'ic_flow_funky': problem.ic_flow_funky,
              'ic_flow_ts_only': problem.ic_flow_ts_only
              }[c['problem']['initial_condition']]
        
        # Create spatial discretisation 
        lat_first = c['spatial_discretisation']['lat_first']
        lat_final = c['spatial_discretisation']['lat_final']
        lat_n     = c['spatial_discretisation']['lat_n']
        lng_first = c['spatial_discretisation']['lng_first']
        lng_final = c['spatial_discretisation']['lng_final']
        lng_n     = c['spatial_discretisation']['lng_n']

        lat,lng = jnp.mgrid[lat_first:lat_final:lat_n*1j,
                            lng_first:lng_final:lng_n*1j]
        
        # Create values of inital conditions on the discretisation
        s.y0 = ic(lat,lng)
        print('ic created')

        # Check how the initial condition data looks
        '''dis.draw_chunk_2d(config, s.y0)'''

        # Set stepsize controller with stepsize options
        controller = {'diffrax_PIDController': diffrax.PIDController,
                      'diffrax_ConstantStepSize': diffrax.ConstantStepSize,
                      }[c['solver_options']['stepsize_controller']['type']] 
        s.stepsize_controller = controller()
        '''s.stepsize_controller = controller(
            pcoeff = c['solver_options']['stepsize_controller']['pcoeff'],
            icoeff = c['solver_options']['stepsize_controller']['icoeff'],
            rtol = float(c['solver_options']['stepsize_controller']['rtol']),
            atol = float(c['solver_options']['stepsize_controller']['atol']),
            dtmax = c['solver_options']['stepsize_controller']['dtmax'])'''

        # Set solver as decided in config
        s.solver = {'diffrax_Tsit5': diffrax.Tsit5(),
                    'diffrax_Euler': diffrax.Euler()
                    }[c['solver_options']['type']]

    # Incomplete
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
    
    # Solve for the given chunk
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
            args = s.args
        )

        print("Done :D")
        return sol.ys
    




if __name__ == '__main__':
    print('Running simulator.py has no effect')