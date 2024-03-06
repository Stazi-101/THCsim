from typing import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping

import problem

# jax.config.update("jax_enable_x64", True)


"""
A magnificent contraption known as the Simulator, designed to simulate the intricate
dance of the cosmos with utmost precision and elegance. This marvel of engineering
encapsulates the essence of scientific inquiry and computational prowess, allowing
mere mortals to unravel the secrets of the universe.

Behold, as the Simulator comes to life, driven by the power of configuration and
the boundless imagination of its creator. With each initialization, it utters a
resounding "Beep boop, sim init," signaling the commencement of a grand journey.

The Simulator is equipped with a multitude of features, carefully crafted to
orchestrate the symphony of vector fields, initial conditions, and spatial
discretizations. It possesses the ability to set the term, initial condition,
and spatial discretization based on the whims of the configuration.

As the Simulator awakens, it conjures a spatial grid, a tapestry of coordinates
woven with mathematical precision. It then breathes life into the initial conditions,
infusing them with the essence of the chosen vector field. The resulting creation,
known as 'y0', stands as a testament to the Simulator's creative prowess.

The Simulator's stepsize controller, a master of precision, ensures that the
calculations proceed with utmost accuracy. It can adapt its stepsize based on
the demands of the problem, maintaining a delicate balance between efficiency
and accuracy. The choice of the stepsize controller is dictated by the configuration,
allowing for customization and fine-tuning.

Finally, the Solver, a valiant knight in the realm of differential equations,
takes center stage. The Simulator bestows upon it the power to solve the equations
of motion, to traverse the temporal domain with grace and determination. The choice
of the Solver, guided by the configuration, determines the path the Simulator shall
tread.

With the stage set, the Simulator is ready to embark on its noble quest. It offers
two methods of simulation: 'simulate_continuous_chunks' and 'simulate_chunk'. The
former, an infinite loop of simulation, tirelessly generates chunks of solutions,
forever hidden from the mortal gaze. The latter, a finite endeavor, simulates a
single chunk of time, providing a glimpse into the intricate dance of the cosmos.

As the Simulator completes its tasks, it exclaims with joy, "Beep boop, simulating
chunks forever. These don't get displayed as this is not finished" or "Beep boop,
simulating a chunk," depending on the chosen method. It then returns the fruits
of its labor, the solutions that capture the essence of the simulated phenomena.

Marvel at the Simulator's magnificence, for it is a testament to human ingenuity
and the relentless pursuit of knowledge. May it guide you on your own journey
of discovery and enlightenment.

"""

class Simulator():

    def __init__(self, config):
        s = self; c = config

        s.args = {'config':config,
                  'q_last':None}

        # Set term as decided in config
        vf = {'vf_flow_basic': problem.vf_flow_basic,
              'vf_flow_incompressible': problem.vf_flow_incompressible,
              'vf_flow_semicompressible': problem.vf_flow_semicompressible
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
        s.stepsize_controller = controller() #PUT THIS BACK IN IF YOU WANNA USE CONSTANT STEP SIZE
        '''
        s.stepsize_controller = controller(
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
            
            #y = SpatialDiscretisation.squish(sol.ys)
            i += t_n_each_solve
            print("{} timesteps saved, now processing time {}".format(i,t_prev))

        print("Done :D")
        return saved_ys
    
    # Solve for the given chunk
    def simulate_chunk(self, config):
        
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
        
        print('Done tracing')

        return sol.ys
    




if __name__ == '__main__':
    print('Running simulator.py has no effect')