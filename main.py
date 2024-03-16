print('started running')
import yaml
import simulator
import jax.numpy as jnp
#import displayer
import save_load
import jax
import datetime
print('stuff imported')
CONFIG_LOCATION = "config_basic.yml"

def main():
    """
    Entry point of the program.
    """
    # Load config data
    with open(CONFIG_LOCATION, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialise objects, including creating initial data
    print('Beep boop, sim creating initial condition state...')

    sim = simulator.Simulator(config)
    #dis = displayer.Displayer()

    # Incomplete: Run indefinitely
    if config['run_infinite']:
        sim.simulate_continuous_chunks(config)
        
    # Simulate a finite chunk of time
    else:
        print('Simulating ..... ')
        t1 = datetime.datetime.now()
        sim_func = jax.jit(lambda x : sim.simulate_chunk(config))
        chunk = sim_func(0)
        t2 = datetime.datetime.now()
        print(f"Time taken: {t2-t1}")
        print('Simulation done :D')
        
    if config['display']:
        print('Display from see.ipynm')
        
    if config['save']['save_pkl']:
        print('Saving...')
        save_load.save(config, chunk, sim.args)
        print('saved')



if __name__ == '__main__':
    main()
    