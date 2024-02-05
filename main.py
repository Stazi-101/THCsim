print('started running')
import yaml
import simulator
import jax.numpy as jnp
import displayer
print('stuff imported')
CONFIG_LOCATION = "config_basic.yml"

def main():

    # Load config data
    with open(CONFIG_LOCATION, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialise objects, including creating initial data
    sim = simulator.Simulator(config)
    dis = displayer.Displayer()

    # Incomplete: Run indefinitely
    if config['run_infinite']:
        sim.simulate_continuous_chunks(config)
    
    # Simulate a finite chunk of time
    elif not config['run_infinite']:
        dis.draw_chunk_3d(config, sim.simulate_chunk(config))



if __name__ == '__main__':
    main()
    