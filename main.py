
import yaml
import simulator
import jax.numpy as jnp

CONFIG_LOCATION = "config_basic.yml"

def main():

    with open(CONFIG_LOCATION, 'r') as file:
        config = yaml.safe_load(file)
    
    sim = simulator.Simulator(config)

    if config['run_infinite']:
        sim.simulate_continuous_chunks(config)
    
    elif not config['run_infinite']:
        sim.draw_chunk(config, sim.simulate_chunk(config))



if __name__ == '__main__':
    main()