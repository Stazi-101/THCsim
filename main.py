
import yaml
import simulator
import jax.numpy as jnp

CONFIG_LOCATION = "config_basic.yml"

def main():

    with open(CONFIG_LOCATION, 'r') as file:
        config = yaml.safe_load(file)
    
    sim = simulator.Simulator(config)

    if config['run_length'] == 'infinite':
        sim.simulate_continuous(config)
    
    elif type(config['run_length']) is float:
        sim.simulate_chunk( config, jnp.arange(0, config['run_length'], config['t_step_saved']))




if __name__ == '__main__':
    main()