import cv2
import numpy as np
import os

def get_map_path(config):

    if config["map"]["mask_image_path_relative"]:
        return os.path.join( os.getcwd(), config["map"]["mask_image_path"] )
    else:
        return config["map"]["mask_image_path"]





def load_np(config):

    image_np = cv2.imread(get_map_path(config), cv2.IMREAD_GRAYSCALE) 

    #image_jax = jnp.array(image_np)
    
    #return image_jax
    return image_np



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    config = {"map": {"mask_image_path_relative": True, 
                      "mask_image_path": "resources\map\equirectangular_1920_wikipedia_blue_marble_2002.png"}}
    
    image_np = load_np(config)
    print(image_np.shape)
    print(np.unique(image_np))

    #image_np = image_jax.numpy()

    plt.imshow(image_np)
    plt.show()


    
