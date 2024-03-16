import cv2
import numpy as np
import os

def get_map_path(config):

    if config["map"]["mask_image_path_relative"]:
        print('opening path ', os.path.join( os.getcwd(), config["map"]["mask_image_path"] ))
        print(os.path.join( os.getcwd(), config["map"]["mask_image_path"] ))
        return os.path.join( os.getcwd(), config["map"]["mask_image_path"] )
    else:
        return config["map"]["mask_image_path"]


def load_image_np(config):

    image_np = cv2.imread(get_map_path(config)) 

    return image_np[:,:,::-1]


def load_state_np(config):

    image = load_image_np(config)

    lat_n, lng_n = config['spatial_discretisation']['lat_n'], config['spatial_discretisation']['lng_n']
    small_im = cv2.resize(image, (lng_n, lat_n), interpolation=cv2.INTER_LINEAR)

    return small_im[:,:,2]>5



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    config = {"map": {"mask_image_path_relative": True, 
                      "mask_image_path": "resources/map/gebco_08_rev_bath_3600x1800_color.jpg"}}
    
    plt.imshow(load_state_np(config))
    

    


    
