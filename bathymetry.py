import cv2
import numpy as np
import os

def get_map_path(config):

    if config["map"]["mask_image_path_relative"]:
        return os.path.join( os.getcwd(), config["map"]["mask_image_path"] )
    else:
        return config["map"]["mask_image_path"]





def load_np(config):

    image_np = cv2.imread(get_map_path(config)) 
    

    #image_jax = jnp.array(image_np)
    
    #return image_jax
    return image_np[:,:,::-1]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    config = {"map": {"mask_image_path_relative": True, 
                      "mask_image_path": "resources\map\gebco_08_rev_bath_3600x1800_color.jpg"}}
    
    im = load_np(config).astype(int)
    print(im.shape)
    #image_np = cv2.resize(image_np, dsize=(90,45), interpolation=cv2.INTER_NEAREST_EXACT)
    #print(image_np.shape)
    im[:,:,:] += (np.isclose(im[:,:,0],im[:,:,2], atol=4))[:,:,np.newaxis]*255
    #print(np.unique(image_np))

    #image_np = image_jax.numpy()
    im[im>255]=255

    plt.imshow(im)
    plt.show()


    
