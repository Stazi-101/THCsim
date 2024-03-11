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


def load_np(config):

    image_np = cv2.imread(get_map_path(config)) 

    return image_np[:,:,::-1]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    config = {"map": {"mask_image_path_relative": True, 
                      "mask_image_path": "resources/map/gebco_08_rev_bath_3600x1800_color.jpg"}}
    
    im = load_np(config).astype(int)
    print(im.shape)

    im[:,:,:] += (np.isclose(im[:,:,0],im[:,:,2], atol=4))[:,:,np.newaxis]*255

    im[im>255]=255

    plt.imshow(im)
    #plt.show()


    
