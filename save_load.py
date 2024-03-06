
from os.path import isfile
import numpy as np
import pickle as pkl

def save(config, X):
    
    path = config['save']['save_npy_path']
    
    # If we have set to not overwrite, find lowest number to add to end of file name to avoid overwriting
    if not config['save']['save_npy_overwrite']:
        empty_path_found = False
        i=0
        while not empty_path_found:
            if isfile(path+str(i)+'.npy'):
                i += 1
                
            else:
                empty_path_found = True
                path = path+str(i)
    
    with open(path+'.npy', 'wb') as file:
        pkl.dump(X, file)
    return True
        

    