
from os.path import isfile
import pickle as pkl

def save(config, X, args):
    
    path = config['save']['save_pkl_path']
    
    # If we have set to not overwrite, find lowest number to add to end of file name to avoid overwriting
    if not config['save']['save_pkl_overwrite']:
        empty_path_found = False
        i=0
        while not empty_path_found:
            if isfile(path+str(i)+'.pkl'):
                i += 1
                
            else:
                empty_path_found = True
                path = path+str(i)
    
    with open(path+'.pkl', 'wb') as file:
        pkl.dump((X, args), file)
    return True
        

    