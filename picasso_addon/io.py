'''
.. _picasso.localize:
    https://picassosr.readthedocs.io/en/latest/localize.html
.. _picasso.render:
    https://picassosr.readthedocs.io/en/latest/render.html
.. _spt:
    https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1
.. _picasso.filter:
    https://picassosr.readthedocs.io/en/latest/filter.html
'''

import pandas as pd
import h5py
import os
import numpy as np
import picasso.io as io

#%%
def load_locs(path):
    """
    Returns localization .hdf5 from picasso as pandas.DataFrame and corresponding .yaml as list of dictionaries.
    
    Args:
        path (str):      Path to .hdf5 file as generated by `picasso.localize`_
        container (str): 
            Name of dataset within .hdf5 container. Must be set to:
                * ``'locs'`` for _locs, _render and _picked .hdf5 files as generated by `picasso.render`_.
                * ``'groups'`` for _pickprops.hdf5 files as generated by `picasso.render`_.
                * Defaults to ``'locs'``.
    Returns:
        tuple:
            - [0] (pandas.DataFrame): Localizations stored under ``'locs'`` in .hdf5 file
            - [1] (list): List of dictionaries contained in .yaml corresponding to .hdf5 file
    """
    
    locs, info = io.load_locs(path)
    locs = pd.DataFrame(locs)
    
    return locs,info

#%%
def save_locs(path,locs,info,mode=None):
    """
    Save localizations in .hdf5 container and corresponding info in .yaml.
    
    Args:
        locs (pandas.DataFrame): Localizations as obtained by `picasso.localize`_ but converted to pands.DataFrame.
        info (list):             Info as list of dicts, will be stored in .yaml
        path (str):              Full save path.
        mode (str):              If ``mode='picasso_compatible'`` localizations will be stored as nmpy.rec_array to 
            be readable with `picasso.filter`_.
    """
    
    if mode=='picasso_compatible':
        locs_rec=locs.to_records(index=False) # Convert to rec_array  
        with h5py.File(path, "w") as store:
            store.create_dataset("locs", data=locs_rec)
    else:
        with pd.HDFStore(path,'w') as store:
            store.put('locs', locs, format='fixed')
 
    ### Save info in .yaml
    from picasso.io import save_info as save_info
    save_info(os.path.splitext(path)[0]+'.yaml',
              info,
              default_flow_style=False)
    
    return

#%%
def save_picks(locs,pick_diameter,path):
    '''
    Save pick centers in .yaml file so they can be loaded with `picasso.render`_
    
    Args:
        locs (pandas.DataFrame or numpy.array): Spot center coordinates. Must contain fields ``x`` and ``y``.
        pick_diameter (int):                    Pick diameter in pixel as in `picasso.render`_.
        path (str):              Full save path.
        
    Returns:
        dict: Converted locs with entries ``'Diameter'``, ``'Centers'`` and ``'Shape'``
        
    '''
    x=locs.x
    if type(x) != np.ndarray: x = np.array(x)
    y=locs.y
    if type(y) != np.ndarray: y = np.array(y)
    
    centers = []
   
    for index, element in enumerate(range(len(x))):
        centers.append([float(x[index]),float(y[index])])
    
    picks = {'Diameter': float(pick_diameter), 'Centers': centers, 'Shape': 'Circle'}
 
    import yaml  as yaml
    with open(path, 'w') as f:
        yaml.dump(picks,f)
    
    return picks