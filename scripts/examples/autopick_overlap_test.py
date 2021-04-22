import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import picasso.io as io
import picasso.render as render
import picasso_addon.autopick as autopick


############################################# Load raw data
dir_names=[]
dir_names.extend([r'C:\Data\p17.lbFCS2\21-04-21_EGFR_2xCTC\09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1\pick_test'])

file_names=[]
file_names.extend(['render_picked_render.hdf5'])

path=os.path.join(dir_names[0],file_names[0])
locs_in,info=io.load_locs(path)

############################################# Render
oversampling = 5 
image = render.render(locs_in,
                      info,
                      oversampling = oversampling,
                      )[1]

############################################# Identify pick centers
pick_box = 11
min_n_locs = 120
centers_image,fit = autopick.spotcenters_in_image(image,
                                                  pick_box,
                                                  min_n_locs,
                                                  fit = False)

############################################# Convert centers original coordinates
centers = autopick.coordinate_convert(centers_image,
                                      (0,0),
                                      5,
                                      )

############################################# Query locs for centers
pick_diameter = 2.5
picks_idx = autopick.query_locs_for_centers(locs_in,
                                            centers[['x','y']].values,
                                            pick_radius = pick_diameter/2,
                                            )

#%%
############################################ Get localizations corresponding to picks
'''
!!!! Localizations that are found in multiple picks simultaneously should be repeated in _picked with different group id!!!!
'''
PICKED_DTYPE = {'group':np.uint32,
                'frame':np.uint32,
                'x':np.float32,
                'y': np.float32,
                'photons':np.float32,
                'sx':np.float32,
                'sy':np.float32,
                'bg':np.float32,
                'lpx':np.float32,
                'lpy':np.float32,
                'ellipticity':np.float32,
                'net_gradient':np.float32,
                }
LOCS_DTYPE = PICKED_DTYPE.copy()
del LOCS_DTYPE['group']

LOCS_COLS = [field for field in LOCS_DTYPE]
PICKED_COLS = [field for field in PICKED_DTYPE]

def get_picked(locs_in,picks_idx):
    '''
    Get locs corresponding to picks_idx as obtained by ``query_locs_for_centers()`` and assign group ID.
    
    Args:
        locs (numpy.recarray):      Localizations as loaded by picasso.io (see picasso.localize).
        picks_idx (numpy.ndarray):  Single rows correpond to indices (as list) in locs within pick_radius around centers. 
                                    See ``query_locs_for_centers()``.
    
    Returns:
        pandas.DataFrame: Picked as in `picasso.render`_. Localizations in overlapping picks appear in each pick, 
                          i.e. same localization appears multiple times but with different group ID!
                          This means ``_picked`` can have more localizations than original ``_locs``!
    '''

    ######### Prepare locs
    ### Convert to DataFrame for easier access
    locs_df = pd.DataFrame(locs_in,dtype=np.float32)
    ### Rearrange colums according to COLS_ORDER
    locs_df = locs_df[LOCS_COLS]
    ### Convert DataFrame to numpy.array for faster computation
    locs = locs_df.values.astype(np.float32)
    
    ######### Prepare picks_idx
    ### Convert pick indices np.ndarray to list
    picks_list = picks_idx.tolist()
    ### Get flattened length of pick_list
    n = np.sum([len(l) for l in picks_list])
    
    ######## This is the important part for speedy group assignment
    ### Get one dimensional array of pick indices in locs
    ids = []
    for l in picks_list: ids.extend(l)
    ### Get one dimensional array of groups ids corresponding to pick indices in locs
    gs = [np.ones(len(l))*group for group,l in enumerate(picks_list)]
    gs = np.concatenate(gs)
    
    ### Initiate picked
    picked = np.zeros((n,len(PICKED_COLS)), dtype = np.float32)
    ### Assign all localization information to picked
    picked[:,1:] = locs[ids,:]
    picked[:,0] = gs
    
    picked_df = pd.DataFrame(picked, columns = PICKED_COLS)
    picked_df = picked_df.astype(PICKED_DTYPE)
    
    return picked_df


# ### Apply function 
picked_df = get_picked(locs_in,picks_idx)

#%%
############################################ Check if it works?? -> Yes!
import matplotlib.pyplot as plt

f = plt.figure(0,figsize = [5,5])
f.clear()
ax = f.add_subplot(111)
g = 0
ax.scatter(picked_df.query('group == @g').x,
           picked_df.query('group == @g').y,
           s = 150,
           marker='o',fc='none',ec='r')
circle=plt.Circle((centers.iloc[g].x,centers.iloc[g].y),pick_diameter/2,color='r',fc='none')
ax.add_patch(circle)

g = 1
ax.scatter(picked_df.query('group == @g').x,
           picked_df.query('group == @g').y,
           marker='x',fc='none',ec='b')
circle=plt.Circle((centers.iloc[g].x,centers.iloc[g].y),pick_diameter/2,color='b',fc='none')
ax.add_patch(circle)

g = 2
ax.scatter(picked_df.query('group == @g').x,
           picked_df.query('group == @g').y,
           s = 100,
           marker='v',fc='none',ec='g')
circle=plt.Circle((centers.iloc[g].x,centers.iloc[g].y),pick_diameter/2,color='g',fc='none')
ax.add_patch(circle)































