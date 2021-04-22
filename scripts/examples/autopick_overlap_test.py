import os
import numpy as np
import pandas as pd

import picasso.io as io
import picasso.render as render
import picasso_addon.autopick as autopick


############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1/pick_test'])

file_names=[]
file_names.extend(['render_picked_render.hdf5'])

path=os.path.join(dir_names[0],file_names[0])
locs,info=io.load_locs(path)

############################################# Render
oversampling = 5 
image = render.render(locs,
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
picks_idx = autopick.query_locs_for_centers(locs,
                                            centers[['x','y']].values,
                                            pick_radius = pick_diameter/2,
                                            )

#%%
############################################ Get localizations corresponding to picks
'''
!!!! Localizations that are found in multiple picks simultaneously should be repeated in _picked with different group id!!!!
'''
LOCS_DTYPE = [("frame", "u4"),
              ("x", "f4"),
              ("y", "f4"),
              ("photons", "f4"),
              ("sx", "f4"),
              ("sy", "f4"),
              ("bg", "f4"),
              ("lpx", "f4"),
              ("lpy", "f4"),
              ('ellipticity', '<f4'),
              ("net_gradient", "f4"),
              ]
PICKED_DTYPE = [('group','u4')] + [dt for dt in LOCS_DTYPE]
COLS_ORDER = [val[0] for val in PICKED_DTYPE]

### Convert to DataFrame for easier access
locs_df = pd.DataFrame(locs)
### Assign group column with value -1 to all locs
locs_df = locs_df.assign(group = -1)
### Rearrange colums according to COLS_ORDER
locs_df = locs_df[COLS_ORDER]
### Convert DataFrame back to numpy.array for faster computation
locs_array = locs_df.values

picked = np.zeros([],dtype=np.float64)

for g,idx in enumerate(picks_idx):
    picked_single = locs_array[idx,:]
    picked_single
#     picked = np.hstack((picked,picked_single))


































