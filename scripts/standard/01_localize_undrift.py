#Script to call picasso_addon.localize.main()
import os
import traceback
import importlib

import picasso.io as io
import picasso_addon.localize as localize

importlib.reload(localize)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1']*5)

file_names=[]
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos0.ome.tif'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos1.ome.tif'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos2.ome.tif'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos3.ome.tif'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos4.ome.tif'])

############################################ Set parameters
params = {}
# params ={'mng':250}

#%%
############################################ Main loop
paths=[os.path.join(dir_names[i],file_name) for i, file_name in enumerate(file_names)]
failed_path=[]

for path in paths:
    ### Try loading movie or localizations
    try:
        try:
            file,info=io.load_movie(path) # Load movie
        except:
            file,info=io.load_locs(path) # Load _locs
        out=localize.main(file,info,path,**params)
    
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))






#%%
############################################ Checkout single files
# import numpy as np
# import matplotlib.pyplot as plt
# import picasso.localize

# ### Load movie
# i=0
# path=os.path.join(dir_names[i],file_names[i])
# movie,info=io.load_movie(path)

# movie = np.array(movie,dtype=np.float32)
# movie_e = (movie - 113) * 0.49 # Convert to e-

#%%
# ## Spot detection
# frame = 2000
# box = 5

# xlim = np.array([0,200]) + 80
# ylim = np.array([0,200]) + 300

# mng = localize.autodetect_mng(movie_e,info,box)
# mng = 2*mng
# print(mng)

# y,x,ng= picasso.localize.identify_in_frame(movie[frame],
#                                             mng,
#                                             box)

# ### Preview
# f=plt.figure(num=1,figsize=[6,6])
# f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
# f.clear()
# ax=f.add_subplot(111)
# ax.imshow(movie_e[frame,:,:],cmap='gray',vmin=-10,vmax=200,interpolation='nearest',origin='upper')
# ax.scatter(x,y,s=200,marker='o',color='None',edgecolor='r',lw=1)
# ax.grid(False)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
