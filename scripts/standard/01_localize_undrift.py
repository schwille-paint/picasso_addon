#Script to call picasso_addon.localize.main()
import os
import traceback
import importlib

import picasso.io as io
import picasso_addon.localize as localize

importlib.reload(localize)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/ibidi_id180_Pm2-20nM_p40uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/ibidi_id180_Pm2-10nM_p40uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/ibidi_id180_Pm2-5nM_p40uW_1'])


file_names=[]
file_names.extend(['ibidi_id180_Pm2-20nM_p40uW_1_MMStack_Pos0.ome.tif'])
file_names.extend(['ibidi_id180_Pm2-10nM_p40uW_1_MMStack_Pos0.ome.tif'])
file_names.extend(['ibidi_id180_Pm2-5nM_p40uW_1_MMStack_Pos0.ome.tif'])



############################################ Set parameters
params ={}

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
# i=1
# path=os.path.join(dir_names[i],file_names[i])
# movie,info=io.load_movie(path)

# movie = np.array(movie,dtype=np.float32)
# movie_e = (movie - 113) * 0.49 # Convert to e-

# #%%
# ### Spot detection
# frame = 0
# box = 7

# xlim = np.array([0,100])
# ylim = np.array([0,100])

# mng, boxvals = localize.autodetect_mng(movie_e,info,box)
# y,x,ng= picasso.localize.identify_in_frame(movie[frame],
#                                             mng,
#                                             box)
# ### Preview
# f=plt.figure(num=1,figsize=[6,6])
# f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
# f.clear()
# ax=f.add_subplot(111)
# ax.imshow(movie_e[frame,:,:],cmap='gray',vmin=0,vmax=80,interpolation='nearest',origin='lower')
# ax.scatter(x,y,s=150,marker='o',color='None',edgecolor='r',lw=2)   
# ax.grid(False)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
