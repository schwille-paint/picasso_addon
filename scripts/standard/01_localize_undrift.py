#Script to call picasso_addon.localize.main()
import os
import traceback
import importlib

import picasso.io as io
import picasso_addon.localize as localize

importlib.reload(localize)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-11-23_N12_T23_CalibrationTest/id45_Pm2-10nM-Meta_p40uW_exp200_Corr-On_1'])

file_names=[]
file_names.extend(['id45_Pm2-10nM-Meta_p40uW_exp200_Corr-On_1_MMStack_Pos0.ome.tif'])

############################################ Set parameters
params ={'calibrate_movie':False,
         'mng' : 300}

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
# ############################################ Checkout single files
# import picasso.localize
# import matplotlib.pyplot as plt

# ### Load movie
# i=2
# path=os.path.join(dir_names[i],file_names[i])
# movie,info=io.load_movie(path)
# #%%
# ### Spot detection
# frame=30
# params={'mng':200,
#         'box':5}

# y,x,ng= picasso.localize.identify_in_frame(movie[frame],
#                                             params['mng'],
#                                             params['box'])
# ### Preview
# f=plt.figure(num=1,figsize=[4,4])
# f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
# f.clear()
# ax=f.add_subplot(111)
# ax.imshow(movie[frame],cmap='gray',vmin=60,vmax=200,interpolation='nearest',origin='lower')
# # ax.scatter(x,y,s=150,marker='o',color='None',edgecolor='r',lw=2)   

# ax.grid(False)
# ax.set_xlim(200,400)
# ax.set_ylim(200,400)