#Script to call picasso_addon.localize.main()
import os
import traceback
import importlib
import matplotlib.pyplot as plt

import picasso.io as io
import picasso.localize
import picasso_addon.localize as localize

importlib.reload(localize)
############################################# Load raw data
dir_names=[]

###################### N=12
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_2-5nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_2-5nM_p35uW_2'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_5nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_5nM_p35uW_2'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_10nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_10nM_p35uW_2'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_20nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_20nM_p35uW_2'])
###################### N=4
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-16_N=4/id125_05nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-16_N=4/id125_10nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-16_N=4/id125_20nM_p35uW_1'])
###################### N=1
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_5nM_p35uW_control_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_10nM_p35uW_control_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_20nM_p35uW_control_1'])


file_names=[]

###################### N=12
# file_names.extend(['id63_2-5nM_p35uW_1_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_2-5nM_p35uW_2_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_5nM_p35uW_1_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_5nM_p35uW_2_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_10nM_p35uW_1_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_10nM_p35uW_2_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_20nM_p35uW_1_MMStack_Pos0.ome.tif'])
# file_names.extend(['id63_20nM_p35uW_2_MMStack_Pos0.ome.tif'])
###################### N=4
# file_names.extend(['id125_05nM_p35uW_1_MMStack_Pos0.ome.tif'])
# file_names.extend(['id125_10nM_p35uW_1_MMStack_Pos0.ome.tif'])
# file_names.extend(['id125_20nM_p35uW_1_MMStack_Pos0.ome.tif'])
###################### N=1
file_names.extend(['id114_5nM_p35uW_control_1_MMStack_Pos0.ome.tif'])
file_names.extend(['id114_10nM_p35uW_control_1_MMStack_Pos0.ome.tif'])
file_names.extend(['id114_20nM_p35uW_control_1_MMStack_Pos0.ome.tif'])

############################################ Set non standard parameters 
### Valid for all evaluations
params_all={}
### Exceptions
params_special={}

#%%
############################################ Main loop                   
failed_path=[]
for i in range(0,len(file_names)):
    ### Create path
    path=os.path.join(dir_names[i],file_names[i])
    ### Set paramters for each run
    params=params_all.copy()
    for key, value in params_special.items():
        params[key]=value[i]
    
    ### Run main function
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

### Load movie
# i=0
# path=os.path.join(dir_names[i],file_names[i])
# movie,info=io.load_movie(path)
# #%%
# ### Spot detection
# frame=10
# params={'mng':1000,
#         'box':5}

# y,x,ng= picasso.localize.identify_in_frame(movie[frame],
#                                            params['mng'],
#                                            params['box'])
# ### Preview
# f=plt.figure(num=1,figsize=[4,4])
# f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
# f.clear()
# ax=f.add_subplot(111)
# ax.imshow(movie[frame],cmap='gray',vmin=60,vmax=400,interpolation='nearest',origin='lower')
# ax.scatter(x,y,s=150,marker='o',color='None',edgecolor='r',lw=2)   

# ax.grid(False)
# ax.set_xlim(200,400)
# ax.set_ylim(200,400)