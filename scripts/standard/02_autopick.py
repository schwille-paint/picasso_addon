#Script to call picass_addon.autopick main()
import os
import traceback
import importlib
import matplotlib.pyplot as plt

import picasso.io as io
import picasso.render as render
import picasso_addon.autopick as autopick

importlib.reload(autopick)

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
# file_names.extend(['id63_2-5nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_2-5nM_p35uW_2_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_5nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_5nM_p35uW_2_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_10nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_10nM_p35uW_2_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_20nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id63_20nM_p35uW_2_MMStack_Pos0.ome_locs_render.hdf5'])
###################### N=4
# file_names.extend(['id125_05nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id125_10nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
# file_names.extend(['id125_20nM_p35uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
###################### N=1
file_names.extend(['id114_5nM_p35uW_control_1_MMStack_Pos0.ome_locs_render.hdf5'])
file_names.extend(['id114_10nM_p35uW_control_1_MMStack_Pos0.ome_locs_render.hdf5'])
file_names.extend(['id114_20nM_p35uW_control_1_MMStack_Pos0.ome_locs_render.hdf5'])


############################################ Set parameters
params={'lbfcs':True}

#%%                   
failed_path=[]
for i in range(0,len(file_names)):
    ### Create path
    path=os.path.join(dir_names[i],file_names[i])
    ### Run main function
    try:
        locs,info=io.load_locs(path)
        out=autopick.main(locs,info,path,**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))





#%%
############################################ Checkout single files
# i=0

# params={'oversampling':5,
#         'pick_box':11, # Usually 2*oversampling+1
#         'min_n_locs':225,
#         'fit_center':False,
#         'pick_diameter':2,
#         }

# ### Load file
# path=os.path.join(dir_names[i],file_names[i])
# locs,info=io.load_locs(path)

# ### Render
# image=render.render(locs,
#                     info,
#                     params['oversampling'],
#                     )[1] 
# ### Pick detection
# centers=autopick.spotcenters_in_image(image,
#                                       params['pick_box'],
#                                       params['min_n_locs'],
#                                       params['fit_center'])[0]
#%%
### Preview
# f=plt.figure(num=1,figsize=[4,4])
# f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
# f.clear()
# ax=f.add_subplot(111)

# ax.imshow(image,cmap='gray',vmin=0,vmax=100,interpolation='nearest',origin='lower')
# for i in range(0,len(centers)):
#     circle=plt.Circle((centers.loc[i,'x'],centers.loc[i,'y']),
#                 params['pick_diameter']/2*params['oversampling'],
#                 facecolor='None',
#                 edgecolor='y',
#                 lw=2)
#     ax.add_artist(circle)

# ax.grid(False)
