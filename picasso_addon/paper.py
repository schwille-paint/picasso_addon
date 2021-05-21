import os
from pathlib import Path
import numpy as np
import h5py
import yaml
import matplotlib as mpl


        
#%%
def hdf5_get_info(path):
    print(path)
    with h5py.File(path,'r') as f:
        for name in f:
            print()
            print('Group:'+name)
            
            print('   Attributes:')
            for k in f[name].attrs: print('      '+k)
            
            print('   Datasets:')
            for k in f[name].keys():
                dest = name + '/' + k
                shape = f[dest].shape
                print('      ' + dest + ': ' + str(shape))

#%%
def hdf5_load_dataset(path,dest):
    with h5py.File(path,'r') as f:
        ds = np.array(f[dest])
    return ds
#%%
def remove_badchar(label):
    
    bad_char = ['$','}','{','\\']
    label = ''.join(char for char in label if char not in bad_char)
    return label

#%%
def dump_lines(lines):
    
    datas = []
    labels =  []
    for line in lines:
        data = np.array(line._xy)     # Get data
        label = line._label # Get label
        
        # Dump
        if label != '_nolegend_':
            #Adjust label
            label = remove_badchar(label) # Remove unwanted characters
            
            datas.extend([data])
            labels.extend([label])
            
    return labels, datas

#%%
def dump_patches(patches):
    
    # For histograms with histype step
    datas = []
    labels =  []
    for patch in patches:
        if isinstance(patch,mpl.patches.Polygon):
            data = np.array(patch.xy)     # Get data
            label = patch._label # Get label
            # Dump
            if label != '_nolegend_':
                #Adjust label
                label = remove_badchar(label) # Remove unwanted characters
                
                datas.extend([data])
                labels.extend([label])
            
    return labels, datas
    
#%%
def dump_images(images):
    
    datas = []
    labels =  []
    for image in images:
        data = np.array(image._A)         # Get data
        label = image._label    # Get label
        
        # Dump
        if label != '_nolegend_':
            #Adjust label
            label = remove_badchar(label) # Remove unwanted characters
            
            datas.extend([data])
            labels.extend([label])
            
    return labels, datas

#%%
def dump_collections(collections):
    
    datas = []
    labels =  []
    for c in collections:
        data_xy = np.array(c._offsets)    # Get xy data
        data_z = np.array(c.get_array())  # Get z data (e.g. color in scatter plot)
        data = np.zeros((len(data_xy),3))  # Combine to numpy array
        data[:,0:2] = data_xy
        data[:,2] = data_z
        
        label = c._label   # Get label
        
        # Dump
        if label != '_nolegend_':
            #Adjust label
            label = remove_badchar(label) # Remove unwanted characters
            
            datas.extend([data])
            labels.extend([label])
            
    return labels, datas
                
#%%
def dump_containers(containers):
    
    datas = []
    labels =  []
    for c in containers:
        
        ### Retrieve hist and bar data
        if isinstance(c,mpl.container.BarContainer):
            data_x = [(p._x0+p._x1)/2 for p in c.patches] # Get x positions as list
            data_y = [p._y1 for p in c.patches]           # Get y positions as list
            
            data = np.zeros((len(data_x),2))  # Combine to numpy array
            data[:,0] = data_x
            data[:,1] = data_y
        
        ### Retrieve errobar data
        elif isinstance(c,mpl.container.ErrorbarContainer):
            data_xy = c[0]._xy
            
            data = np.zeros((len(data_xy),4))
            data[:,:2] = data_xy
            
            if not isinstance(c[1],tuple):
                data_yerr_low = c[1][0]._y
                data_yerr_up = c[1][1]._y
                data[:,2] = np.array(data_yerr_low)
                data[:,3] = np.array(data_yerr_up)
            
        ### Get label
        label = c._label    # Get label
        if label[0] == '_': # In case of histogram BarContainer misses the correct label
            label = c.patches[0]._label
                
        # Dump
        if label != '_nolegend_':
            #Adjust label
            label = remove_badchar(label) # Remove unwanted characters
            
            datas.extend([data])
            labels.extend([label])
            
    return labels, datas
    
    
#%%
def dump_data_from_ax(ax):
    
    ### Get lines(plot,step),images(imshow),collections(scatter,fill_between) and containers (hist,bar)
    lines = ax.get_lines()
    patches = ax.patches
    images = ax.get_images()
    collections = ax.collections
    containers = ax.containers
    
    ### Create info dictionary about ax
    xlim = [float(ax.get_xlim()[0]),float(ax.get_xlim()[1])]
    ylim = [float(ax.get_ylim()[0]),float(ax.get_ylim()[1])]
    info = {'xlim':xlim,
            'ylim':ylim,
            }
    
    labels = []
    datas = []
    
    ### Dump lines
    if len(lines) > 0: # Check if not empty
        label, data = dump_lines(lines)
        for l in label: labels.extend([l])
        for d in data: datas.extend([d])
    
    ### Dump patches
    if len(patches) > 0: # Check if not empty
        label, data = dump_patches(patches)
        for l in label: labels.extend([l])
        for d in data: datas.extend([d])
        
    ### Dump images
    if len(images) > 0: # Check if not empty
        label, data = dump_images(images)
        for l in label: labels.extend([l])
        for d in data: datas.extend([d])
    
    ### Dump collections
    if len(collections) > 0: # Check if not empty
        label, data = dump_collections(collections)
        for l in label: labels.extend([l])
        for d in data: datas.extend([d])
    
    ### Dump containers
    if len(containers) > 0: # Check if not empty
        label, data = dump_containers(containers)
        for l in label: labels.extend([l])
        for d in data: datas.extend([d])
    
    
    return labels, datas, info
    
#%%
def dump_data_from_fig(fig,scriptpath,save_name = 'fig'):
    
    ### Retrieve path to directory of script 
    path = os.path.dirname(os.path.realpath(scriptpath))
    
    ### Create new subfolder to save data in script directory
    path = os.path.join(path,save_name + '.hdf5')
    
    labels = []
    datas = []
    infos = []
    for ax_id,ax in enumerate(fig.axes):
        label, data, info = dump_data_from_ax(ax)
        
        ### Add ax_id to info
        info['ax_id'] = 'ax%i'%ax_id
        
        labels.append(label)
        datas.append(data)
        infos.append(info)
    
    ### Save in .hdf5 with
    with h5py.File(path,'w') as f:
        
        ### Create group for each axis
        for i,info in enumerate(infos):
            grp = f.create_group(info['ax_id'])
            
            # Add ax info as attributes
            for k in info.keys(): grp.attrs[k] = info[k]
            
            ### Create datasets within group
            for j,label in enumerate(labels[i]):
                grp.create_dataset(name = label, data = datas[i][j])
    
    return  labels, datas, infos, path
        
    