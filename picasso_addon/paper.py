import os
from pathlib import Path
import numpy as np
import h5py
import yaml



#%%
def save_data(path,data):
    
    with h5py.File(path, 'w') as data_file:
        data_file.create_dataset('data', data=data)

#%%
def save_info(path,info):
    
    with open(path, "w") as info_file:
        dump = yaml.dump(info, default_flow_style = None,)
        info_file.write(dump)
        
#%%
def load_data(path):
    
    with h5py.File(path, 'r') as data_file:
        data = data_file['data'][...]
    return data

#%%
def remove_badchar(label):
    
    bad_char = ['$','}','{','\\']
    label = ''.join(char for char in label if char not in bad_char)
    return label

#%%
def dump_lines(lines,path,ax_id):
    
    for line in lines:
        data = line._xy     # Get data
        label = line._label # Get label
        
        # Dump
        label = remove_badchar(label) # Remove unwanted characters
        save_path = os.path.join(path,'%i_'%ax_id+label+'.hdf5')
        save_data(save_path,data)

#%%
def dump_images(images,path,ax_id):
    
    for image in images:
        data = image._A         # Get data
        label = image._label    # Get label
        
        # Dump
        label = remove_badchar(label) # Remove unwanted characters
        save_path = os.path.join(path,'%i_'%ax_id+label+'.hdf5')
        save_data(save_path,data)

#%%
def dump_collections(collections,path,ax_id):
    
    for c in collections:
        data_xy = c._offsets    # Get xy data
        data_z = c.get_array()  # Get z data (e.g. color in scatter plot)
        data = np.zeros((len(data_xy),3))  # Combine to numpy array
        data[:,0:2] = data_xy
        data[:,2] = data_z
        
        label = c._label   # Get label
        
        # Dump
        label = remove_badchar(label) # Remove unwanted characters
        save_path = os.path.join(path,'%i_'%ax_id+label+'.hdf5')
        save_data(save_path,data)
                
#%%
def dump_containers(containers,path,ax_id):
    
    for c in containers:
        data_x = [(p._x0+p._x1)/2 for p in c.patches] # Get x positions as list
        data_y = [p._y1 for p in c.patches]           # Get y positions as list
        
        data = np.zeros((len(data_x),2))  # Combine to numpy array
        data[:,0] = data_x
        data[:,1] = data_y
        
        label = c._label    # Get label
        if label[0] == '_': # In case of histogram BarContainer misses the correct label
            label = c.patches[0]._label
        
        # Dump
        label = remove_badchar(label) # Remove unwanted characters
        save_path = os.path.join(path,'%i_'%ax_id+label+'.hdf5')
        save_data(save_path,data)
    
    
#%%
def dump_data_from_ax(ax,ax_id,path):
    
    ### Get lines(plot,step),images(imshow),collections(scatter,fill_between) and containers (hist,bar)
    lines = ax.get_lines()
    images = ax.get_images()
    collections = ax.collections
    containers = ax.containers
    
    ### Create info dictionary about ax
    xlim = [float(ax.get_xlim()[0]),float(ax.get_xlim()[1])]
    ylim = [float(ax.get_ylim()[0]),float(ax.get_ylim()[1])]
    info = {'xlim':xlim,
            'ylim':ylim,
            }
    
    ### Save info about ax as .yaml
    save_info(os.path.join(path,'%i_'%ax_id+'axinfo.yaml'),info)
    
    ### Get legend labels
    legend_labels = ax.get_legend_handles_labels()[1]
    
    ### Dump lines
    if len(lines) > 0: # Check if not empty
        dump_lines(lines,path,ax_id)
     
    ### Dump images
    if len(images) > 0: # Check if not empty
        dump_images(images,path,ax_id)
    
    ### Dump collections
    if len(collections) > 0: # Check if not empty
        dump_collections(collections,path,ax_id)
    
    ### Dump containers
    if len(containers) > 0: # Check if not empty
        dump_containers(containers,path,ax_id)
    
    
    return [lines, images, collections, containers], info
    
#%%
def dump_data_from_fig(fig,scriptpath,subfolder_name = ''):
    
    ### Retrieve path to directory of script 
    path = os.path.dirname(os.path.realpath(scriptpath))
    
    ### Create new subfolder to save data in script directory
    path = os.path.join(path,subfolder_name)
    Path(path).mkdir(parents=False, exist_ok=True)
    
    data_axes = []
    info_axes = []
    
    for ax_id,ax in enumerate(fig.axes):
        data, info = dump_data_from_ax(ax,ax_id,path)
        data_axes.extend(data)
        info_axes.extend(data)
        
    return data_axes, info_axes, path
        
    