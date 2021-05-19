import os
import numpy as np
import pandas as pd
import h5py
import yaml
import picasso.io as io

LINE_TYPE = 'Line2D'
IMAGE_TYPE = 'AxesImage'

#%%
def save_data(path,data):
    with h5py.File(path, 'w') as data_file:
        data_file.create_dataset('data', data=data)

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
def dump_lines(lines,path,legend_labels):
    for line in lines:
            data = line._xy     # Get data
            label = line._label # Get label
            
            if label in legend_labels: # Check if label is in legend
                label = remove_badchar(label) # Remove unwanted characters
                save_path = os.path.join(path,label+'.hdf5')
                save_data(save_path,data)

#%%
def dump_images(images,path):
    for image in images:
            data = image._A         # Get data
            label = image._label    # Get label
            
            label = remove_badchar(label) # Remove unwanted characters
            save_path = os.path.join(path,label+'.hdf5')
            save_data(save_path,data)

#%%
def dump_collections(collections,path,legend_labels):
    for c in collections:
            data = c._offsets     # Get data
            label = c._label # Get label
            
            if label in legend_labels: # Check if label is in legend
                label = remove_badchar(label) # Remove unwanted characters
                save_path = os.path.join(path,label+'.hdf5')
                save_data(save_path,data)
                
#%%
def dump_patches(patches,path,legend_labels):
    xs = [(p._x1-p._x0)/2 for p in patches]
    ys = [p._y1 for p in patches]
    labels = [p._label for p in patches]
    data = pd.DataFrame({'label':labels,'x':xs,'y':ys})
    
    return data
    
#%%
def dump_data_from_axes(ax,scriptpath):
    '''
    Function to dump all data contained in matplotlib.axes object into directory of script.
    Data is save according to labels.
    '''
    
    ### Retrieve path to directory of script 
    path = os.path.dirname(os.path.realpath(scriptpath))
    
    ### Get lines,images, patches(histogram) and collections(scatter)
    lines = ax.get_lines()
    images = ax.get_images()
    collections = ax.collections
    patches = ax.patches
    
    ### Get x and y limits and dump in yaml
    info = [{'xlowlim':float(ax.get_xlim()[0]),
             'xuplim':float(ax.get_xlim()[1]),
             'ylowlim':float(ax.get_ylim()[0]),
             'yuplim':float(ax.get_ylim()[0]),
            }]
    io.save_info(os.path.join(path,'axinfo.yaml'),info)
    
    ### Get legend labels
    legend_labels = ax.get_legend_handles_labels()[1]
    
    ### Dump lines
    if len(lines) > 0: # Check if not empty
        dump_lines(lines,path,legend_labels)
     
    ### Dump images
    if len(images) > 0: # Check if not empty
        dump_images(images,path)
    
    ### Dump collections
    if len(collections) > 0: # Check if not empty
        dump_collections(collections,path,legend_labels)
    
    
    return [lines, images, collections, patches], info, path
    
    