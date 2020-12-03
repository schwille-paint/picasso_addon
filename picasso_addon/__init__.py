'''
    picasso_addon/__init__.py
    ~~~~~~~~~~~~~~~~~~~~

    :authors: Stehr Florian

'''   
import os
import h5py

_this_file = os.path.abspath(__file__)         # Absolute path to __init__.py
_this_dir = os.path.dirname(_this_file)       # Absolute path to source code
_map_dir = os.path.join(_this_dir,'maps')  # Absolute path to camera calibration maps directory


def load_offset_map():
    with h5py.File(os.path.join(_map_dir,'offset.hdf5'), "r") as store:
        offset= store['data'][...]
        return offset

def load_gain_map():
    with h5py.File(os.path.join(_map_dir,'gain.hdf5'), "r") as store:
        gain= store['data'][...]
        return gain
    
def load_readvar_map():
    with h5py.File(os.path.join(_map_dir,'readvar.hdf5'), "r") as store:
        readvar= store['data'][...]
        return readvar