'''
.. _picasso.localize:
    https://picassosr.readthedocs.io/en/latest/localize.html
.. _picasso.render:
    https://picassosr.readthedocs.io/en/latest/render.html
.. _spt:
    https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1
'''

import os
import numba
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import picasso.io as io
import picasso.localize as localize
import picasso.gausslq as gausslq
import picasso.postprocess as postprocess
import picasso_addon

### Test if GPU fitting can be used
try:
    from pygpufit import gpufit as gf
    gpufit_available=True
    gpufit_available=gf.cuda_available()
except ImportError:
    gpufit_available = False


#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def boxvals_in_frame(frame, y, x, box):
    
    box_half = int(box / 2)
    
    ### Create sort of a meshgrid for net gradient calculation
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux ** 2 + uy ** 2)
    ux /= unorm
    uy /= unorm
    
    ### Initiate output
    ng = np.zeros(len(x), dtype=np.float32)
    boxsum = np.zeros(len(x), dtype=np.float32)
    
    for i, (yi, xi) in enumerate(zip(y, x)):                                                               # Loop through local maxima
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):        # Loop through rows within box
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):    # Loop through  columns within box
                
                ### Sum of pixel values in box
                boxsum[i] +=  frame[k,m]
               
                ### Net gradient
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += ( gy * uy[k_index, l_index] + gx * ux[k_index, l_index] )
    
    out = np.zeros((len(x),2), dtype=np.float32)
    out[:,0] = ng
    out[:,1] = boxsum
    
    return out

#%%
def autodetect_mng(movie,info,box):
    '''
    Automatically detect minimal net-gradient for localization. 
    Based on net-gradient distribution of all boxes in raw movie using identify_in_image().
    Distribution is taken from 10 frames evenly distributed over total aquisition time.
    
    Args:
        movie (io.TiffMultiMap): Movie object as created by picasso.io.load_movie().
        info (list): Info file as created by picasso.io.load_movie()
        box (uneven int): Box size in pixels.
        
    Returns:
		int: Minimal net gradient
    '''

    ### Select 10 evenly distributed frames
    frames_num=np.linspace(0,info[0]['Frames']-1,10).astype(int) 
    
    #### Loop over frames and calculate net gradient & sum of all boxes
    boxvals = np.zeros((1,2)) 
    for f in frames_num:
        frame = movie[f,:,:]                                                      # Select frame
        y, x = localize.local_maxima(frame, box)                     # Identify local maxima in frame
        boxvals_frame = boxvals_in_frame(frame, y, x, box)   # Net gradient [:,0] and sum [:,1] of boxes
        boxvals = np.vstack([boxvals,boxvals_frame])
    
    ### Sort according to boxsums
    idx_sort = np.argsort(boxvals[:,1])
    boxvals_sort = boxvals[idx_sort,:]
    
    ### Get lowest  (10%) boxsum values and respective net gradients
    boxvals_sort_low = boxvals_sort[:int(0.1*len(boxvals)),:]
    
    ### Set minimum net gradient to 2x(median + 4 x the standard deviation) of remaining netgradients (i.e. 10% boxsum percentile)
    mng = 4 * np.std(boxvals_sort_low[:,0]) + np.median(boxvals_sort_low[:,0])
    # mng = mng*2
    
    mng=int(np.ceil(mng/10)*10) 
    
    return mng

#%%
def cut_spots_readvar(ids,box):
    
    ### Load maps
    readvar = picasso_addon.load_readvar_map() # Readout variance
    gain = picasso_addon.load_gain_map()            # Gain
    
    ### Identify readout variances of spots corresponding to ids
    spots_readvar = np.zeros((len(ids), box, box), dtype=np.float32)                                 # Prepare spots_readvar
    readvar_convert = readvar/gain**2                                                                                # Convert ADU variance map to electrons using gain map
    localize._cut_spots_frame(readvar_convert,                                                                   # Assign spots_readvar values
                                        0,
                                        np.zeros(len(ids),dtype=np.int32),
                                        np.round(ids['x']).astype(np.int16),
                                        np.round(ids['y']).astype(np.int16),
                                        int(box / 2),
                                        0,
                                        len(ids),
                                        spots_readvar)
    
    return spots_readvar

#%%
def weightfit_spots_gpufit(spots,spots_readvar):
    
    size = spots.shape[1]
    initial_parameters = gausslq.initial_parameters_gpufit(spots, size)
    spots.shape = (len(spots), (size * size))
    spots_readvar.shape = (len(spots), (size * size))
    weights = 1/(spots+spots_readvar)                      # Weights should equal 1/var of data
    model_id = gf.ModelID.GAUSS_2D_ELLIPTIC

    parameters, states, chi_squares, number_iterations, exec_time = gf.fit(
        spots,
        weights,
        model_id,
        initial_parameters,
        tolerance=1e-2,
        max_number_iterations=20,
    )

    parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]
    
    return parameters

#%%
def localize_movie(movie,
                                 box,
                                 mng,
                                 baseline,
                                 sensitivity,
                                 qe,
                                 gain,
                                 weight_fit):
    '''
    Localize using least square fitting in either non-weighted version (CPU,GPU) or weighted version (GPU) using provided CMOS camera readout variance map.
    
    Args:
        movie (io.TiffMultiMap): Movie object as created by picasso.io.load_movie()
        box(int):                         Box length (uneven!) of fitted spots (see picasso.localize)
        mng(float):                     Minimal net-gradient spot detection threshold (see picasso.localize)
        baseline(int):                  Camera spec. baseline (see picasso.localize)
        sensitivity(float):            Camera spec. sensitivity (see picasso.localize)
        qe(float):                        Camera spec. quantum efficiency (see picasso.localize)
        gain(float):                     Camera spec. EM gain (see picasso.localize)
        weight_fit(bool):             Use weighted least square fitting based on readout variance and gain maps(only GPU implemented!)
        
    
    Returns:
        locs: Localizations (numpy.recarray)
    '''

    ############################ Set camera specs
    camera_info = {}
    camera_info['baseline'] = baseline
    camera_info['sensitivity'] = sensitivity
    camera_info['qe'] = qe
    camera_info['gain'] = gain
    em = gain > 1 # Check if EMCCD
    
    ############################ Spot detection
    print('Identifying spots ...')
    current,futures=localize.identify_async(movie,
                                                                   mng,
                                                                   box,
                                                                   None)

    identifications=localize.identifications_from_futures(futures)
    spots=localize.get_spots(movie,
                                             identifications,
                                             box,
                                             camera_info)

    ############################ Least-square fitting    
    if gpufit_available:
        if weight_fit:
            print('Identifying spot readout variances ... ')
            spots_readvar = cut_spots_readvar(identifications, box)
            print('Weighted least square fitting (GPU) ...')
            theta = weightfit_spots_gpufit(spots,spots_readvar)
        
        else:
            print('Non-weighted least square fitting (GPU) ...')
            theta = gausslq.fit_spots_gpufit(spots)
            
        locs = gausslq.locs_from_fits_gpufit(identifications,
                                                                  theta,
                                                                  box,
                                                                  em)
    else:
        print('Non-weighted least square fitting (CPU) ..')
        fs = gausslq.fit_spots_parallel(spots,asynch=True)
        theta = gausslq.fits_from_futures(fs)   
        locs = gausslq.locs_from_fits(identifications,
                                                      theta,
                                                      box,
                                                      em)
        
    assert len(spots) == len(locs)
    
   ############################  Check for any negative values in photon, bg, sx, sy values
    positives = (locs.photons>0) & (locs.bg>0) & (locs.sx>0) & (locs.sy>0)
    locs = locs[positives]
    
    
    return locs

#%%
def main(file,info,path,**params):
    '''
    Localize movie (least squares, GPU fitting if available) and undrift resulting localizations using rcc.
     
    Args:
        file(picasso.io):          Either raw movie loaded with picasso.io.load_movie() or_locs.hdf5 loaded with picasso.io.load_locs()
        info(list(dicts)):         Info to raw movie/_locs.hdf5 loaded with picasso.io.load_movie() or picasso.io.load_locs()
    
    Keyword Arguments:
        calibrate_movie(bool=True): Pixel calibration of raw movie using offset and gain maps for CMOS sensors
        localize(bool=True):        Localize raw or calibrated movie (CMOS) (see picasso_addon.localize)
        box(int=9):                 Box length (uneven!) of fitted spots (see picasso.localize)
        mng(int or str='auto'):     Minimal net-gradient spot detection threshold(see picasso.localize. If set to 'auto' minimal net_gradient is determined by autodetect_mng().
        baseline(int=113):          Camera spec. baseline (see picasso.localize).
        gain(float=1):              Camera spec. EM gain (see picasso.localize)
        sensitivity(float=0.49):    Camera spec. sensitivity (see picasso.localize)
        qe(float=1):                Camera spec. quantum gain (see picasso.localize), set to 1 to count generated photoelectrons.
        
        undrift(bool=True):              Apply RCC drift correction (see picasso.postprocess)
        segments(int=450 or str='auto'): Segment length (frames) for undrifting by RCC (see picasso.render). If set to 'auto' segment length is ```int(ceil(NoFrames))```.
    
    Returns: 
        list:
        - [0][0](dict):     kwargs passed to localize_movie()
        - [0][1](dict):     kwargs passed to undriftrcc_locs()
        - [1](numpy.array): Undrifted localization list saved as _render.hdf5 (if ``undrift`` was ``True``) otherwise localization list (no undrifting) saved as _locs.hdf5
        - [2](str):         Last full file saving path for input to further modules
    '''
    ############################# Set standard paramters if not given with function call
    standard_params={'use_maps': False,
                                    'localize':True,
                                    'box':5,
                                    'mng':'auto',
                                    'baseline':113,
                                    'sensitivity':0.49,
                                    'qe':1,
                                    'gain':1,
                                    'weight_fit':False,
                                    'undrift':True,
                                    'segments':450,
                                   }
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key] == None: params[key] = standard_params[key]
        except:
            params[key] = standard_params[key]
    
    #############################  Path of file that is processed
    print()
    path=os.path.splitext(path)[0]
    
    ############################# Localize
    if params['localize']==True:
        
        file = np.array(file,dtype=np.float32) # Convert from io.TiffMultiMap to numpy.array
        
        if params['use_maps']: ### Calibrate movie using px-maps (CMOS)
            offset = picasso_addon.load_offset_map()   # Load offset map
            gain   = picasso_addon.load_gain_map()     # Load gain map
            print('Converting movie to e- using maps ...')
            file = (file - offset[np.newaxis,:,:]) / gain[np.newaxis,:,:]  # Calibrate movie
        
        else: ### Just conversion to e- using standard settings
            offset = params['baseline']           # Equal offset for every px
            gain   = (1/params['sensitivity'])   # Equal gain for every px
            print('Converting movie to e- ...')
            file = (file - offset) / gain  # Substract median offset and multiply by median sensitivity
            
        ### Autodetect mng
        if params['mng'] == 'auto':
            params['mng'] = autodetect_mng(file,info,params['box'])
            params['auto_mng'] = True
            print('Minimum net-gradient set to %i'%(params['mng']))
           
        ### Localize movie
        locs = localize_movie(file,
                              params['box'],
                              params['mng'],
                              0,         # Baseline set to 0 since movie already converted to e-
                              1,         # Sensitivity set to 1 since movie already converted to e-
                              params['qe'],
                              params['gain'],
                              params['weight_fit']
                              )
        
        ### Save _locs and yaml
        print('Saving _locs ...')
        params_localize = params.copy() 
        for key in ['undrift','segments']: params_localize.pop(key) # Remove keys for undrifting
        info = info + [params_localize]  # Update info
        next_path=path+'_locs.hdf5' # Update path
        io.save_locs(next_path,
                     locs,
                     info,
                     )
        
        ### Update path
        next_path = os.path.splitext(next_path)[0] # Remove file extension
        
    else:
        locs = file
        next_path = path 
    
    ############################## Undrift
    if params['undrift']==True:
        if params['segments'] == 'auto':
            NoFrames = info[0]['Frames']
            params['segments'] = int(np.ceil(NoFrames/10))
        try:
            drift, locs_undrift = postprocess.undrift(locs,
                                                      info,
                                                      params['segments'],
                                                      display=False)
            ### Save _locs_render and yaml
            print('Saving _render ...')
            info= info + [ {'segments':params['segments']} ] # Update info
            next_path = next_path + '_render.hdf5' # Update path
            io.save_locs(next_path,
                         locs_undrift,
                         info,
                         )
        except:
            print('Undrifting by RCC was not possible')
            locs_undrift = locs
            
    else:
          print('No undrifting')
          locs_undrift = locs
          
    return [info,locs_undrift]
