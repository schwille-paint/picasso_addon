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
def identify_in_image(image, box):
    '''
    Calculate net-gradient of all boxes in one image that have a local maximum in its center pixel.
    
    Args:
        image (numpy.array): One image of movie.
        box (int): Box size in pixels. Must be uneven!
        
    Returns:
        numpy.array: Net-gradient of all boxes thath have a local maximum in its center pixel
    '''
    y, x = localize.local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux ** 2 + uy ** 2)
    ux /= unorm
    uy /= unorm
    ng = localize.net_gradient(image, y, x, box, uy, ux)
    return ng

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

    ### Get the distribution of netgradients in all boxes for 10 frames evenly distributed over total aquisition time.
    frames=np.linspace(0,info[0]['Frames']-1,10).astype(int) # Select 10 frames
    
    ng_all=np.zeros(1) # Loop over 10 frames and get net-gradients of all boxes within
    for f in frames:
        ng=identify_in_image(movie[f].astype(float),box)
        ng_all=np.hstack([ng_all,ng])
    
    ### Choose minimal net-gradient. Idea is to get the upper boundary of net-gradients 
    ### corresponding to 'black' boxes, i.e backround noise
    
    ### First roughly cut out ower part of net-gradient distribution
    ng_median=np.percentile(ng_all,50)
    positives=(ng_all<=10*ng_median)&(ng_all>=-10*ng_median)
    ng_all=ng_all[positives]
    
    ### Iteratively redefine the cut-out of the lower distribution:
    ### Get median and interquartile width -> Define filter based on these quantities -> Cut out -> Repeat from beginning
    for i in range(0,3):
        ng_median=np.percentile(ng_all,50)
        ng_width=np.percentile(ng_all,75)-np.percentile(ng_all,25)
        ng_crit=2.5*ng_width
        positives=(ng_all<=(ng_median+ng_crit))&(ng_all>=(ng_median-ng_crit))
        ng_all=ng_all[positives]
        
    ng_median=np.percentile(ng_all,50)
    ng_width=np.percentile(ng_all,75)-np.percentile(ng_all,25)
    
    mng=ng_median+2.5*ng_width
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
def localize_movie(movie,**params):
    '''
    Localize using least square fitting in either non-weighted version (CPU,GPU) or weighted version (GPU) using provided CMOS camera readout variance map.
    
    Args:
        movie (io.TiffMultiMap): Movie object as created by picasso.io.load_movie().
    
    Keyword Args:
        calibrate_movie(bool=True)      Pixel calibration of raw movie using offset and gain maps for CMOS sensors
        localize(bool=True):                  Localize raw or calibrated movie (CMOS) (see picasso_addon.localize)
        box(int=9):                                Box length (uneven!) of fitted spots (see picasso.localize)
        mng(int or str='auto'):               Minimal net-gradient spot detection threshold(see picasso.localize. If set to 'auto' minimal net_gradient is determined by autodetect_mng().
        baseline(int=113):                     Camera spec. baseline (see picasso.localize).
        gain(float=1):                            Camera spec. EM gain (see picasso.localize)
        sensitivity(float=0.49):              Camera spec. sensitivity (see picasso.localize)
    
    Returns:
        list: 
		- [0] (dict):               kwargs passed to function
        - [1] (numpy.array): Localizations
    '''
    ### Set standard paramters if not given with function call
    standard_params = {'calibrate_movie':False,
                                      'baseline':113,
                                      'gain':1,
                                      'sensitivity':0.49,
                                      'qe':1,
                                      'box':9,
                                      'mng':400,
                                      }
    ### Check if key in params, if not set to standard_params
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key] == None: params[key]=standard_params[key]
        except:
            params[key] = standard_params[key]
    
    ### Remove keys that are not needed
    delete_key = []
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
        
    ### Procsessing marks:
    params['generatedby'] = 'picasso_addon.localize.localize_movie()'
    
    ############################ Set camera specs
    camera_info = {}
    camera_info['baseline'] = params['baseline']
    camera_info['gain'] = params['gain']
    camera_info['sensitivity'] = params['sensitivity']
    camera_info['qe'] = params['qe']
    em = camera_info['gain'] > 1 # Check if EMCCD
    
    ############################ Spot detection
    print('Identifying spots ...')
    current,futures=localize.identify_async(movie,
                                            params['mng'],
                                            params['box'],
                                            None)

    identifications=localize.identifications_from_futures(futures)
    spots=localize.get_spots(movie,
                             identifications,
                             params['box'],
                             camera_info)

    ############################ Least-square fitting    
    if gpufit_available:
        if params['calibrate_movie']:
            print('Identifying spot readout variances ... ')
            spots_readvar = cut_spots_readvar(identifications, params['box'])
            print('Weighted least square fitting (GPU) ...')
            theta = weightfit_spots_gpufit(spots,spots_readvar)
        
        else:
            print('Non-weighted least square fitting (GPU) ...')
            theta = gausslq.fit_spots_gpufit(spots)
            
        locs = gausslq.locs_from_fits_gpufit(identifications,
                                           theta,
                                           params['box'],
                                           em)
    else:
        print('Non-weighted least square fitting (CPU) ..')
        fs = gausslq.fit_spots_parallel(spots,asynch=True)
        theta = gausslq.fits_from_futures(fs)   
        locs = gausslq.locs_from_fits(identifications,
                                    theta,
                                    params['box'],
                                    em)
        
    assert len(spots) == len(locs)
    
    ############################ Remove localizations with photon values below median background signal
    locs = locs[locs.photons > np.median(locs.bg)]
    
    return [params,locs]

#%%
def main(file,info,path,**params):
    '''
    Localize movie (least squares, GPU fitting if available) and undrift resulting localizations using rcc.
     
    Args:
        file(picasso.io):          Either raw movie loaded with picasso.io.load_movie() or_locs.hdf5 loaded with picasso.io.load_locs()
        info(list(dicts)):         Info to raw movie/_locs.hdf5 loaded with picasso.io.load_movie() or picasso.io.load_locs()
    
    Keyword Arguments:
        calibrate_movie(bool=True)      Pixel calibration of raw movie using offset and gain maps for CMOS sensors
        localize(bool=True):                  Localize raw or calibrated movie (CMOS) (see picasso_addon.localize)
        box(int=9):                                Box length (uneven!) of fitted spots (see picasso.localize)
        mng(int or str='auto'):               Minimal net-gradient spot detection threshold(see picasso.localize. If set to 'auto' minimal net_gradient is determined by autodetect_mng().
        baseline(int=113):                     Camera spec. baseline (see picasso.localize).
        gain(float=1):                            Camera spec. EM gain (see picasso.localize)
        sensitivity(float=0.49):              Camera spec. sensitivity (see picasso.localize)
        qe(float=1):                               Camera spec. quantum gain (see picasso.localize), set to 1 to count generated photoelectrons.
        
        undrift(bool=True):        Apply RCC drift correction (see picasso.postprocess)
        segments(int=1000):     Segment length (frames) for undrifting by RCC (see picasso.render)
    
    Returns: 
        list:
        - [0][0](dict):     kwargs passed to localize_movie()
        - [0][1](dict):     kwargs passed to undriftrcc_locs()
        - [1](numpy.array): Undrifted localization list saved as _render.hdf5 (if ``undrift`` was ``True``) otherwise localization list (no undrifting) saved as _locs.hdf5
        - [2](str):         Last full file saving path for input to further modules
    '''
    ############################# Set standard paramters if not given with function call
    standard_params={'calibrate_movie': True,
                                    'localize':True,
                                    'box':9,
                                    'mng':'auto',
                                    'undrift':True,
                                    'segments':1000,
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
        
        ### Calibrate movie (CMOS)
        if params['calibrate_movie']:
            offset = picasso_addon.load_offset_map() # Load offset map
            gain = picasso_addon.load_gain_map()     # Load gain map
            print('Calibrating movie ...')
            file = (file - offset)/gain    # Calibrate movie
            params['baseline'] = 0     # Change camera info
            params['sensitivity'] = 1
            
        ### Autodetect mng
        if params['mng'] == 'auto':
            params['mng'] = autodetect_mng(file,info,params['box'])
            params['auto_mng'] = True
            print('Minimum net-gradient set to %i'%(params['mng']))
           
        ### Localize movie
        params_localize, locs = localize_movie(file,**params)
        try: params_localize['auto_mng'] = params['auto_mng'] # Add auto_mng entry to localize params to indicate it was active
        except: pass
    
        ### Save _locs and yaml
        print('Saving _locs ...')
        info_localize=info.copy()+[params_localize] # Update info
        next_path=path+'_locs.hdf5' # Update path
        io.save_locs(next_path,
                            locs,
                            info_localize,
                            )
        
        ### Update path and info 
        next_path = os.path.splitext(next_path)[0] # Remove file extension
        info = info + [params_localize] # Update info
        
    else:
        locs = file
        next_path = path 
    
    ############################## Undrift
    if params['undrift']==True:
        try:
            drift, locs_undrift = postprocess.undrift(locs,
                                                                           info,
                                                                           params['segments'],
                                                                           display=False)
            ### Save _locs_render and yaml
            print('Saving _render ...')
            info_undrift = info.copy() + [ {'generatedby':'picasso.postprocess.undrift', 'segments':params['segments']} ] # Update info
            next_path = next_path + '_render.hdf5' # Update path
            io.save_locs(next_path,
                                locs_undrift,
                                info_undrift,
                                )
        except:
            print('Undrifting by RCC was not possible')
            locs_undrift = locs
            info_undrift = info_localize
            
    else:
          print('No undrifting')
          locs_undrift = locs
          info_undrift = info_localize
          
    return [info_undrift,locs_undrift]
