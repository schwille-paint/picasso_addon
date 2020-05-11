import os
import numba
import numpy as np

import picasso.io as io
import picasso.localize as localize
import picasso.gausslq as gausslq
import picasso.postprocess as postprocess

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
def localize_movie(movie,**params):
    '''
    Localize raw movie using picasso.gausslq
    
    Args:
        movie (io.TiffMultiMap): Movie object as created by picasso.io.load_movie().
    
    Keyword Args:
        localize (bool=True): Localize raw movie (see picasso.localize)
        baseline (int=70): Camera spec. baseline (see picasso.localize).
    Returns:
        list: 
		- [0] Dictionary of kwargs passed to function
        - [1] Localizations (numpy.array) according to picasso.gausslq
    '''
    ### Set standard paramters if not given with function call
    standard_params={'baseline':70,
                     'gain':1,
                     'sensitivity':0.56,
                     'qe':0.82,
                     'box':5,
                     'mng':400,
                     }
    ### Check if key in params, if not set to standard_params
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    ### Remove keys that are not needed
    delete_key=[]
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
        
    ### Procsessing marks:
    params['generatedby']='picasso_addon.localize.localize_movie()'
    
    ### Set camera specs for photon conversion
    camera_info={}
    camera_info['baseline']=params['baseline']
    camera_info['gain']=params['gain']
    camera_info['sensitivity']=params['sensitivity']
    camera_info['qe']=params['qe']
    
    ### Spot detection
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

    #### Gauss-Fitting
    em = camera_info['gain'] > 1
    
    if gpufit_available:
        print('GPU fitting ...')
        theta = gausslq.fit_spots_gpufit(spots)
        locs=gausslq.locs_from_fits_gpufit(identifications,
                                           theta,
                                           params['box'],
                                           em)
    else:
        print('Fitting ...')
        fs=gausslq.fit_spots_parallel(spots,asynch=True)
        theta=gausslq.fits_from_futures(fs)   
        locs=gausslq.locs_from_fits(identifications,
                                    theta,
                                    params['box'],
                                    em)
        
    assert len(spots) == len(locs)
    
    return [params,locs]

#%%
def undriftrcc_locs(locs,info,**params):
    '''
    Undrift localization by rcc (see picasso.render).
    
    Args:
        locs (numpy.array): Localizations as obtained by picasso.localize()
        info (dict): Meta-data as generated by picasso.localize (.yaml files)
    Keyword Arguments:
        segments (int=1000): Number of frames per segment used for correlation.
    Returns:
        list: 
		- [0] Dictionary of kwargs passed to function
        - [1] Undrifted localizations (numpy.array) using picasso.postprocess.undrift().
        
    '''
    ### Set standard paramters if not given with function call
    standard_params={'segments':1000,
                     }
    ### Check if key in params, if not set to standard_params, also when None was passed as value
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
            
    ### Remove keys that are not needed
    delete_key=[]
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
        
    ### Procsessing marks: generatedby
    params['generatedby']='picasso_addon.localize.undriftrcc_locs()'
    
    drift,locs_render=postprocess.undrift(locs,
                                          info,
                                          params['segments'],
                                          display=False,
                                          segmentation_callback=None,
                                          rcc_callback=None,
                                          )    
    return [params,locs_render]

#%%
def main(file,info,path,**params):
    '''
    Localize movie (least squares, GPU fitting if available) and undrift resulting localizations using rcc.
     
    Args:
        file(picasso.io):          Either raw movie loaded with picasso.io.load_movie() or_locs.hdf5 loaded with picasso.io.load_locs()
        info(list(dicts)):         Info to raw movie/_locs.hdf5 loaded with picasso.io.load_movie() or picasso.io.load_locs()
    
    Keyword Arguments:
        localize(bool=True)        Localize raw movie (see picasso.localize)
        baseline(int=70):          Camera spec. baseline (see picasso.localize).
        gain(float=1):             Camera spec. EM gain (see picasso.localize)
        sensitivity(float=0.56):   Camera spec. sensitivity (see picasso.localize)
        qe(float=0.82):            Camera spec. sensitivity (see picasso.localize)
        box(int=5):                Box length (uneven!) of fitted spots (see picasso.localize)
        mng(int or str='auto'):    Minimal net-gradient spot detection threshold(see picasso.localize. If set to 'auto' minimal net_gradient is determined by autodetect_mng().
        undrift(bool=True)         Apply RCC drift correction (see picasso.render)
        segments(int=1000):        Segment length (frames) for undrifting by RCC (see picasso.render)
    
    Returns: 
        list:
        - [0][0] Dict of **kwargs passed to localize
        - | [0][1] Localizations (numpy.array) as created by picasso.localize
          | Will be saved with extension '_locs.hdf5' for usage in picasso.render
        - [1][0] Dict of **kwargs passed to undriftrcc_locs        
        - | [1][1] Undrifted(RCC) localizations as created by picasso.render. 
          | If undrifing was not succesfull just corresponds to original loclizations.
          | Will be saved with extension '_locs_render.hdf5' for usage in picasso.render                            
          | If undrifting was not succesfull only _locs will be saved.
    '''
    ### Path of file that is processed
    path=os.path.splitext(path)[0]
    
    ### Set standard paramters if not given with function call
    standard_params={'localize':True,
                     'undrift':True,
                     'mng':'auto',
                     'box':5,
                     }
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    
    ############################# Localize
    if params['localize']==True:
        
        ### Autodetect optimum minimal net-gradient
        if params['mng']=='auto':
            params['mng']=autodetect_mng(file,info,params['box'])
            params['auto_mng']=True
            print()
            print('Minimum net-gradient set to %i'%(params['mng']))
           
        ### Localize movie
        out_localize=localize_movie(file,**params)
        try: out_localize[0]['auto_mng']=True # Add auto_mng entry to localize yaml if active
        except: pass
    
        ### Save _locs and yaml
        print('Saving _locs ...')
        info_locs=info.copy()+[out_localize[0]] # Update inof
        next_path=path+'_locs.hdf5' # Update path
        io.save_locs(next_path,
                     out_localize[1],
                     info_locs,
                     )
        
    else: 
        out_localize=[info,file]
        next_path=path
    
    info=out_localize[0] # Update info
    
    ############################## Undrift
    if params['undrift']==True:
        try:
            out_undrift=undriftrcc_locs(out_localize[1],info,**params)
            
            ### Save _locs_render and yaml
            print('Saving _render ...')
            info_render=info.copy()+[out_undrift[0]] # Update info
            next_path=path+'_render.hdf5' # Update path
            io.save_locs(next_path,
                         out_undrift[1],
                         info_render,
                         )
        except:
            print('Undrifting by RCC was not possible')
            out_undrift=out_localize.copy()
            out_undrift[0]={'undrift':'Error','extension':'_locs'}
            
    else:
         print('No undrifting')
         out_undrift=out_localize.copy()
         out_undrift[0]={'undrift':False,'extension':'_locs'}
         
    
    return [out_undrift[0],out_undrift[1],next_path]