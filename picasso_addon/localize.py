import os

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
def localize_movie(movie,**params):
    '''
    Localize movie by least squares fit.
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
    ### Procsessing marks: extension&generatedby
    params['extension']='_locs'
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
    Undrift localization by rcc.
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
    ### Procsessing marks: extension&generatedby
    params['extension']='_locs_render'
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
def main(movie,info,**params):
    '''
    Localize movie (least squares, GPU fitting if available) and 
    undrift resulting localizations using rcc.
    
    
    args:
        movie(picasso.io):         Raw movie loaded with picasso.io.load_movie()
        info(picasso.io):          Info to raw movie loaded with picasso.io.load_movie()
    
    **kwargs: If not explicitly specified set to default, also when specified as None
        baseline(int=70):          Camera spec. baseline (see picasso.localize).
        gain(float=1):             Camera spec. EM gain (see picasso.localize)
        sensitivity(float=0.56):   Camera spec. sensitivity (see picasso.localize)
        qe(float=0.82):            Camera spec. sensitivity (see picasso.localize)
        box(int=5):                Box length (uneven!) of fitted spots (see picasso.localize)
        mng(in=400):               Minimal net-gradient spot detection threshold(see picasso.localize)
        segments(int=1000):        Segment length (frames) for undrifting by RCC (see picasso.render)
    
    '''
    ### Set standard paramters if not given with function call
    standard_params={'undrift':True}
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    ### Get path of raw data
    path=info[0]['File']
    path=os.path.splitext(path)[0]
    ### Localize movie
    out_localize=localize_movie(movie,**params)
    
    ### Save _locs and yaml
    print('Saving _locs ...')
    info_locs=info.copy()+[out_localize[0]]
    io.save_locs(path+out_localize[0]['extension']+'.hdf5',
                 out_localize[1],
                 info_locs,
                 )
    ### Undrift
    if params['undrift']==True:
        try:
            out_undrift=undriftrcc_locs(out_localize[1],info,**params)
            
            ### Save _locs_render and yaml
            print('Saving _render ...')
            info_render=info_locs.copy()+[out_undrift[0]]
            io.save_locs(path+out_undrift[0]['extension']+'.hdf5',
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
    
    return [out_localize, out_undrift]