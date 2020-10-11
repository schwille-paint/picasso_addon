'''
.. _picasso.localize:
    https://picassosr.readthedocs.io/en/latest/localize.html
.. _picasso.render:
    https://picassosr.readthedocs.io/en/latest/render.html
.. _spt:
    https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1
'''

import os
import numpy as np
import numba as numba
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

import picasso.localize as localize
import picasso.gausslq as gausslq
import picasso.render as render
import picasso.io as io

import picasso_addon.io as addon_io

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def check_spots(frame,y,x,box):
    '''
    Check all boxes around x,y for number of localizations, active pixels and center of mass position.
    
    Args:
        frame (np.array): Subpixel rendered localization image.
        y (int):          Box center y-coordinate (row)
        x (int):          Box center x-coordinate (column)
        box (int):        Box size in subpixels. Must be uneven!
        
    Returns:
        tuple: 
            - [0] (float): Center of mass y-position as measured from y
            - [1] (float): Center of mass x-position as measured from x
            - [2] (int):   Number of localizations within box
            - [3] (int):   Number of subpixels within box with at least one localization
    '''
    box_half = int(box / 2)
    dx = np.zeros(len(x), dtype=np.float32)
    dy = np.zeros(len(x), dtype=np.float32)
    n_locs = np.zeros(len(x), dtype=np.float32)
    n_activepx = np.zeros(len(x), dtype=np.int32)
    
    for i in range(0,len(x)):
        ### Get spot
        spot=frame[y[i]-box_half:y[i]+box_half+1,
                   x[i]-box_half:x[i]+box_half+1,
                   ]
        ### Get center of mass position, sum and number of active pixels in spot 
        n_locs[i],dx[i],dy[i]=gausslq._sum_and_center_of_mass(spot,box)
        dx[i]=dx[i]-box_half # dx is now measured from center position x
        dy[i]=dy[i]-box_half # dy is now measured from center position y
        n_activepx[i]=np.sum(spot>0)
    
    return dy,dx,n_locs,n_activepx

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def identifyspots_in_image(image, minimum_n_locs, box):
    '''
    Identify boxes above localization threshold in rendered localizations image by using check_spots().
	
	Args:
		image (np.array): Subpixel rendered localization image.
		minimum_n_locs (int): Number of localization threshold for valid boxes
		box (int): Box size in subpixels. Must be uneven!
	
	Returns:
		tuple:
    		- [0] (int):   Local box maximum y (i.e. box y center coordinate)
    		- [1] (int):   Local box maximum x (i.e. box x center coordinate)
    		- [2] (float): Center of mass y-position as measured from y
    		- [3] (float): Center of mass x-position as measured from x
    		- [4] (int):   Number of localizations within box	
    '''
    ### Get local maxima in each box by covering complete image
    y,x=localize.local_maxima(image, box)
    ### Check spots for number of loclizations and number active pixels
    dy,dx,n_locs, n_activepx = check_spots(image,y,x,box)
    ### Apply filter and return positive spots
    positives = (n_locs >= minimum_n_locs)&(n_activepx>1)
    y = y[positives]
    x = x[positives]
    dy = dy[positives]
    dx = dx[positives]
    n_locs = n_locs[positives]
    n_activepx = n_activepx[positives]
    return y, x, dy, dx, n_locs

#%%
def spotcenters_in_image(image,box,min_nlocs,fit=False):
    '''
    Get pick center coordinates in subpixel rendered localization image by either 
    calculating center of mass or fitting 2D Gaussian.
    
    Args:
  		image (np.array): Subpixel rendered localization image.
  		box (int):        Box size in subpixels. Must be uneven!
  		min_nlocs (int):  Number of localization threshold for valid boxes
  		fit (bool=False): Employ 2D gaussian fitting? Normally not better performing. If False only center of mass position.
    Returns:
        tuple:
            - [0] (pandas.DataFrame): Spot center coordinates (image units) and number of localizations (x,y,n_locs)
            - [1] (bool): True if fit was performed
    '''
    image=image.astype(float)
    ### 1. Cover complete image with boxes and identify local maxima
    ### 2. Reposition boxes around local maxima and check for:
    ###         - Number of localizations (n_locs)
    ###         - Center of mass positions (dx,dy) in spot coordinates with x,y at center position (0,0)
    y,x,dy,dx,n_locs=identifyspots_in_image(image,min_nlocs,box)
       
    ### Remove spots closer than box_half to image boarder
    box_half=int(box/2)
    im_size_x=np.shape(image)[0]
    im_size_y=np.shape(image)[1]
    
    positives=(y>box_half)&((y+box_half)<im_size_y)&(x>box_half)&((x+box_half)<im_size_x)
    y=y[positives]
    x=x[positives]
    dy=dy[positives]
    dx=dx[positives]
    n_locs=n_locs[positives]
    
    ### Try gaussian fitting of boxes to update center position, return center of mass position in case of failure
    if fit==True:
        print('Fitting spots ...')
        ### Prepare spots
        spots = np.zeros((len(x), box, box), dtype=image.dtype) # Initialize spots
        localize._cut_spots_frame(image,0,np.zeros(len(x)),x,y,int(box/2),0,len(x),spots) # Fill spots
        spots=spots.astype(float) # Convert to float for fitting
        try:
            ### Fitting
            theta=gausslq.fit_spots_parallel(spots,asynch=False)
            ### Add box offset
            x_new=theta[:, 0]+x
            y_new=theta[:, 1]+y
            
            ### Check again for boundaries after fitting
            positives=(y_new>box_half)&((y_new+box_half)<im_size_y)&(x_new>box_half)&((x_new+box_half)<im_size_x)
            y_new=y_new[positives]
            x_new=x_new[positives]
            n_locs=n_locs[positives]
        except:
            print('Fitting failed switching to center of mass position!')
            x_new=x+dx
            y_new=y+dy
            fit=False
    else: # Also return center of mass position
        x_new=x+dx
        y_new=y+dy
    
    ### Assign
    spot_centers=pd.DataFrame(columns=['x','y','n_locs'])
    spot_centers.x=x_new+0.5 # We have to add 0.5 to go from a pixelated presentation to continuous
    spot_centers.y=y_new+0.5 # We have to add 0.5 to go from a pixelated presentation to continuous
    spot_centers.n_locs=n_locs
    
    return spot_centers,fit

#%%
def coordinate_convert(spot_centers,viewport_min,oversampling):
    '''
    Convert spot centers as detected in rendered localizations by given viewport minima and oversampling 
    back to original localization coordinates.
    
    Args:
        spot_centers (pandas.DataFrame): Output of function ``spotcenters_in_image()[0]``, i.e. spot center coordinates.
        viewport_min (tuple): ``(min_x,min_y)`` of viewport as in picasso.render.
        oversampling (int): Pixel oversampling as in picasso.render
        
    Returns:
        pandas.DataFrame: spot_centers converted back to orignal localization coordinates.
    '''
    spot_centers_convert=spot_centers.copy()
    spot_centers_convert.x=spot_centers.x/oversampling+viewport_min[1]
    spot_centers_convert.y=spot_centers.y/oversampling+viewport_min[0]
    
    return spot_centers_convert
#%%
def query_locs_for_centers(locs,centers,pick_radius=1):
    '''
    Builds up KDtree for locs, queries the tree for localizations within pick_radius (norm of p=2) around centers (xy coordinates).
    Output will be list of indices as result of query.
    
    Args:
        locs (numpy.recarray or pandas.DataFrame): Localizations as created by picasso.localize with fields ``x`` and ``y``.
        centers (np.array):                        Pick center coordinates (x,y)
        pick_radius (float=1):                     Pick diameter in px.
    
    Return:
        list: 
            List of ``len(centers)``. Single list entries correpond to indices in ``locs`` of localizations 
            within ``pick_radius`` around ``centers``. Hence every list indicates one pick (group).
    '''
    
    #### Prepare data for KDtree
    data=np.vstack([locs.x,locs.y]).transpose()
    #### Build up KDtree
    tree=cKDTree(data,compact_nodes=False,balanced_tree=False)
    #### Query KDtree for indices belonging to picks
    picks_idx=tree.query_ball_point(centers,r=pick_radius,p=2)
    
    return picks_idx
#%%
def get_picked(locs,picks_idx,field='group'):
    '''
    Assign group ID to locs according to picks_idx as obtained by ``query_locs_for_centers()`` and sort.
    
    Args:
        locs (pandas.DataFrame): Localizations (see picasso.localize) converted to DataFrame for faster sorting.
        picks_idx (list): 		 Single list entries correpond to indices in locs within pick_radius around centers. 
                                 See ``query_locs_for_centers()``.
        field (str='group'):     Name of column for group ID in output DataFrame.
    
    Returns:
        pandas.DataFrame: Picked as in `picasso.render`_, i.e. original locs with group assigned according to picks defined by picks_idx.
    '''
    locs_picked=locs.copy()

    #### Assign group index
    groups=np.full(len(locs),-1)
    for g,idx in enumerate(picks_idx): groups[idx]=g   
    locs_picked[field]=groups
    
    #### Dropping, type conversion and sorting
    locs_picked=locs_picked.loc[locs_picked.group>=0,:] # Drop all unassigned localizations i.e. group==-1
    locs_picked=locs_picked.astype({'group':np.uint32}) # Convert to right dtype
    locs_picked.sort_values(['group','frame'],inplace=True) # Sort
    
    return locs_picked

#%%
def ck_nlf(trace, M = 10, N = [2, 4, 8, 16], p = 30):
    '''
    Chung-Kennedy nonlinear filter

    Nonlinear local low-pass filter optimized for edge preservation. Used
    quite frequently in step detection in noisy traces.
    For details, see:
    Chung SH & Kennedy RA, J Neurosci Meth 1991,
    DOI: 10.1016/0165-0270(91)90118-J

    Note that the algorithm frequently raises warnings saying "invalid value
    encountered in true_divide". There is no need to worry about that:
    The filter, when used with high p, tends to create extreme numbers in
    calculation intermediates, which may even end up creating zeros. This only
    affects weighting functions of a handful of data points, though, and has
    negligible impact on the overall filter result. It will NOT create any
    extreme values in the returned filtered trace, at least I have never
    encountered such results.

    Args:
        trace (numpyp.array): Time trace to be filtered. Signal only, no time axis required.
        M (int):              Field width for predictor weight determination. Typical values: 5-20.
        N (list):             Field width set for predictor calculation. Typical values: [4, 8, 16, 32] or subsets thereof. Minumum ``len(N) > 1``
        p (int):              Positive single number (usually int, but not strictly required). Nonlinear order of predictor weight calculation. 
                              Typical values: 5-100, strongly application-dependent. Higher values
                              mean sharper edge preservation.

    Returns:
        trace_f (numpy.array): 1D filtered time trace of same structure as input.

    '''

    # Initializations
    K = len(N)
    l_trace = len(trace)
    pred_f = np.zeros((l_trace, K))
    pred_b = np.zeros((l_trace, K))
    weight_f = np.zeros((l_trace, K))
    weight_b = np.zeros((l_trace, K))
    weight_f_norm = np.zeros((l_trace, K))
    weight_b_norm = np.zeros((l_trace, K))

    # Raw predictors.
    for k in range(K):
        trace_shift = np.empty((l_trace + N[k], N[k]))
        trace_shift[:, :] = np.nan
        for l in range(N[k]):
            trace_shift[l:l_trace + l, l] = trace
        # Forward predictor
        pred_f[0] = trace[0]
        pred_f[1:, k] = np.nanmean(trace_shift[:l_trace - 1], axis = 1)
        # Backward predictor
        pred_b[0:-1, k] = np.nanmean(trace_shift[N[k]:-1], axis = 1)
        pred_b[-1, k] = trace[-1]

    # Non-normalized weights.
    # First value of forward and last value of backward would lead to div by 0.
    # They are skipped and their weights set to (rather left at) 0.
        # Forward predictor weights
        trace_shift = np.empty((l_trace + M - 1, M))
        trace_shift.fill(np.nan)
        MSE_f = (trace - pred_f[:,k]) ** 2
        for m in range(M):
            trace_shift[m:l_trace + m, m] = MSE_f
        weight_f[1:, k] = np.nansum(trace_shift[1:l_trace, :], axis = 1) ** (-p)
        # Backward predictor weights
        trace_shift = np.empty((l_trace + M - 1, M))
        trace_shift.fill(np.nan)
        MSE_b = (trace - pred_b[:, k]) ** 2
        for m in range(M):
            trace_shift[m:l_trace + m, m] = MSE_b
        weight_b[:-1, k] = np.nansum(trace_shift[-l_trace:-1, :], axis = 1) ** (-p)

    # Normalize weights.
    weightsum = np.nansum(weight_f, axis = 1) + np.nansum(weight_b, axis = 1)
    for k in range(K):
        weight_f_norm[:, k] = weight_f[:, k] / weightsum
        weight_b_norm[:, k] = weight_b[:, k] / weightsum

    # Get filtered trace.
    pred_f_w = pred_f * weight_f_norm
    pred_b_w = pred_b * weight_b_norm
    trace_f = np.nansum(pred_f_w, axis = 1) + np.nansum(pred_b_w, axis = 1)

    return trace_f

#%%
def main(locs,info,path,**params):
    '''
    Cluster detection (pick) in localizations by thresholding in number of localizations per cluster.
    Cluster centers are determined by creating images of localization list with set oversampling using picasso.render.
    
    Args:
        locs(numpy.recarray):           (Undrifted) localization list as created by `picasso.render`_.
        info(list(dict)):               Info to localization list when loaded with picasso.io.load_locs().
    
    Keyword Arguments:
        oversampling(int=5):            Oversampling for rendering of localization list, i.e. sub-pixels per pixel of original image.
        pick_box(int=2*oversampling+1): Box length for spot detection in rendered image similar to `picasso.localize`_.
        min_n_locs(float=0.2*NoFrames): Detection threshold for number of localizations in cluster.
                                        Standard value is set for `spt`_. 
                                        Set to lower value for usual DNA-PAINT signal (see ``lbfcs``).
        fit_center(bool=False):         False = Center of mass. True = 2D Gaussian fitting of center.
        pick_diameter(float=2):         Pick diameter in original pixels.
        lbfcs(bool=False):              If set to True will overrun min_n_locs and sets it to 0.05*NoFrames.
        
    Returns:
        list:
            - [0] (dict):             kwargs passed to function.
            - [1] (pandas.DataFrame): Picked localizations, saved with extension _picked.hdf5.
            - [2] (pandas.DataFrame): Center positions and number of localizations per pick, saved with extension _autopick.yaml.
            - [3] (str):              Full save path.

    '''
    ### Path of file that is processed
    path=os.path.splitext(path)[0]
    
    ### Set standard conditions if not set as input
    oversampling=5
    NoFrames=info[0]['Frames']
    
    standard_params={'oversampling':oversampling,
                     'pick_box':2*oversampling+1,
                     'min_n_locs':0.2*NoFrames,
                     'fit_center':False,
                     'pick_diameter':2,
                     }
    ### If lbFCS given overrun min_n_locs
    try: 
        if params['lbfcs']==True: 
            standard_params['min_n_locs']=0.05*NoFrames
            standard_params['lbfcs']=True
    except KeyError:
        pass
        
    ### Remove keys in params that are not needed
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    ### Remove keys in params that are not needed
    delete_key=[]
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
        
    ### Procsessing marks: generatedby
    params['generatedby']='picasso_addon.autopick.main()'
    
    
    print('Minimum number of localizations in pick set to %i'%(params['min_n_locs']))
    
    ### Check if locs is given as numpy.recarray as created by picasso.localize
    if type(locs) is not np.recarray:
        raise SystemExit('locs must be given as numpy.recarray')
  
    ### Render locs
    print('Rendering locs for pick detection ...')
    image=render.render(locs,
                        info,
                        oversampling=params['oversampling'],
                        )[1]  
    ### Get pick centers in image coordinates
    print('Identifiying valid picks ...')
    centers_image,fit=spotcenters_in_image(image,
                                           params['pick_box'],
                                           params['min_n_locs'],
                                           fit=params['fit_center'])
    params['fit_center']=fit # Update if it was not successful!
    
    ### Convert image coordinate centers to original values as in locs
    centers=coordinate_convert(centers_image,
                               (0,0),
                               params['oversampling'],
                               )
    ### Save converted centers as picks.yaml for later usage in picasso.render
    addon_io.save_picks(centers,
                        params['pick_diameter'],
                        path+'_autopick.yaml',
                        )
    
    ### Query locs for centers
    print('Build up and query KDtree ...')
    picks_idx=query_locs_for_centers(locs,
                                     centers[['x','y']].values,
                                     pick_radius=params['pick_diameter']/2,
                                     )
    ### Assign group ID to locs
    print('Assigning group ID ...')
    locs_picked=get_picked(pd.DataFrame(locs),
                           picks_idx,
                           field='group',
                           )
    
    ### Apply Chung-Kennedy local mean filter to photons of each group
    print('Applying Chung-Kennedy filter ...')
    tqdm.pandas()
    locs_picked = locs_picked.groupby('group').progress_apply(lambda df: df.assign( photons_ck = ck_nlf(df.photons.values).astype(np.float32) ) )
    
    ### Save locs_picked as .hdf5 and info+params as .yaml
    print('Saving _picked ...')
    info_picked=info.copy()+[params]      # Update info
    next_path=path+'_picked.hdf5'         # Update path
    io.save_locs(next_path,
                 locs_picked.to_records(index=False),
                 info_picked,
                 )
    
    return [params,locs_picked,centers,next_path]
