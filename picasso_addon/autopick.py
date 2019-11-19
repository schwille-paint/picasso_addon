import os
import numpy as np
import numba as numba
import pandas as pd
from scipy.spatial import cKDTree

import picasso.localize as localize
import picasso.gausslq as gausslq
import picasso.render as render
import picasso.io as io

import picasso_addon.io as addon_io

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def check_spots(frame, y, x, box):
    '''
    Check all boxes around x,y for number of localizations, active pixels and center of mass position
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
    Identify picks in rendered image according to number of localizations.

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
    Convert spot centers as detected in rendered localizations by given viewport minima and oversampling back to original localization coordinates.
    
    Parameters
    ---------
    spot_centers : pandas.DataFrame
        Output spotcenters_in_image function.
    viewport_min: tuple of len=2
        First tuple of viewport as in picasso.render: viewport=[(min_y,min_x),(max_y,max_x)]
    oversampling: int
        Pixel oversampling as in picasso.render
    
    Returns
    -------
    spot_centers_convert: pandas.DataFrame
        Spot centers converted back to orignal localization coordinates.
    
    '''
    spot_centers_convert=spot_centers.copy()
    spot_centers_convert.x=spot_centers.x/oversampling+viewport_min[1]
    spot_centers_convert.y=spot_centers.y/oversampling+viewport_min[0]
    
    return spot_centers_convert
#%%
def query_locs_for_centers(locs,centers,pick_radius=1):
    '''
    Builds up KDtree for locs, queries the tree for localizations within pick_radius (norm of p=2) aorund centers (xy coordinates).
    Output will be list of indices as result of query.
    Parameters
    ---------
    locs : numpy.recarray or pandas.DataFrame
        locs as created by picasso.localize with fields 'x' and 'y'
    centers: np.array of shape (m,2)
        x and y pick center positions
    pick_diameter: float
        Pick diameter in px
    Returns
    -------
    picks_idx: list
        List of len(centers). Single list entries correpond to indices in locs within pick_radius around centers.
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
    Assign group ID to locs and sort
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
def main(locs,info,**params):
    '''
    Cluster detection (pick) in localization list by thresholding in number of localizations per cluster.
    Cluster centers are determined by creating images of localization list with set oversampling.
    
    
    args:
        locs(numpy.recarray):      (Undrifted) localization list as created by picasso.localize
        info(list(dict)):          Info to localization list when loaded with picasso.io.load_locs()
    
    **kwargs: If not explicitly specified set to default, also when specified as None
        oversampling(int=5):            Oversampling for rendering of loclization list, i.e. sub-pixels per pixel of original image
        pick_box(int=2*oversampling+1): Box length for spot detection in rendered image similar to picasso.localize 
        min_n_locs(float=0.2*NoFrames): Detection threshold for number of localizations in cluster
        fit_center(bool=False):         If set to False center corresponds to center of mass of of loclizations per sub-pixel 
                                        If set to True center corresponds to gaussian fit of pick_box area
        pick_diameter(float=2):         Pick diameter in original pixels, i.e. cluster = localizations within 
                                        circle around center of diameter = pick_diameter (see picasso.render)
        lbfcs(bool=False):              If set to True will overrun min_n_locs and sets it to 0.01*NoFrames
    
    '''
    ### Set standard conditions if not set as input
    oversampling=5
    NoFrames=info[0]['Frames']
    
    standard_params={'oversampling':oversampling,
                     'pick_box':2*oversampling+1,
                     'min_n_locs':0.2*NoFrames,
                     'fit_center':False,
                     'pick_diameter':2,
                     }
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
    ### Add some cases to cover both lbfcs and spt usage regarding min_value
    try: 
        if params['lbfcs']==True: params['min_n_locs']=0.01*NoFrames
    except KeyError:
        params['lbfcs']=False
        
    ### Procsessing marks: extension&generatedby
    try: extension=info[-1]['extension']+'_picked'
    except: extension='_locs_xxx_picked'
    params['extension']=extension
    params['generatedby']='picasso.autopick.main()'
    
    ### Get path of raw data
    path=info[0]['File']
    path=os.path.splitext(path)[0]
    
    ### Check if locs is given as numpy.recarray as created by piocasso.localize
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
    addon_io._save_picks(centers,
                         params['pick_diameter'],
                         path+extension.replace('_picked','_autopick.yaml'),
                         )
    
    ### Query locs for centers
    print('Build up and query KDtree ...')
    picks_idx=query_locs_for_centers(locs,
                                     centers[['x','y']].values,
                                     pick_radius=params['pick_diameter']/2,
                                     )
    ## Assign group ID to locs
    print('Assigning group ID ...')
    locs_picked=get_picked(pd.DataFrame(locs),
                           picks_idx,
                           field='group',
                           )
    ### Save locs_picked as .hdf5 and info+params as .yaml
    print('Saving _picked ...')
    info_picked=info.copy()+[params]
    io.save_locs(path+extension+'.hdf5',
                 locs_picked.to_records(index=False),
                 info_picked,
                 )
    
    return [params,centers,locs_picked]