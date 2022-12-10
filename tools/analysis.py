#!/usr/bin/env python

import numpy as np
import reader
import xarray as xr
xr.set_options(keep_attrs=True)

######################################################################
######################################################################

def calc_stream_function( dset ):

    
    # get height increments
    full_height, half_height = reader.get_mean_geoptential_height( averaging_dims = ('time', 'lon') )
    
    dz = np.abs( half_height.diff('lev') ) * 1e3   # units: m
    dz = dz.assign_coords({'lev': full_height})

    rhov = uv['rhov'].mean(['time',])
    
    # integration
    kernel = rhov * dz
    kernel = kernel.reindex( lev = np.sort(dz.lev) )
    integral = kernel.cumsum('lev')

    phi = np.deg2rad( kernel.lat )
    cosphi = np.cos( phi )
    a = 6372e3    # units: m

    Psi = 2*np.pi*a*cosphi * integral / 1e10   # units 1e10 kg s-1

    return Psi

######################################################################
######################################################################

def ens_stat( vin ):

    v = vin.drop_sel({'ensemble':'nudged'})

    ensmean = v.mean('ensemble')
    ensvar  = v.var('ensemble')

    fire_perturb_list = ['fire1.0', 'fire2.0', 'fire3.0', 'fire5.0']

    mean_reference =  ensmean.sel( mode = 'fire0.0' )
    mean_difference = ( ensmean.sel(mode = fire_perturb_list) - mean_reference )  #/ fire_scaling


    enslist = v.ensemble.data
    nens = len( enslist )

    var_difference = ( ensvar.sel(mode = fire_perturb_list) + ensvar.sel( mode = 'fire0.0' ) ) # / fire_scaling
    conf_difference = 2 * np.sqrt( var_difference / nens )

    upper = mean_difference + conf_difference
    lower = mean_difference - conf_difference


    mean = mean_difference.expand_dims('stats')
    mean['stats'] = ['mean']

    conf = conf_difference.expand_dims('stats')
    conf['stats'] = ['confidence']

    upper = upper.expand_dims('stats')
    upper['stats'] = ['upper']

    lower = lower.expand_dims('stats')
    lower['stats'] = ['lower']

    ref = mean_reference.expand_dims('stats') * xr.ones_like( mean )    # expand ref to each fire mode
    ref['stats'] = ['reference']

    vstat = xr.concat( [mean, conf, upper, lower, ref], dim = 'stats')
    
    vstat.attrs = v.attrs

    return vstat


######################################################################
######################################################################

def stats_and_nudged( v ):
    
    v_stats = ens_stat( v )
    
    v_nudged = v.sel(ensemble = 'nudged')
    v_nudged = v_nudged.expand_dims('stats')
    v_nudged['stats'] = ['nudged',]
    dv_nudged = v_nudged.sel( mode = v_stats.mode ) - v_nudged.sel( mode = 'fire0.0' )
    v_stats = xr.concat( [v_stats, dv_nudged], dim = 'stats')                            
        
    
    return v_stats.squeeze()

######################################################################
######################################################################

def weighted_mean( v ):
    
    lat = v.lat
    phi = np.deg2rad( lat )
    
    cosphi = np.cos(phi)
    w = cosphi
    
    v_mean = ( v * w ).mean('lat') / w.mean('lat')
    
    return v_mean 

######################################################################
######################################################################

def glob_mean( v ):
    return weighted_mean( v ) 

######################################################################
######################################################################


def sh_mean( v ):
    vs = v.sel( lat = slice(0, -90) )
    return weighted_mean( vs )

######################################################################
######################################################################

def nh_mean( v ):
    vs = v.sel( lat = slice(90, 0 ))
    return weighted_mean( vs )

######################################################################
######################################################################


######################################################################
######################################################################

######################################################################
######################################################################


######################################################################
######################################################################

######################################################################
######################################################################


######################################################################
######################################################################

######################################################################
######################################################################



