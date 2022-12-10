#!/usr/bin/env python

import numpy as np
import pylab as plt
import seaborn as sns
import matplotlib.colors as mcol

from analysis import glob_mean, sh_mean, nh_mean, ens_stat

######################################################################
######################################################################

def colored_contours( dth, 
                            it_start = 120, it_end = 240, it_step = 12,
                            vname = 'tpot', value_thresh = 4., title = None, cmap = plt.cm.inferno):
    

    for it in range(it_start, it_end, it_step):
        irel = (1.*it - it_start) / (it_end - it_start)
        crgb = cmap(irel)
        chex = mcol.to_hex( crgb )
        d = dth[vname].isel(time = it)
        d.plot.contour( levels = [value_thresh,], colors = [chex,])

        date = str( d.time.data.astype('datetime64[D]' ) )
        plt.figtext( 0.9, 0.82 - 0.75*irel, date, c = chex, fontsize = 'small')

    plt.subplots_adjust(right = 0.8)
    
    if title is not None:
        plt.title( title, pad = 20 )
    sns.despine()
    
    return 

######################################################################
######################################################################

def plot_diff(vin, method = 'timemean', ylabel = None, title = None):
    
    v = vin
    lat = v.lat
    
    if method == 'timemean':
        v = v.mean('time')
        
    v.load()
    stats = ens_stat( v ) 

    modelist = stats.mode

    for mode in modelist:
        
        vm = stats.sel( mode = mode, stats = 'mean' ).squeeze()
       
        # upper & lower bounds
        v1 = stats.sel( mode = mode, stats = 'lower' ).squeeze()
        v2 = stats.sel( mode = mode, stats = 'upper' ).squeeze()
        
        line, = plt.plot(lat, vm, lw = 5)
        plt.fill_between( lat, v1, v2, alpha = 0.2, color = line.get_color() )
    
    plt.xlabel( 'latitude / (deg N)')
    try:
        if ylabel is None:
            plt.ylabel( '%s \n (%s)' % (v.name, v.units) )
        else:
            plt.ylabel( ylabel)
        
        if title is None:
            plt.title(v.long_name, fontweight = 'bold')
        else:
            plt.title(title, fontweight = 'bold')
    except:
        pass
 

    return
    

######################################################################
######################################################################

def plot_diff_ts(vin, method = 'globalmean', style = 'errorbar', add_nudged = False):

    time = np.array( [1,2,3] )
    v = vin.resample({'time':'1M'}).mean() 

    if method == 'globalmean':
        v = glob_mean( v )
    elif method == 'shmean':
        v = sh_mean( v )
    elif method == 'nhmean':
        v = nh_mean( v )
    elif method == 'keep':
        v = v
        
        
    v.load()
    stats = ens_stat( v ) 

    modelist = stats.mode
    i = 0

    for mode in modelist:
        
        vm = stats.sel( mode = mode, stats = 'mean' ).squeeze()
       
        # upper & lower bounds
        dv = stats.sel( mode = mode, stats = 'confidence' ).squeeze()

    
        #line, = 
        if style == 'errorbar':
            plt.errorbar(time + 0.1*(i-2), vm, yerr = dv, marker = 'o', lw = 1)
        elif style == 'bar':
            plt.bar(time + 0.15*(i-1.5), vm, width = 0.08, yerr = dv, lw = 1, error_kw = {'alpha': 0.3, 'color':'lightgray', 'lw': 1} )
        
        i += 1
        
    plt.xticks( time, ['Jan', 'Feb', 'Mar'])
    plt.xlabel( '2020' )
    try:
        plt.ylabel( '%s \n (%s)' % (v.name, v.units) )
        plt.title(v.long_name, fontweight = 'bold')
    except:
        pass
    
    if add_nudged:
        plt.gca().set_prop_cycle(None)
        i = 0
        for mode in modelist:
            diff = (v.sel(ensemble='nudged', mode = mode) - v.sel(ensemble='nudged', mode = 'fire0.0')).squeeze()
#            plt.plot(time + 0.15*(i-2.1), diff, marker = '*', lw = 0, ms = 20, mew = 2, mfc = 'w')
            plt.bar(time + 0.15*(i-2.), diff, width = 0.04, alpha = 0.3)
            i += 1
                    
    
    return

######################################################################
######################################################################

