#!/usr/bin/env python

import numpy as np
import pylab as plt
import seaborn as sns
import matplotlib.colors as mcol
import xarray as xr

from analysis import glob_mean, sh_mean, nh_mean, ens_stat



######################################################################
######################################################################

def standardize_firenames( modes ):
    
    mlist = []
    for mode in modes:
        old_name = str( mode )
        new_name = old_name.upper()[:-2]
        
        mlist += [new_name, ]
        
    return  mlist

######################################################################
######################################################################


def set_levs(nmax, depth, largest = 5, sym = True, sign = True, zmax = 'None'):

    '''
    Makes a non-linear level set based on pre-defined numbers.
    
    Parameters
    ----------
    nmax : int
        exponent of maximum number
    depth : int
        number of iterations used down to scales of 10**(nmax - depth)
    largest : {5, 8}, optional
        set the largest number in the base array either to 5 or 8
    sym : {True, False}, optional
        switch if levels are symmetric around origin
    sign : {True, False}, optional
        switches sign if negative
    zmax : float, optional, default = 'none'
        limiter for levels, |levs| > zmax are not allowed
    
    Returns
    --------
    levs : np.array
        set of non-linear levels
    
    '''

    # nmax: 
    # depth: 

    # base
    if largest == 5:
        b = np.array([1,2,3,5], dtype = 'float')
    elif largest == 8:
        b = np.array([1,2,3,5,8], dtype = 'float')
    
    # exponents
    nmin = nmax - depth + 1
    ex = np.arange(nmin, nmax + 1, dtype = 'float')

    levs = np.array([])

    # calculation of levels ------------------------------------------
    # range below one
    for n in ex:
        levs = np.append(levs, b * 10**n)

    # ================================================================

    if not zmax == 'None':
       levs = levs[np.logical_and(levs>=-zmax,levs<=zmax)] 

    if sym:
        return np.append(-levs[::-1],np.append(0,levs))
    elif not sign:
        return np.append(-levs[::-1],0)
    else:
        return np.append(0,levs)
    


######################################################################
######################################################################

def forcing_overview_plot(forc_stats, stats_method = 'ensemble', alpha_method = 'confidence', only_plot_modemean = False,
                          add_nudged = False, ylim = 1, title = None, ylabel = None, ax = None, legend_labelsize = 16,
                          print_condition_label = True,
                          global_values = False,
                          varnames = ['IRF_sw', 'Adj_sw', 'ERF_sw', 'IRF_lw', 'Adj_lw', 'ERF_lw', 'IRF_net', 'Adj_net', 'ERF_net',],
                          colors = ['xkcd:sun yellow', 'gold', 'orange', 'tab:red', 'darkred', 'purple', 'yellowgreen', 'forestgreen', 'darkgreen']):

    
    if 'sky' in forc_stats.coords:
        nrows = 2
        skys = forc_stats.sky
    else:
        nrows = 1
        skys = [np.array(['none']),]
    
    if ax is None:
        fig, axs = plt.subplots( nrows = nrows, figsize = (20, 3 * nrows))
        if nrows == 1:
            axs = [axs,]
    else:
        axs = [ax,]
        
    plt.subplots_adjust( hspace = 0.6, top = 0.85, right = 0.8 )
    
    if title is None:
        title = stats_method
    
    x = 0
    dmode = 0.1
    dvar = dmode * 1
    dtime = dmode * 2
    dlatrange = dmode * 4
    
    
    scale = [1., 2., 3., 5.]
    scale_factors = xr.DataArray( data = np.array(scale), dims = 'mode', 
                                      coords = dict( mode = ('mode', forc_stats.mode.data) ) )
    months = ['Jan', 'Feb', 'Mar']
    
    for isky, sky in enumerate( skys ):
        plt.sca( axs[isky] )
        if nrows > 1:
            axs[isky].set_title( str( sky.values ), fontweight = 'bold')
            fstat_sky = forc_stats.sel( sky = sky )
        else:
            axs[isky].set_title( title, fontsize = 22, fontweight = 'bold', x = 0.46, ha = 'center' )
    
            fstat_sky = forc_stats
            
        x = 0
        latlabel_points = []


        
        for ilat, latrange in enumerate(forc_stats.latrange):
            for itime, time in enumerate(forc_stats.time):
                for ivar, vname in enumerate( varnames ):
                    fmodes = fstat_sky[vname].sel( time = time, latrange = latrange) / scale_factors
                    
                    frange = fmodes.sel(stats = 'mean').std('mode') #- fmodes.sel(stats = 'mean').min('mode')
                    fmodemean = fmodes.sel(stats = 'mean').mean('mode')
                    fnu_modemean = fmodes.sel(stats = 'nudged').mean('mode')
                    
                    if ivar == 5:
                        plt.text(x, -0.7* ylim, months[itime], ha = 'center', fontsize = 14)

                    for imode, mode in enumerate( forc_stats.mode):
                        f = fmodes.sel( mode = mode,)
                        fs = f # / scale[imode]
                        fnudged = fs.sel(stats = 'nudged')
                        
                        if stats_method == 'nudged':
                            fmean = fnudged
                            fmodemean = fnu_modemean
                            alpha = 1.
                        else:
                            fmean = fs.sel(stats = 'mean')
                            if alpha_method == 'confidence':
                                fconf = fs.sel(stats = 'confidence')
                                alpha_condition = np.abs(fmean/fconf) > 1

                            elif alpha_method == 'mode_range':
                                alpha_condition = np.abs(fmodemean/frange) > 1
                            
                            elif alpha_method == 'agree_with_nudged':
                                alpha_condition = fnu_modemean * fmodemean > 0 # else they have different sign 
                                
                            if alpha_condition:
                                alpha = 1.
                            else:
                                alpha = 0.25


                        if only_plot_modemean:
                            plt.bar(x, fmodemean, color = colors[ivar], width = dmode * 0.8, alpha = alpha)
                            if not add_nudged:
                                break
                        else:
                            plt.bar(x, fmean, color = colors[ivar], width = dmode * 0.8, alpha = alpha)
                            
                        if add_nudged:
                            if only_plot_modemean:
                                plt.plot(x, fnu_modemean, color = 'gray', marker = '_')
                                break
                            else:
                                plt.plot(x, fnudged, color = 'gray', marker = '_')

                        if not only_plot_modemean:
                            x += dmode
                        
                    x += dvar
                
                x += dtime
                

                
            latlabel_points += [x,]
            x += dlatrange

        latlabel_points = np.array( latlabel_points )
        latlabel_points -= latlabel_points[0]/2
        
        if not global_values:
            plt.xticks(latlabel_points, ['Southern Hemisphere', 'Tropics', 'Northern Hemisphere'], fontsize = 18) 
        plt.ylim(-ylim, ylim)
        if ylabel is None:
            plt.ylabel('Forcing \n (W m${}^{-2}$)' )
        else:
            plt.ylabel(ylabel)
        
        y = 0.7
        for ivar, vname in enumerate( varnames ):
            plt.figtext(0.82, y, vname, color = colors[ivar], fontsize = legend_labelsize)
            y-=0.06
            
            if np.mod(ivar + 1, 3) == 0:
                y -= 0.02
        
    if stats_method == 'ensemble' and print_condition_label:
        plt.figtext(0.8, 0.8, 'Condition: ' + alpha_method, fontsize = 'small')
        
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
        
        line, = plt.plot(lat, vm, lw = 5, label = str( mode.data ) )
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

def plot_diff_ts(vin, method = 'globalmean', style = 'errorbar', add_nudged = False, relative = False):

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

        if relative:
            vref = stats.sel( mode = mode, stats = 'reference').squeeze()
            
            vm = 100 * vm / vref
            dv = 100 * dv / vref
            
            
        #line, = 
        if style == 'errorbar':
            plt.errorbar(time + 0.1*(i-2), vm, yerr = dv, marker = 'o', lw = 1)
        elif style == 'bar':
            plt.bar(time + 0.15*(i-1.5), vm, width = 0.08, yerr = dv, lw = 1, error_kw = {'alpha': 0.3, 'color':'lightgray', 'lw': 1}, label = str( mode.data ) )
        
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
            vref = v.sel(ensemble='nudged', mode = 'fire0.0').squeeze()
            diff = (v.sel(ensemble='nudged', mode = mode) - vref).squeeze()
#            plt.plot(time + 0.15*(i-2.1), diff, marker = '*', lw = 0, ms = 20, mew = 2, mfc = 'w')

            if relative:
                diff = 100*diff / vref
            
            plt.bar(time + 0.15*(i-2.), diff, width = 0.04, alpha = 0.3, label = str( mode.data ))
            i += 1
                    
    
    return

######################################################################
######################################################################

