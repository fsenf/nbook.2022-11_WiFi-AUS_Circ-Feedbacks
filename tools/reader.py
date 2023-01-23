#!/usr/bin/env python

import os, glob
import xarray as xr
import numpy as np

xr.set_options(keep_attrs=True)


######################################################################
######################################################################


def open_ppecham_data(expname, data_type="rad", average_type="zonmean"):

    """
    
    OUTDATED:   Open Wifi-AUS time mean data for a certain experiment.

    Parameters
    ----------
    expname : str
        name of experiment subpatch, e.g.
        * 'wifiaus_ham_long_nu_gfas2020_inj14km_dj_ao_iaer0'
        * 'wifiaus_ham_long_nu_gfas2020_inj14km_dj_ao'
    
    data_type : str, optional
         (Default value = "rad")

    average_type : str, optional
         (Default value = "zonmean")


    Returns
    -------
    dset : xr.Dataset
        input data
    
    """

    data_path = "%s/wifi-aus/%s" % (os.environ["LOCAL_DATA_PATH"], expname)
    print("... open data from %s" % data_path)
    dset = xr.open_mfdataset(
        "%s/%s*%s.nc" % (data_path, average_type, data_type),
        chunks={"time": 1},
        combine="by_coords",
    )

    # add experiment name as new dimension
    dset = dset.expand_dims(["expname"])
    dset["expname"] = [
        expname,
    ]

    return dset


######################################################################
######################################################################


def open_ppecham_from_explist(
    explist,
    add_height=True,
    height_type="full",
    data_type="rad",
    average_type="zonmean",
):
    """
    OUTDATED:   Open WiFi-AUS experiments from a experiment list.

    Parameters
    ----------
    explist :
        
    add_height : {True, False}, optional
        switch if height info is added
        (Default value = True)

    height_type : {"full", "half"}, optional
        set either full or half level for a consistent level notation
        (Default value = "full")

    data_type : str, optional
        type of data stream read
        (Default value = "rad")

    average_type : str, optional
        shortcut for averaged data
        (Default value = "zonmean")

    Returns
    -------
    dset : xr.Dataset
        input data
 
    """

    dlist = []

    for expname in explist:
        d = open_ppecham_data(expname, data_type=data_type, average_type=average_type)

        dlist += [
            d.copy(),
        ]

    if add_height:
        full_height, half_height = get_mean_geoptential_height()

        if height_type == "full":
            height = full_height
        else:
            height = half_height

    dset = xr.concat(dlist, dim="expname")

    # set geometric height
    dset = dset.assign_coords({"lev": height})
    dset = dset.reindex({"lev": np.sort(dset.lev)})

    return dset


######################################################################
######################################################################


def get_mean_geoptential_height(averaging_dims=("time", "lat", "lon")):
    """
    OUTDATED : Input geopotential height.


    Parameters
    ----------
    averaging_dims : tuple or list, optional
       list of dimensions for which averaging is done 


    Returns
    -------
    height : xr.DataArray
        full height levels

    half_height : xr.DataArray
        half height levels
    """

    gfile = "/work/bb0883/m300279/WIFIAUS/wifiaus_ham_long_nu_gfas2020_inj14km_dj_ao/wifiaus_ham_long_nu_gfas2020_inj14km_dj_ao_202001.01_vphysc.nc"

    geo = xr.open_dataset(gfile)

    half_height = geo["geohm1"].mean(averaging_dims) / 9.81e3  # units:km
    height = geo["geom1"].mean(averaging_dims) / 9.81e3  # units:km

    half_height = half_height.rename({"lev_2": "lev"})

    half_height[0] = 80.0

    half_height.attrs = dict(long_name="average height", units="km")
    height.attrs = dict(long_name="average height", units="km")

    return height, half_height


######################################################################
######################################################################


def read_wifiaus_inner(
    data_dir, file_type="echam", run_type="nudged", average_type="zonmean"
):
    """
    Helper function to read WiFi-AUS data.


    Parameters
    ----------
    data_dir : str
        main data directory from which sub-directories 
        (each containing POSTPROC folders) are searched
        
    file_type : str, optional
        selects file type, i.e. ECHAM output stream (last part of filenames)
        (Default value = "echam")

    run_type : {"nudged", "ensemble"}, optional
        swtiches input methods between nudged and ensemble data
        (Default value = "nudged")

    average_type : {"zonmean", "tmean", "tzmean"}, optional
        Type of data to which averaging had been applied before.

        * "zonmean" : zonal average, hourly data
        * "tmean" : monthly data
        * "tzmean" : monthly and zonally mean data
        (Default value = "zonmean")
        

    Returns
    -------
    dset : xr.Dataset
        input data


    Notes
    -----
    Nudged and ensemble data have different data depths, i.e. there is an additional
    sub-directory for each ensmeble member. The structure is
    `{data_dir}/{fire_mode}/{ens_spec}/POSTPROC/{echamfile}.nc`

    """

    subdir_list = sorted(glob.glob(f"{data_dir}/*"))

    dlist = []
    for subdir in subdir_list:

        modename = subdir.split("/")[-1]

        # print( f'{subdir}/POSTPROC/zonmean*{file_type}.nc')

        filelist = sorted(glob.glob(f"{subdir}/POSTPROC/{average_type}*{file_type}.nc"))

        if len(filelist) > 0:
            d = xr.open_mfdataset(filelist, parallel=True)

            if run_type == "nudged":
                d = d.expand_dims(("mode", "ensemble"))
                d["mode"] = [modename]
                d["ensemble"] = ["nudged"]
                concat_dim = "mode"

            elif run_type == "ensemble":
                d = d.expand_dims("ensemble")
                ensname = modename
                concat_dim = "ensemble"
                d["ensemble"] = [ensname]

            dlist += [
                d,
            ]

    dset = xr.concat(dlist, dim=concat_dim)

    return dset


######################################################################
######################################################################


def read_wifiaus_nudged(data_dir, file_type="echam", average_type="zonmean"):

    """
    Reads nudged WiFi-AUS data.


    Parameters
    ----------
    data_dir : str
        main data directory from which sub-directories 
        (each containing POSTPROC folders) are searched
        
    file_type : str, optional
        selects file type, i.e. ECHAM output stream (last part of filenames)
        (Default value = "echam")

    average_type : {"zonmean", "tmean", "tzmean"}, optional
        Type of data to which averaging had been applied before.

        * "zonmean" : zonal average, hourly data
        * "tmean" : monthly data
        * "tzmean" : monthly and zonally mean data
        (Default value = "zonmean")
        

    Returns
    -------
     : xr.Dataset
        input data


    Notes
    -----
    A new dimension "ensemble" is introduced such that nudged data can be treated 
    in a similar way as ensemble members.
    """

    return read_wifiaus_inner(
        data_dir, file_type=file_type, run_type="nudged", average_type=average_type
    )


######################################################################
######################################################################


def read_wifiaus_ensemble(data_dir, file_type="echam", average_type="zonmean"):

    """
    Reads ensemble WiFi-AUS data.


    Parameters
    ----------
    data_dir : str
        main data directory from which sub-directories 
        (each containing POSTPROC folders) are searched
        
    file_type : str, optional
        selects file type, i.e. ECHAM output stream (last part of filenames)
        (Default value = "echam")

    average_type : {"zonmean", "tmean", "tzmean"}, optional
        Type of data to which averaging had been applied before.

        * "zonmean" : zonal average, hourly data
        * "tmean" : monthly data
        * "tzmean" : monthly and zonally mean data
        (Default value = "zonmean")
        

    Returns
    -------
    dset : xr.Dataset
        input data
    """


    subdir_list = sorted(glob.glob(f"{data_dir}/*"))

    dlist = []
    for subdir in subdir_list:

        modename = subdir.split("/")[-1]

        #        data_sub_dir = f'{data_dir}/{subdir}'
        d = read_wifiaus_inner(
            subdir, file_type=file_type, run_type="ensemble", average_type=average_type
        )

        d = d.expand_dims("mode")
        d["mode"] = [modename]
        concat_dim = "mode"

        dlist += [
            d,
        ]

    dset = xr.concat(dlist, dim=concat_dim)

    return dset


######################################################################
######################################################################


def read_wifiaus_combination(
    base_data_dir="/work/bb1262/data/echam-ham/wifi-aus/",
    file_type="echam",
    average_type="zonmean",
    time_range=slice("2020-01", "2020-03"),
):
    """
    Wrapper that reads both, nudged and ensemble data.


    Parameters
    ----------
    base_data_dir : str, optional
        base path under which nudged and ensemble data are stored
         (Default value = "/work/bb1262/data/echam-ham/wifi-aus/")

    file_type : str, optional
        selects file type, i.e. ECHAM output stream (last part of filenames)
        (Default value = "echam")

    average_type : {"zonmean", "tmean", "tzmean"}, optional
        Type of data to which averaging had been applied before.

        * "zonmean" : zonal average, hourly data
        * "tmean" : monthly data
        * "tzmean" : monthly and zonally mean data
        (Default value = "zonmean")
 
    time_range : slice, optional
        Time slices to be applied to all data
        (Default value = slice("2020-01", "2020-03"))
        

    Returns
    -------
    dset : xr.Dataset
        input data
 
    """
    # ### Open Nudged Data
    data_dir = f"{base_data_dir}/wifiaus_ham_long_nu_gfas2020_injtrp+1_pcb_ao"

    nudged = read_wifiaus_nudged(
        data_dir, file_type=file_type, average_type=average_type
    )

    # ### Open Ensemble Data
    data_dir = f"{base_data_dir}/wifiaus_ham_long_gfas2020_injtrp+1_pcb_ao_ens"
    ensdat = read_wifiaus_ensemble(
        data_dir, file_type=file_type, average_type=average_type
    )

    enslist = ensdat.ensemble

    # ###  Combine the two
    dset = xr.concat([nudged, ensdat], dim="ensemble")
    dset = dset.sel(time=time_range)

    return dset


######################################################################
######################################################################


def wifiaus_add_vars(dset):

    """
    Adds some net TOA radiation fluxes to the data. 

    Parameters
    ----------
    dset : xr.Dataset
        in-/output dataset, needs to include TOA radiation fluxes
        

    Returns
    -------
    dset : xr.Dataset
        output is written onto input

    """

    # ## Add Variables
    # ### Net ERF
    dset["net_toa_clear"] = dset["sraf0"] + dset["traf0"]
    dset["net_toa_clear"].attrs = {
        "long_name": "net TOA radiation (clear-sky)",
        "units": "W/m**2",
    }

    dset["net_toa"] = dset["srad0"] + dset["trad0"]
    dset["net_toa"].attrs = {
        "long_name": "net TOA radiation",
        "units": "W/m**2",
    }

    return dset


######################################################################
######################################################################
