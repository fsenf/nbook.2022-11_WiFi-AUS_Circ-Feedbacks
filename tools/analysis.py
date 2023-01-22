#!/usr/bin/env python

import numpy as np
import reader
import xarray as xr

xr.set_options(keep_attrs=True)

######################################################################
# Constants ----------------------------------------------------------
######################################################################

# from mo_physical_constants.f90:90:  REAL(wp), PARAMETER :: earth_radius           = 6.371229e6_wp    !! [m]    average radius
earth_radius = 6.371229e6
a = earth_radius
g = 9.81
day = 24 * 3600.0
Omega = 2 * np.pi / day


######################################################################
# Circulation Analysis  ----------------------------------------------
######################################################################


def calc_stream_function(v, dset, method="p-coordinate", invert_p=True):

    """

    Calculates mass stream function.


    Parameters
    ----------
    v : xr.DataArray
        meridonal velocity already averaged (time, lon)

    dset : xr.Dataset
        this dataset need to contain surface pressure

    method : {"p-coordinate", "z-coordinate"}, optional
         switch between different vertical coordinates

         - "z-coordinate": height coordinate, currently broken!
         - "p-coordinate": pressure coordinate

    invert_p : {True, False}
         if pressure coordinate is inverted


    Returns
    -------
    Psi : xr.DataArray
        mass streamfunction
    """

    # define constants and coordinates
    phi = np.deg2rad(v.lat)
    cosphi = np.cos(phi)

    if method == "z-coordinate":  # maybe broken ...

        # get height increments
        full_height, half_height = reader.get_mean_geoptential_height(
            averaging_dims=("time", "lon")
        )

        dz = np.abs(half_height.diff("lev")) * 1e3  # units: m
        dz = dz.assign_coords({"lev": full_height})

        rhov = v["rhov"].mean(
            [
                "time",
            ]
        )

        # integration
        kernel = rhov * dz
        kernel = kernel.reindex(lev=np.sort(dz.lev))
        integral = kernel.cumsum("lev")

    elif method == "p-coordinate":

        pressure_mask = dset["aps"] > dset.plev

        # mask meridional wind and sort in descending order
        vm = v.where(pressure_mask, 0)

        if invert_p:
            vm_re = vm.reindex(plev=np.sort(vm.plev.values)[::-1])
        else:
            vm_re = vm

        # calculate the vertical integral
        fac = 1 / g
        integral = fac * vm_re.cumulative_integrate("plev").where(
            pressure_mask
        )  # .sel ( time = '2020-01').mean('time')

    Psi = (
        2 * np.pi * a * cosphi * integral
    )  # ReRun_Mean_Circulation_ResponseReRun_Mean_Circulation_Response/ 1e10   # units 1e10 kg s-1

    return Psi


######################################################################
######################################################################


def calc_vstar(dset):

    """

    Calculates residual meridional velocity.


    Parameters
    ----------
    dset : xarray.Dataset
        input dataset containing meanflow vars and covariances
        stored at pressure levels


    Returns
    -------
    v_star : xarray.Dataset
        residual meridional velocity


    Notes
    -----
    Following calculations are done:

    $$
    v^* = \overline{v} -
    \partial_p\bigg(
    \frac{\;\overline{v`\theta`}}
    {\partial_p\overline{\theta}} \bigg)
    = \overline{v} + v_\mathrm{eddy}
    $$
    """

    # mean vars (w.r.t to zonal average)
    v = dset["v"]
    theta = dset["tpot"]

    # get stablity
    Gamma = theta.differentiate("plev")

    # calculate covariance
    F_vtheta = dset["vtheta"] - v * theta

    # combine mean and eddy components
    v_eddy = -(F_vtheta / Gamma).differentiate("plev")
    v_star = v + v_eddy

    return v_star


######################################################################
######################################################################


def horizontal_divergence(f, latitude_name="lat", exponent=1):

    """

    Calculates horizontal divergence (only meridional direction) of a flux f.


    Parameters
    ----------
    f : xr.DataArray
        input flux

    latitude_name : string, optional
        name of latitude in `f`

    exponent : integer, optional
        exponent of cosphi, maybe quadratic of tensor products


    Returns
    -------
    div_f : xr.DataArray
        divergance of flux f in spherical coordinates


    Notes
    -----
    Metric correction of the generalized vertical coordinate are ignored.
    """

    # latitudes
    phi = np.deg2rad(f[latitude_name])
    cosphi = np.cos(phi) ** exponent

    # derivatives
    d_cosphi_f = (cosphi * f).differentiate(latitude_name)
    dphi = (phi).differentiate(latitude_name)

    return 1 / (a * cosphi) * d_cosphi_f / dphi


######################################################################
######################################################################


def calculate_eddy_omega(d, level_name="plev"):

    """

    Calculates eddy part of residual pressure velocity.


    Parameters
    ----------
    d : xr.Dataset
        complete dataset containing wind and pot. temperature

    level_name : string, optional
        name of the vertical level coordinate


    Returns
    -------
    omega_e : xr.DataArray
        eddy part of residual pressure velocity

    """

    # calculate heat flux
    v = d["v"]
    theta = d["tpot"]

    # get stablity

    # calculate covariance
    F_vtheta = d["vtheta"] - v * theta

    # and its divergence
    div_F_vtheta = horizontal_divergence(F_vtheta)

    # calculate stability
    Gamma = theta.differentiate("plev")

    # put all together
    omega_e = div_F_vtheta / Gamma

    return omega_e


######################################################################
######################################################################


def calculate_residual_omega(d, level_name="plev"):

    """

    Calculates residual pressure velocity.


    Parameters
    ----------
    d : xr.Dataset
        complete dataset containing wind and pot. temperature

    level_name : string, optional
        name of the vertical level coordinate


    Returns
    -------
    omega_star : xr.DataArray
        residual pressure velocity


    Notes
    -----
    This formula is implemented
    $$
    \omega^* = \overline{\omega} +
    \frac{1}{a\cos\varphi}
    \frac{\partial_\varphi\big( \cos\varphi \;\overline{v`\theta`} \big) }
    {\partial_p\overline{\theta}}
    = \overline{\omega} + \omega_\mathrm{eddy}
    $$
    """

    # get vertical velocity in pressure coordinates
    omega_bar = d["omega"]

    # calculate omega components
    omega_eddy = calculate_eddy_omega(d, level_name=level_name)

    omega_star = omega_bar + omega_eddy

    return omega_star


######################################################################
######################################################################


def calc_meridional_EPF(d):

    """
    Calculates meridional component of Eliassen-Palm Flux (EPF). 


    Parameters
    ----------
    d : xr.Dataset
        complete dataset containing winds


    Returns
    -------
    EPF_phi : xr.DataArray
        meridional EPF component.

    """

    # define coordinates
    phi = np.deg2rad(d.lat)
    cosphi = np.cos(phi)

    # calculate momentum flux
    u = d["u"]
    v = d["v"]

    # calculate covariance
    F_uv = d["uv"] - v * u

    EPF_phi = -a * cosphi * F_uv

    return EPF_phi


######################################################################
######################################################################


def calc_vertical_EPF(d):
 
    """
    Calculates vertical component of Eliassen-Palm Flux (EPF). 


    Parameters
    ----------
    d : xr.Dataset
        complete dataset containing winds and pot. temperature


    Returns
    -------
    EPF_p : xr.DataArray
        vertical EPF component.

    """


    # define coordinates
    phi = np.deg2rad(d.lat)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    # calculate heat flux
    v = d["v"]
    theta = d["tpot"]

    # calculate covariance
    F_vtheta = d["vtheta"] - v * theta

    # calculate stability
    Gamma = theta.differentiate("plev")

    f = 2 * Omega * sinphi

    EPF_p = a * f * cosphi * F_vtheta / Gamma

    return EPF_p


######################################################################
######################################################################


def EPF_divergence(d):

    """
    Calculates divergence of Eliassen-Palm Flux (EPF). 


    Parameters
    ----------
    d : xr.Dataset
        complete dataset containing winds and pot. temperature


    Returns
    -------
    div_EPF : xr.DataArray
        divergence of EPF

    """


    # define coordinates
    phi = np.deg2rad(d.lat)
    cosphi = np.cos(phi)

    # EPF components
    EPF_phi = calc_meridional_EPF(
        d,
    )
    EPF_p = calc_vertical_EPF(
        d,
    )

    div_EPF = horizontal_divergence(EPF_phi) + EPF_p.differentiate("plev")

    return div_EPF


######################################################################
######################################################################


######################################################################
# Statistics and Averaging -------------------------------------------
######################################################################


def ens_stat(vin):

    """
    Calculates ensemble statistics.


    Parameters
    ----------
    vin : xr.Dataset
        input data with an ensemble dimension


    Returns
    -------
    vstat : xr.Dataset
        ensemble statistics gathered along the new 
        coordinate `stats`

    """

    v = vin.drop_sel({"ensemble": "nudged"})

    ensmean = v.mean("ensemble")
    ensvar = v.var("ensemble")

    fire_perturb_list = ["fire1.0", "fire2.0", "fire3.0", "fire5.0"]

    mean_reference = ensmean.sel(mode="fire0.0")
    mean_difference = (
        ensmean.sel(mode=fire_perturb_list) - mean_reference
    )  # / fire_scaling

    enslist = v.ensemble.data
    nens = len(enslist)

    var_difference = ensvar.sel(mode=fire_perturb_list) + ensvar.sel(
        mode="fire0.0"
    )  # / fire_scaling
    conf_difference = 2 * np.sqrt(var_difference / nens)

    upper = mean_difference + conf_difference
    lower = mean_difference - conf_difference

    mean = mean_difference.expand_dims("stats")
    mean["stats"] = ["mean"]

    conf = conf_difference.expand_dims("stats")
    conf["stats"] = ["confidence"]

    upper = upper.expand_dims("stats")
    upper["stats"] = ["upper"]

    lower = lower.expand_dims("stats")
    lower["stats"] = ["lower"]

    ref = mean_reference.expand_dims("stats") * xr.ones_like(
        mean
    )  # expand ref to each fire mode
    ref["stats"] = ["reference"]

    vstat = xr.concat([mean, conf, upper, lower, ref], dim="stats")

    vstat.attrs = v.attrs

    return vstat


######################################################################
######################################################################


def stats_and_nudged(v):

    """
    Combines ensembles statistics with nudged data.


    Parameters
    ----------
    v : xr.Dataset
        input data with an ensemble dimension


    Returns
    -------
    vstat : xr.Dataset
        ensemble statistics gathered along the new 
        coordinate `stats` + nudged data included there as well



    """

    v_stats = ens_stat(v)

    v_nudged = v.sel(ensemble="nudged")
    v_nudged = v_nudged.expand_dims("stats")

    v_nudged_ref = v_nudged.sel(mode="fire0.0")
    dv_nudged = v_nudged.sel(mode=v_stats.mode) - v_nudged_ref

    dv_nudged["stats"] = [
        "nudged",
    ]

    v_nudged_ref = v_nudged_ref * xr.ones_like(
        dv_nudged
    )  # get the right shape, incl. modes
    v_nudged_ref["stats"] = [
        "nudged-reference",
    ]

    v_stats = xr.concat([v_stats, dv_nudged, v_nudged_ref], dim="stats")

    return v_stats  # .squeeze()


######################################################################
######################################################################


def weighted_mean(v):
    
    """
    Calculates mean weighted by cos(phi) (proportional to surface area on Earth).


    Parameters
    ----------
    v : xr.Dataset
        input data


    Returns
    -------
    v_mean : xr.Dataset
        data averaged along latitude with area weight
    """

    lat = v.lat
    phi = np.deg2rad(lat)

    cosphi = np.cos(phi)
    w = cosphi

    v_mean = (v * w).mean("lat") / w.mean("lat")

    return v_mean


######################################################################
######################################################################


def glob_mean(v):

    """
    Calculates global mean.


    Parameters
    ----------
    v : xr.Dataset
        input data


    Returns
    -------
    vm : xr.Dataset
        globally averaged data


    Notes
    -----
    weighted by cos(phi) (proportional to surface area on Earth).
    """

    return weighted_mean(v)


######################################################################
######################################################################


def sh_mean(v):

    """
    Calculates Southern hemispheric mean.


    Parameters
    ----------
    v : xr.Dataset
        input data


    Returns
    -------
    vm : xr.Dataset
        globally averaged data


    Notes
    -----
    weighted by cos(phi) (proportional to surface area on Earth).
    """

    vs = v.sel(lat=slice(0, -90))
    return weighted_mean(vs)


######################################################################
######################################################################


def nh_mean(v):


    """
    Calculates Northern hemispheric mean.


    Parameters
    ----------
    v : xr.Dataset
        input data


    Returns
    -------
    vm : xr.Dataset
        globally averaged data


    Notes
    -----
    weighted by cos(phi) (proportional to surface area on Earth).
    """

    vs = v.sel(lat=slice(90, 0))
    return weighted_mean(vs)


######################################################################
######################################################################


def averaged_dset_for_latranges(d, latranges):

    """
    Calculates averages for predefined latitude ranges.


    Parameters
    ----------
    v : xr.Dataset
        input data

    latranges : list of list
        list of latitude ranges 
        
        - structure `[[lat1, lat2], [lat2, lat3], ...]`
        - sort from North to South for ECHAM!


    Returns
    -------
    drange : xr.Dataset
        data averaged over latitude bands

        - new coordinate `latrange`is introduced
        - latrange is converted to string and used as key


    Notes
    -----
    weighted by cos(phi) (proportional to surface area on Earth).
    """

    drange = []
    for latrange in latranges:
        ds = d.sel(lat=slice(*latrange))
        d_weighted_ave = weighted_mean(ds)
        d_weighted_ave = d_weighted_ave.expand_dims("latrange")
        d_weighted_ave["latrange"] = [
            str(latrange),
        ]

        drange += [
            d_weighted_ave,
        ]

    drange = xr.concat(drange, dim="latrange")
    return drange


######################################################################
######################################################################


def averaged_dset(d, diving_lat=20):
    
    """
    Calculates averages for three predefined latitude ranges. Ranges are divided such that
    (90°S - phi°S, phi°S - phi°N, phi°N - 90°N) results wher phi is the `diving_lat`.


    Parameters
    ----------
    v : xr.Dataset
        input data

    diving_lat : float, optional
        dividing latitude

    Returns
    -------
    drange : xr.Dataset
        data averaged over latitude bands

        - new coordinate `latrange`is introduced
        - latrange is converted to string and used as key


    Notes
    -----
    weighted by cos(phi) (proportional to surface area on Earth).
    """

    if type(diving_lat) == type(20):
        # lat is stored backwards
        sh_range = [-diving_lat, -90]
        tropics_range = [diving_lat, -diving_lat]
        nh_range = [90, diving_lat]

        latranges = [sh_range, tropics_range, nh_range]

    else:
        lats = np.hstack([-90, diving_lat, 90])

        nlats = len(lats)

        latranges = []
        for i in range(nlats - 1):

            i1 = i + 1
            i2 = i

            lrange = [lats[i1], lats[i2]]
            latranges += [
                lrange,
            ]

    return averaged_dset_for_latranges(d, latranges)


######################################################################
######################################################################

