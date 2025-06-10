# %% 0) Imports
import os
import copy

import pandas as pd
import numpy as np
import math
import random
import decimal
from nlmpy.nlmpy import randomElementNN, mpd, classifyArray

from skimage.filters import gaussian
import time as time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from BOSSE.helpers import zone_snum, div_zeros, print_et
from BOSSE.scsim_emulation import MPL_pred, correct_var_names
from BOSSE import scsim_species as sp
from BOSSE import scsim_gsi_model as gsi
from BOSSE import scsim_meteo as mt
from BOSSE.plotter import do_plot_meteo_ts, do_show_bosse_map, do_plot_pft_map

# %% 1) Ancillary functions
def update_wr(meteo__, meteo_av_, meteo_av30_, FC_):
    # Update the relative soil moisture content
    # The relative soil moisture, which is defined as the ratio (in
    # percentage) of water in weight to the field capacity in weight of
    # the measured soil layer. Here used as a fraction. Correct SMC which is
    # also a fraction
    meteo__['wr'] = meteo__['SMC'] / FC_
    meteo_av_['wr'] = meteo_av_['SMC'] / FC_
    meteo_av30_['wr'] = meteo_av30_['SMC'] / FC_

    return (meteo__, meteo_av_, meteo_av30_)


# %% 2) Spatial simulation
def generate_pft_map(sp_map, sp_id, veg_, sp_pft):
    pft_map = np.zeros(sp_map.shape, dtype=int)
    pft_col = np.zeros((sp_map.shape[0], sp_map.shape[1], 3), dtype=int)
    for i_, spi_ in enumerate(sp_id):
        I_ = np.where(sp_map == spi_)
        Ipft = veg_['pft_in'].index(sp_pft[i_])
        pft_map[I_[0], I_[1]] = Ipft
        pft_col[I_[0], I_[1], :] = (
            np.array(mcolors.to_rgb(veg_['pft_col'][Ipft])[:3]) * 255
                ).astype(int).reshape(1, 1, -1)
    
    return(pft_map, pft_col)


def generate_map(simnum_, inputs_, paths_, P_pft, veg_, meteo_):
    # Foreseen number of species, could change during scene colonization
    s_max = np.random.randint(1, inputs_['S_max'] + 1)

    # Filters PFT according to biome, GSI parameters and meteo conditions.
    sp_pft_ok, P_pft_ok = filter_pft_meteo(veg_, P_pft, meteo_)

    # Assign randomly a number of PFTs
    num_pft = random.sample(range(len(sp_pft_ok)), 1)[0] + 1

    # Assing species PFTs from a subsample of PFTs present in the scene
    I_ = random.sample(range(len(sp_pft_ok)), num_pft)
    prob_I = (P_pft_ok.iloc[I_, 1] / P_pft_ok.iloc[I_, 1].sum()).values
    sp_pft = np.random.choice([sp_pft_ok[i_] for i_ in I_], size=s_max,
                              p=prob_I)

    # Foreseen species abundances, could change during scene colonization
    sp_ab_0 = np.random.random(s_max)
    sp_ab_0 = sp_ab_0 / np.sum(sp_ab_0)

    # Number of seeds per species. Ensure it cannot be > scene_sz ** 2
    # Use a separated RNG so that the rest of the random values are
    # the same
    nRow, nCol = inputs_['scene_sz'], inputs_['scene_sz']
    if inputs_['sp_pattern'] == 'clustered':
        env_background = randomElementNN(
            nRow, nCol, s_max + np.random.randint(0, int((nRow + nCol) // 2)))
        sp_map = classifyArray(env_background, sp_ab_0).astype(int)
    elif inputs_['sp_pattern'] == 'intermediate':
        h_ = np.random.uniform(1., 1.5)
        env_background = mpd(nRow, nCol, h=h_)
        sp_map = classifyArray(env_background, sp_ab_0).astype(int)
    elif inputs_['sp_pattern'] == 'even':
        h_ = np.random.uniform(0, .1)
        env_background = mpd(nRow, nCol, h=h_)
        sp_map = classifyArray(env_background, sp_ab_0).astype(int)
    else:
        raise ValueError('Simulation spatial pattern not recognized.')

    soil_map = gaussian(env_background, sigma=np.random.uniform(.5, 1.5),
                        mode='mirror', preserve_range=False)

    # Update the abundances
    sp_id, freq_ = np.unique(sp_map.reshape(-1), return_counts=True)
    sp_ab = freq_ / inputs_['scene_sz']**2

    # Make sure the original number of species is kept for comparability
    # between spatial patters
    while sp_id.shape[0] < s_max:
        sp_missing = [i_ for i_ in range(s_max) if i_ not in sp_id]
        for i_, spi in enumerate(sp_missing):
            s_most = sp_id[freq_ == max(freq_)]
            if s_most.shape[0] > 1:
                s_most = s_most[0]
            I_ = np.where(sp_map == s_most)
            pos_ = np.random.choice(range(I_[0].shape[0]))
            sp_map[I_[0][pos_], I_[1][pos_]] = spi
            # Update the abundances
            sp_id, freq_ = np.unique(sp_map.reshape(-1), return_counts=True)
            sp_ab = freq_ / inputs_['scene_sz']**2
    while sp_id.shape[0] > s_max:
        s_most = sp_id[freq_ == max(freq_)]
        s_least = sp_id[freq_ == min(freq_)]
        if s_most.shape[0] > 1:
            s_most = s_most[0]
        I_ = np.where(sp_map == s_most)
        pos_ = np.random.choice(range(I_[0].shape[0]))
        sp_map[I_[0], I_[1]] = s_least
        # Update the abundances
        sp_id, freq_ = np.unique(sp_map.reshape(-1), return_counts=True)
        sp_ab = freq_ / inputs_['scene_sz']**2

    # Finally, short species ID by abundance in oreder to keep more comparable
    # simulations with different spatial patterns. Use mergesort to preserve the
    # relative order of equal valuesensure and ensure reproducibility between
    # different machines
    sp_map_tmp = 1E5 + sp_map
    is_ = np.argsort(sp_ab, kind='mergesort')[::-1]
    for i_, idi_ in enumerate(is_):
        I_ = sp_map_tmp == 1E5 + idi_
        sp_map[I_] = i_

    # Generate the PFT map from the species
    (pft_map, pft_col) = generate_pft_map(sp_map, sp_id, veg_, sp_pft)
    
    if inputs_['inspect']:
        title_lb = ('Scene %d. Smax = %d. %s' % (
            simnum_, s_max, inputs_['sp_pattern']))
        fname = (paths_['2_out_folder_plots'] +'Map_species_%s' % (
            zone_snum(simnum_, inputs_['clim_zone'])))
        do_show_bosse_map(sp_map, title_lb=title_lb, add_colorbar=True,
                          cmap='tab20', fname=fname, plt_show=False)
        
        fname = (paths_['2_out_folder_plots'] +'Map_PFT_%s' % (
            zone_snum(simnum_, inputs_['clim_zone'])))
        do_plot_pft_map(pft_col, veg_, title_lb=title_lb, add_colorbar=False,
                        cmap='viridis', fname=fname, plt_show=False)

    return(sp_map, pft_map, s_max, sp_ab, sp_id, sp_pft, soil_map)


def filter_pft_meteo(veg_, P_pft, meteo_):
    # Determine which PFTs can habitate in the climate with the criteria that
    # inflection point of the GSI functions must be within the ranges of the
    # meteo data
    # Get meteo conditions
    wav_ = gsi.gsi_smooth_input(gsi.get_water_avail(
        veg_, np.array(veg_['pft_in']), meteo_))
    Rin_ = gsi.gsi_smooth_input(meteo_['Rin'].values)
    Ta_ = gsi.gsi_smooth_input(meteo_['Ta'].values)

    # Compare ranges with "base" or GSI inflection point. If none is selected, 
    # keep the most competitive
    Ipft_wav = np.percentile(wav_, 95, axis=0) >= veg_['base_water']
    Ipft_rin = np.percentile(Rin_, 95) >= veg_['base_light']
    Ipft_tmin = np.percentile(Ta_, 95) >= veg_['base_tmin']
    Ipft_heat = np.percentile(Ta_, 5) <= veg_['base_heat']
    I_all = np.stack((Ipft_wav, Ipft_rin, Ipft_tmin, Ipft_heat), axis=0)

    # If none of the PFTs meets the requirements, select the ones which are the
    # closest to the limiting conditions
    Ipft_ = np.all(I_all, axis=0)
    if any(Ipft_) is False:        
        if any(Ipft_wav) is False:
            dif_ = veg_['base_water'] - wav_.max(axis=0)
            Ipft_wav[dif_ == min(dif_)] = True
            
        if any(Ipft_rin) is False:
            dif_ = veg_['base_light'] - Rin_.max(axis=0)
            Ipft_rin[dif_ == min(dif_)] = True
            
        if any(Ipft_tmin) is False:
            dif_ = veg_['base_tmin'] - Ta_.max()
            Ipft_tmin[dif_ == min(dif_)] = True
            
        if any(Ipft_heat) is False:
            dif_ = Ta_.min() - veg_['base_heat']
            Ipft_heat[dif_ == min(dif_)] = True
                   
        I_all = np.stack((Ipft_wav, Ipft_rin, Ipft_tmin, Ipft_heat), axis=0)
        Ipft_ = np.all(I_all, axis=0)
        
    pft_ok = [veg_['pft_in'][i_] for i_, sel_ in enumerate(Ipft_) if sel_]
    
    P_pft_ok = P_pft[Ipft_]

    return (pft_ok, P_pft_ok)


# %% 3) Soil models
def rss_model(M_, SMC_FC, level_):
    # Input the relative soil water content and the empirical rss level
    # that is assigned to the site
    rss_ = M_(np.stack((SMC_FC.reshape(-1), level_.reshape(-1)), axis=1)
              ).reshape(SMC_FC.shape)

    return (rss_)


def generate_soil_parameters_map(inputs_, Soil_vars, Soil_LB, Soil_UB,
                                 SMC_):
    nvar_ = len(Soil_vars)
    sz_ = inputs_['scene_sz']
    soil_range = Soil_UB - Soil_LB
    soil_map = np.repeat(np.repeat((Soil_LB + soil_range *
                         np.random.random(nvar_)).reshape(1, 1, nvar_),
                         sz_, axis=0), sz_, axis=1)
    # Set FC within the 90-110 % of the max SMC to avoid weird water-related
    # phenology. It is bounded with the FC limits
    soil_map[:, :, -1] = np.min((Soil_UB[-1],
                                np.max((Soil_LB[-1],
                                       SMC_ * np.random.uniform(1., 1.5)))))

    return (soil_map)


# %% 4) Ecosystem funcitons
def gCs2mumolCs():
    # Conversion factor from gC m-2 day-1 to mumolCO2 m-2 s-1
    # From https://rdrr.io/cran/bigleaf/src/R/unit_conversions.r#sym-gC.to.umolCO2
    c_g2kg = 0.001
    c_days2seconds  = 86400.
    c_Cmol       = 0.012011
    c_mol2umol =  1e06
    uds_conv = ((c_g2kg / c_days2seconds)) / c_Cmol * c_mol2umol
    
    return(uds_conv)


def umolCs2gCs2():
    # Conversion factor from gC m-2 day-1 to mumolCO2 m-2 s-1
    # From https://rdrr.io/github/lhmet-forks/bigleaf/man/umolCO2.to.gC.html
    c_umol2mol = 1e-06
    c_Cmol       = 0.012011
    c_kg2g = 1e03
    c_days2seconds  = 86400.
    uds_conv = c_umol2mol * c_Cmol * c_kg2g * c_days2seconds
    
    return(uds_conv)


def get_Reco(Rlai0, alai, LAImax, k2, E0, alpha_, k_, GPP, Tair, P,
             Tref=288.15, T0=227.13):
    decimal.getcontext().prec = 40
    uds_conv = gCs2mumolCs()
    
    # Eq 9 in Migliavacca et al. 2010. 
    R0_ = ((Rlai0 + alai * LAImax) + (k2 *(GPP / uds_conv)))
    # As Tair equals T0, the invese of the
    # difference tends to 1000. Thus, avoid the division by 0 outputing the
    # limit instead.
    TairK = np.max((Tair + 273.15, T0))
    R1_ = np.exp(E0 * ((1 / (Tref - T0)) -
                       div_zeros(np.ones(1), TairK - T0, out_=1000.)))
    
    # Add a filter to prevent extremely large values
    sign = np.ones(k_.shape)
    sign[(k_ + P * (1 - alpha_)) < 0.] = -1.
    R2_ = (((alpha_ * k_) + P * (1 - alpha_)) /
           (sign * np.maximum(.01, np.abs(k_ + P * (1 - alpha_)))))

    return (uds_conv * R0_ * R1_ * R2_)


# %% 5) Ancillary simulations
def generate_clouds(sz_, cloud_cover=-1, cloud_th=70):
    # Seeds server
    if cloud_cover == -1:
        cloud_cover = float(np.random.randint(0, cloud_th))
    max_cloud = (cloud_cover / 100.) * (sz_ ** 2)

    # Cloud seeds
    num_seeds = np.random.randint(1, sz_)
    seeds_ = random.sample(range(sz_ ** 2), num_seeds)

    # Image coordinates
    xg_, yg_ = np.meshgrid(np.arange(sz_, dtype=int),
                           np.arange(sz_, dtype=int))
    xg_ = xg_.reshape(-1)
    yg_ = yg_.reshape(-1)

    # Wind effect. This makes the clouds less circular
    vx_ = np.random.uniform(.2, 1.7) * random.sample([-1, 1], 1)[0]
    vy_ = np.random.uniform(.2, 1.7) * random.sample([-1, 1], 1)[0]

    # Populate map
    cloud_map = np.zeros((sz_, sz_), dtype=bool)
    sim_cloud, k_ = 0, 0
    while np.any(sim_cloud < max_cloud):
        sp_order_tmp = random.sample(seeds_, num_seeds)

        if k_ > 0:
            # Search area to colonize, can be squared or circular
            dx, dy = np.meshgrid(np.arange(-k_, k_+1, dtype=int),
                                 np.arange(-k_, k_+1, dtype=int))
            dx = dx.reshape(-1)
            dy = dy.reshape(-1)
            # Make a cricle
            tmp_dist = (dx ** 2 + dy ** 2) ** .5
            dx = dx[tmp_dist <= k_]
            dy = dy[tmp_dist <= k_]
        else:
            dx, dy = 0, 0

        for j_ in sp_order_tmp:
            if k_ == 0:
                x_, y_ = xg_[j_], yg_[j_]
                cloud_map[y_, x_] = True
            else:
                # Generate a cricular or a squared window to colonize
                x_ = xg_[j_] + dx + int(vx_ * k_)
                y_ = yg_[j_] + dy + int(vy_ * k_)

                # Find pixels inside the scene
                I_ = np.where((np.abs(x_) < sz_) &
                              (np.abs(y_) < sz_))[0]
                if I_.shape[0] > 0:
                    if I_.shape[0] > 4:
                        I_ = random.sample(I_.tolist(),
                                           int(I_.shape[0] *
                                               np.random.uniform(.5, .9)))
                    x_ = x_[I_]
                    y_ = y_[I_]
                    cloud_map[y_, x_] = True
            sim_cloud = np.sum(cloud_map)
            if sim_cloud > max_cloud:
                break
        k_ += 1

    return (cloud_map, sim_cloud / (sz_ ** 2))


# %% 6) Time series
def get_traits_map_t(PT_map_min, PT_map_delta, sp_map, sp_id, GSI_,
                     out_sts=False):
    pt_map = np.zeros(PT_map_min.shape) * np.nan
    for i_, ind_ in enumerate(sp_id):
        I_ = sp_map == ind_
        gsi_i = np.repeat(GSI_[i_].reshape(1, -1), I_.sum(), axis=0)
        pt_map[I_] = PT_map_min[I_] + gsi_i * PT_map_delta[I_]

    if out_sts is True:
        pt_mean = np.zeros((len(sp_id), PT_map_min.shape[-1])) * np.nan
        pt_min = np.zeros((len(sp_id), PT_map_min.shape[-1])) * np.nan
        pt_max = np.zeros((len(sp_id), PT_map_min.shape[-1])) * np.nan
        for i_, ind_ in enumerate(sp_id):
            I_ = sp_map == ind_
            pt_mean[i_] = np.mean(pt_map[I_], axis=0)
            pt_min[i_] = np.min(pt_map[I_], axis=0)
            pt_max[i_] = np.max(pt_map[I_], axis=0)

        return (pt_map, pt_mean, pt_min, pt_max)
    else:
        return (pt_map)


def update_scene_timestamp(t_, indx_day, PT_map_min, PT_map_max, PT_map_delta,
                           sp_map, sp_id, GSI_all, GSI_wav, X_, PT_mean, PT_min,
                           PT_max, meteo_, M_rss, Soil_UB, I_pt, I_sf, I_smc,
                           I_fc, I_rss, I_rssl, I_lai, I_ang, I_met, all_vars,
                           lai020_pt=False, lai020_ang=False, lai020_met=False):
    # Update plant traits daily only, and GSI
    # Floor the index to integer
    td_ = indx_day[t_].astype(int)
    if ((t_ == 0) or td_ != indx_day[t_-1]):
        # Plant traits
        X_[:, :, I_pt], PT_mean[td_], PT_min[td_], PT_max[td_] = (
            get_traits_map_t(PT_map_min, PT_map_delta, sp_map, sp_id,
                             GSI_all[td_], out_sts=True))

    # Update the vegetation stress factor, using the specie's depednent 
    # GSI index value dependent on water availability as it's value.
    # Average the GSI_wav value for all the plan traits, for each species
    for i_, id_ in enumerate(sp_id):
        I_ = np.where(sp_map == id_)
        X_[I_[0], I_[1], I_sf] = GSI_wav[td_, i_, :].mean()

    # Update meteo and environmenal conditions for each timestamp
    # Update Soil Moisture
    X_[:, :, I_smc] = np.maximum(np.minimum(meteo_.iloc[t_]['SMC'],
                                            Soil_UB[-2]), 5.)
    X_[:, :, I_rss] = rss_model(M_rss, X_[:, :, I_smc] / X_[:, :, I_fc],
                                X_[:, :, I_rssl])
    # Update Meteo
    for j_ in I_met:
        X_[:, :, j_] = meteo_.iloc[t_][all_vars[j_]]
    
    # Set tts
    X_[:, :, I_ang[0]] = meteo_.iloc[t_][all_vars[I_ang[0]]]
    
    ## Missing. Set off-nadir view angles
   
    if any((lai020_pt, lai020_ang, lai020_met)):
        IL0 = np.where(X_[:, :, I_lai] == 0.)
        
        if IL0[0].shape[0] > 0:
            ones_i = np.ones(IL0[0].shape[0], dtype=int)
            # When LAI == 0, set all the plant traits to 0
            if lai020_pt:
                for i_ in I_pt:
                    X_[IL0[0], IL0[1], ones_i*i_] = 0.
                X_[IL0[0], IL0[1], ones_i*I_sf] = 0.
            # X_[IL0[0], IL0[1]][:, I_pt]
            # When LAI == 0, set all the plant traits to 0
            if lai020_ang:
                for i_ in I_ang:
                    X_[IL0[0], IL0[1], ones_i*i_] = 0.
            # X_[IL0[0], IL0[1]][:, I_ang]
            # When LAI == 0, set all the plant traits to 0
            if lai020_met:
                for i_ in I_met:
                    X_[IL0[0], IL0[1], ones_i*i_] = 0.
            # X_[IL0[0], IL0[1]][:, I_met] = 0.

    return (X_, PT_mean, PT_min, PT_max)


def plot_PT_timeseries_perSp(df_PT_, PT_vars, sp_id, sp_pft, veg_, ts_days,
                             inputs_, paths_, simnum_, PT_LB, PT_UB):
    x_ = np.arange(ts_days)
    for i_, vr_ in enumerate(PT_vars):
        plt.clf()
        plt.grid()
        leg_el = [0] * len(veg_['pft_in'])
        for j_, id_ in enumerate(sp_id):
            Ipft = veg_['pft_in'].index(sp_pft[j_])
            col_ = veg_['pft_col'][Ipft]
            y_ = df_PT_.loc[df_PT_['SpID'] == id_, 'mean_%s' % vr_].values
            y_err = df_PT_.loc[df_PT_['SpID'] == id_, 'std_%s' % vr_].values
            plt.fill_between(x_, y_ - y_err, y_ + y_err,
                             facecolor=col_, alpha=.5)
            leg_el[Ipft] = plt.plot(x_, y_, color=col_)
        plt.ylim([PT_LB[i_], PT_UB[i_]])
        plt.xlabel('$t$ [days]')
        plt.ylabel(correct_var_names([vr_])[0])
        plt.title('Simulation %2d' % simnum_)
        l_hnd = [l_[0] for l_ in leg_el if l_ != 0]
        l_lbl = [l_ for j_, l_ in enumerate(veg_['pft_in']) if leg_el[j_] != 0]
        plt.legend(l_hnd, l_lbl, ncol=len(l_lbl), loc='upper center',
                   fontsize=8)
        plt.savefig(paths_['2_out_folder_plots'] + 'PT-sp_%s_%s.png' % (
            vr_, zone_snum(simnum_, inputs_['clim_zone'])), dpi=250)
        plt.close()


def inspect_PT_timeseries_perSp(
        inputs_, paths_, simnum_, S_max, ts_days, PT_vars, indx_mdy,
        PT_map_min, PT_map_max, PT_map_delta, sp_map, sp_id, sp_pft, veg_,
        GSI_all, GSI_wav, X_, PT_mean, PT_min, PT_max, meteo_mdy, M_rss, Soil_UB,
        I_pt, I_sf, I_smc, I_fc, I_rss, I_rssl, I_lai, I_ang, I_met, PT_LB,
        PT_UB, all_vars, lai020_pt=False, lai020_ang=False, lai020_met=False):

    if inputs_['inspect']:
        df_PT_ = pd.DataFrame(
            data=((np.zeros((S_max * ts_days, 2 * len(PT_vars) + 1))) *
                  np.nan), columns=(
                      ['SpID'] + ['mean_%s' % i_ for i_ in PT_vars] +
                      ['std_%s' % i_ for i_ in PT_vars]))
        kk_ = 0
        # t_ = 0
        for t_ in range(ts_days):
            # # Set scene values -----------------------------------------
            # Update plant traits daily only
            X_, PT_mean, PT_min, PT_max = update_scene_timestamp(
                t_, indx_mdy, PT_map_min, PT_map_max, PT_map_delta, sp_map,
                sp_id, GSI_all, GSI_wav, X_, PT_mean, PT_min, PT_max, meteo_mdy,
                M_rss, Soil_UB, I_pt, I_sf, I_smc, I_fc, I_rss, I_rssl, I_lai,
                I_ang, I_met, all_vars, lai020_pt=lai020_pt, 
                lai020_ang=lai020_ang, lai020_met=lai020_met)

            # Checking variables generated
            for id_ in sp_id:
                df_PT_.loc[kk_, :] = np.concatenate(
                    (np.array([id_]),
                        np.mean(X_[sp_map == id_, :], axis=0)[I_pt],
                        np.std(X_[sp_map == id_, :], axis=0)[I_pt]))
                kk_ += 1
        # Plot
        plot_PT_timeseries_perSp(df_PT_, PT_vars, sp_id, sp_pft, veg_, ts_days,
                                 inputs_, paths_, simnum_, PT_LB, PT_UB)


def get_ustar(X_, all_vars, meteo_, H_tot):
    # Use equations from the model SCOPE
    # Since ustar is not known from the beggining, iterate until changes in L 
    # are small.
    # Prevent hc = 0, to circumvent log(0.)
    canopy_h =  np.maximum(np.finfo(float).eps,
                           X_[:, :, all_vars.index('hc')].reshape(-1))
    # According to Wallace and Verhoef 2000
    canopy_z0 = .13 * canopy_h
    canopy_d = (2/3) * canopy_h
    z_minus_d = 15 - canopy_d
    z_minus_d[z_minus_d < 0.1] = 0.1
    kappa_ = 0.4
    
    ustar= np.nan
    delta_L = 100
    L_new = -1 * np.ones(1)
    coutner_  = 0
    while ((delta_L > 10) and (coutner_ <= 20)):
        # Monin-Obukhov length L
        L_ = copy.deepcopy(L_new)
        unst = np.where(L_ <  -4)[0]
        st = np.where(L_ >  4E3)[0]
    
        # Stability correction functions, friction velocity and Kh=Km=Kv
        pm_z = psim(z_minus_d, L_, unst, st)
        ustar = div_zeros((kappa_ * meteo_['u']), (
            (np.log(div_zeros(z_minus_d, canopy_z0)) - pm_z)))
        ustar[ustar < 0.001] = .001
        
        L_new = Monin_Obukhov_L(ustar, meteo_['Ta'], H_tot)
        delta_L = np.sum(np.abs(L_ - L_new))
        coutner_ += 1

    return(ustar)


def Monin_Obukhov_L(ustar_, Ta, Htot):
    kappa_ = 0.4
    rhoa_ = 1.2047
    cp_ =  1004.
    g_ = 9.81 
    # Monin-Obukhov length L
    L = -rhoa_ * cp_* ustar_ ** 3 * (Ta + 273.15) / (kappa_ * g_ * Htot)
    L[L<-1E3] = -1E3 
    L[L>1E2] = 1E2      
    L[np.isnan(L)] = -1
    
    return(L)


def psim(z, L, unst, st):
    pm = np.zeros(L.shape)
    if unst.shape[0] > 0:
        x = (1-16 * z[unst] / L[unst])**(1/4)
        pm[unst] = (2 * np.log((1 + x) / 2) + np.log(( 1 + x**2) / 2) -
                    2 * np.array([math.atan(i_) for i_ in x]) + math.pi / 2)
    if st.shape[0] > 0:
        pm[st]  = -5 * z[st] / L[st]

    return(pm)


def get_24hGPP(X_, t_, meteo_, all_vars, I_met, M_eF):
    # Output the average daily rate of GPP in umolC m-2 s-1. Units are taken
    # care in the get_Reco() function. 
    X_0 = copy.deepcopy(X_)
    met_lab = [all_vars[j_] for j_ in I_met]
    
    time_steps = range(max(0, t_-12), min(meteo_.shape[0], t_+12))
    
    GPP = 0.
    i_ = 0
    for i_ in time_steps:
        X_0[:, :, I_met] = meteo_.iloc[i_][met_lab]
        EF_ = MPL_pred(M_eF, X_[:, :, M_eF['I_']].reshape(-1, M_eF['nI']))
        GPP += np.maximum(EF_[:, 0], 0.)
    
    # Average since the data is the accumulation
    GPP = GPP / len(time_steps)

    return(GPP)


def update_ecosys_funct(t_, X_, all_vars, M_eF, reco_P, meteo_, P30dav, I_met,
                        output_map=False):
    # Functions predicted by the emulator
    EF_ = MPL_pred(M_eF, X_[:, :, M_eF['I_']].reshape(-1, M_eF['nI']))
    # print(M_eF['input_MPL']['pred_vars'])

    # Calculate ecosystem respiration
    # Since the model is for daily Reco, using daily mean air temperature and 
    # Montly precipitation, I use a running 24h average temperature and the
    # monthly precipitation average. Units are converted to mumolC m-2 s-1
    Tair_reco = np.nanmean(
        meteo_.iloc[max(0, t_-12):min(meteo_.shape[0], t_+12)]['Ta'])
    
    # Get daily GPP  (in [mumolC m-2 s-1])
    GPP_daily = get_24hGPP(X_, t_, meteo_, all_vars, I_met, M_eF)

    # Reco at environment temperature (in [mumolC m-2 s-1])
    Reco = get_Reco(reco_P[:, 0], reco_P[:, 1], reco_P[:, 2], reco_P[:, 3],
                    reco_P[:, 4], reco_P[:, 5], reco_P[:, 6], GPP_daily,
                    Tair_reco, P30dav)
    
    # At 15 C as a reference  (in [mumolC m-2 s-1])
    Reco_15C = get_Reco(reco_P[:, 0], reco_P[:, 1], reco_P[:, 2], reco_P[:, 3],
                reco_P[:, 4], reco_P[:, 5], reco_P[:, 6], GPP_daily, 15.,
                P30dav)
    
    # Compute ustar, it is used to compute ecosystem fuctional properties
    ustar = get_ustar(X_, all_vars, meteo_.iloc[t_, :], EF_[:, 3])
    
    # Generate / allocate outputs
    # Output the time at the centre of the flux integration period 
    time_out = t_ + .5 * (meteo_.iloc[t_]['hour'] / 24.)
    if output_map:
        sz_ = (X_.shape[0], X_.shape[1])
        GPP = (EF_[:, 0]).reshape(sz_)
        Rb = (Reco).reshape(sz_)
        Rb_15C = (Reco_15C).reshape(sz_)
        NEP = (EF_[:, 0] - Reco).reshape(sz_)
        LUE = (EF_[:, 6]).reshape(sz_)
        LUEgreen = (EF_[:, 7]).reshape(sz_)
        lE = (EF_[:, 1]).reshape(sz_)
        T = (EF_[:, 2]).reshape(sz_)
        H = (EF_[:, 3]).reshape(sz_)
        Rn = (EF_[:, 4]).reshape(sz_)
        G = (EF_[:, 5]).reshape(sz_)
        ustar = (ustar).reshape(sz_)
    else:    
        GPP = np.mean(EF_[:, 0])
        Rb = np.mean(Reco)
        Rb_15C = np.mean(Reco_15C)
        NEP = np.mean(EF_[:, 0] - Reco)
        LUE = np.mean(EF_[:, 6])
        LUEgreen = np.mean(EF_[:, 7])
        lE = np.mean(EF_[:, 1])
        T = np.mean(EF_[:, 2])
        H = np.mean(EF_[:, 3])
        Rn = np.mean(EF_[:, 4])
        G = np.mean(EF_[:, 5])
        ustar = np.mean(ustar)
    
    return(time_out, GPP, Rb, Rb_15C, NEP, LUE, LUEgreen, lE, T, H, Rn, G,
           ustar)


def plot_PT_timeseries(PT_vars, PT_mean, PT_min, PT_max, ts_length, S_max):
    xtime = np.arange(ts_length)
    for i_, v_ in enumerate(PT_vars):
        x_ = PT_mean[:, :, i_].T
        xmin_ = PT_min[:, :, i_].T
        xmax_ = PT_max[:, :, i_].T

        plt.clf()
        for j_ in range(S_max):
            p_ = plt.fill_between(xtime, xmin_[j_], xmax_[j_], alpha=.2)
            plt.plot(x_[j_], c=p_.get_facecolors()[0][:3])
        plt.xlabel('Time')
        plt.ylabel(v_)


def plot_R_timeseries(RF_, wl, sp_id, sp_map_2D):
    plt.clf()
    for i_, ind_ in enumerate(sp_id):
        I_ = sp_map_2D == ind_
        Rmean = np.mean(RF_[:, I_], axis=1)
        Rmin = np.min(RF_[:, I_], axis=1)
        Rmax = np.min(RF_[:, I_], axis=1)

        p_ = plt.fill_between(wl, Rmin, Rmax, alpha=.3)
        plt.plot(wl, Rmean, c=p_.get_facecolors()[0][:3])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('$RF$ [-]')


def soil_spat_var(X_, var_, all_vars, soilB_map, v_LB, v_UB, low_, high_):
    sB =  X_[0, 0, all_vars.index(var_)]
    sb_sc = np.random.uniform(low_, high_)
    sign_ = (float(np.random.uniform() >= .5) or -1.) 
    X_[:, :, all_vars.index(var_)] = np.maximum(
        v_LB, np.minimum(v_UB, ((sB) + sign_ * (2 * sb_sc * (soilB_map - .5)))))

    return(X_)


# %% 8) Main Functions
def scene_generator(simnum_, seednum_, inputs_, paths_, scsz_, X_, all_vars,
                    PT_vars, PT_LB, PT_UB, Soil_vars, Soil_LB, Soil_UB, I_cab,
                    I_cs, I_lai, I_hc, I_vcmo, I_m, I_pt, I_gmm, I_rnd, I_sf,
                    I_soil, I_smc, I_fc, I_met, I_rss, I_rssl, I_ang, veg_,
                    P_pft, GM_T, reco_, M_rss, meteo_, meteo_av, meteo_av30,
                    meteo_mdy, ts_days, indx_mdy, local_pft_lim=None):
    # -------------------------------------------------------------------------
    # Seed RNG to generate sepecies maps and soil parameters
    np.random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 2) * 10000)
    random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 2) * 10000)

    # Generate soil properties map and update meteo relative water content.
    # This is done before the species maps, so that updated water availability
    # can be used to filter out PFTs
    X_[:, :, I_soil] = generate_soil_parameters_map(
        inputs_, Soil_vars, Soil_LB, Soil_UB, meteo_['SMC'].max())    
    (meteo_, meteo_av, meteo_av30) = update_wr(meteo_, meteo_av, meteo_av30,
                                               np.unique(X_[:, :, I_fc]))

    # Generate species map
    (sp_map, pft_map, S_max, sp_ab, sp_id, sp_pft, soilB_map) = generate_map(
        simnum_, inputs_, paths_, P_pft, veg_, meteo_av30) 
    
    # Update soil bright parameter to make spatially vary with some correlation
    # with the species. Respect the bounds of the parameter by shifting the
    # average value if necessary  plt.imshow(soilB_map)
    X_ = soil_spat_var(X_, 'BSMBrightness', all_vars, soilB_map,
                       Soil_LB[0], Soil_UB[0], .01, .1)
    X_ = soil_spat_var(X_, 'BSMlat', all_vars, soilB_map,
                       Soil_LB[1], Soil_UB[1], 1, 10)
    X_ = soil_spat_var(X_, 'BSMlon', all_vars, soilB_map,
                       Soil_LB[2], Soil_UB[2], 1, 10)
    # Avoid modifying SMC, since it'd require having a different GSI per pixel
    X_ = soil_spat_var(X_, 'rss_level', all_vars, soilB_map,
                       .1, 1., .01, .1)
    
    # -------------------------------------------------------------------------
    # Reseed RNG to ensure same properties for maps with different features
    rng_intra_sp_mean = np.random.default_rng(seed=(seednum_ + 2) * 500)
    rng_intra_sp_var = np.random.default_rng(seed=(seednum_ + 2) * 100)
    np.random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 3) * 10000)
    random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 3) * 10000)
    # Seed also the GMM providing foliar traits
    GM_T['GMMtraits'].random_state = ((inputs_['rseed_num'] + 4) +
                                      (seednum_ + 10) * 10)

    # Generate min and max plant trait maps
    (PT_map_min, PT_map_max, PT_map_delta, num_dis, local_av, local_LB,
     local_UB) = (
        sp.generate_plant_trait_limits(
            sp_map, sp_id, sp_ab, S_max, sp_pft, veg_, GM_T, PT_vars, PT_LB,
            PT_UB, I_cab, I_cs, I_lai, I_hc, I_vcmo, I_m, I_gmm, I_rnd,
            rng_intra_sp_mean, rng_intra_sp_var, local_pft_lim=local_pft_lim))

    # -------------------------------------------------------------------------
    # Reseed RNG to ensure same properties for maps with different features
    np.random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 5) * 10000)
    random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 5) * 10000)

    # Generate GSI species coefficients
    (GSI_all, GSI_wav, GSI_wav_param, GSI_rin, GSI_rin_param, GSI_tcol,
        GSI_tcol_param, GSI_twrm, GSI_twrm_param) = gsi.generate_GSI(
        meteo_av30, paths_, simnum_, veg_, sp_pft, PT_vars,
        inputs_['clim_zone'], doplot=inputs_['inspect'])

    # -------------------------------------------------------------------------
    # Reseed RNG to ensure same properties for maps with different features
    np.random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 6) * 10000)
    random.seed((inputs_['rseed_num'] + 1) * (seednum_ + 6) * 10000)

    # Generate Ecosystem Respiration parameters for the model of 
    # Migliavacca et al. 2011. Update LAImax for the respiration parameters,
    # accounting for the maximum phenological state achieved by each pixel.
    LAI_max_map = copy.deepcopy(PT_map_max[:, :, I_lai])
    GSI_all_max_sp = np.max(GSI_all[:, :, I_lai], axis=0)
    for i_, id_ in enumerate(sp_id):
        I_ = (sp_map == id_)
        LAI_max_map[I_] = LAI_max_map[I_] * GSI_all_max_sp[i_]
    # Produce the Reco model parameters
    reco_P = sp.generate_Reco_params(sp_map, sp_id, S_max, sp_pft, reco_,
                                     LAI_max_map)

    # -------------------------------------------------------------------------
    # Clouds generator
    if ('Clouds' in inputs_.keys()) and (inputs_['Clouds'] is True):
        coulds_map, cloud_cover = generate_clouds(scsz_)
        plt.imshow(coulds_map)
    else:
        coulds_map = None

    # -------------------------------------------------------------------------
    # ### Inspect the dynamics of plant traits simulated ###################
    PT_mean = np.zeros((ts_days, S_max, len(PT_vars)))
    PT_min = np.zeros((ts_days, S_max, len(PT_vars)))
    PT_max = np.zeros((ts_days, S_max, len(PT_vars)))
    if inputs_['inspect']:
        inspect_PT_timeseries_perSp(
            inputs_, paths_, simnum_, S_max, ts_days, PT_vars, indx_mdy,
            PT_map_min, PT_map_max, PT_map_delta, sp_map, sp_id, sp_pft,
            veg_, GSI_all, GSI_wav, X_, PT_mean, PT_min, PT_max, meteo_mdy,
            M_rss, Soil_UB, I_pt, I_sf, I_smc, I_fc, I_rss, I_rssl, I_lai,
            I_ang, I_met, PT_LB, PT_UB, all_vars, lai020_pt=True)

        do_plot_meteo_ts(meteo_, fname_=(
            paths_['2_out_folder_plots'] + 'Meteo_%s.png' % (
                zone_snum(simnum_,inputs_['clim_zone']))))

    return(X_, meteo_, meteo_av, meteo_av30, sp_map, pft_map, S_max, sp_ab,
           sp_id,  sp_pft, PT_map_min, PT_map_max, PT_map_delta, num_dis,
           reco_P, GSI_all, GSI_wav, GSI_wav_param, GSI_rin, GSI_rin_param,
           GSI_tcol, GSI_tcol_param, GSI_twrm, GSI_twrm_param, PT_mean, PT_min,
           PT_max, local_av, local_LB, local_UB, coulds_map)


def create_scene_data(paths_, inputs_, simnum_, seednum_, scsz_, X0_, all_vars,
                      PT_vars, PT_LB, PT_UB, Soil_vars, Soil_LB, Soil_UB, I_cab,
                      I_cs, I_lai, I_hc, I_vcmo, I_m, I_pt, I_gmm, I_rnd, I_sf,
                      I_soil, I_smc, I_fc, I_met, I_rss, I_rssl, I_ang, veg_,
                      P_pft, GM_T, reco_, M_rss, minLAImax=1.,
                      local_pft_lim=None):
    
    # Define the variable used to seed the random number generators
    if seednum_ == None:
        seednum_ = copy.deepcopy(simnum_)
    elif isinstance(seednum_, int) == False:
        seednum_ = int(seednum_)
    
    # Create output folder if plots are to be produced
    if (inputs_['inspect'] and
        (os.path.isdir(paths_['2_out_folder_plots']) == False)):
        os.makedirs(paths_['2_out_folder_plots'])
    
    # Get meteorology data
    if inputs_['verbose']:
        print('\tLoading meteo data...')
        t0 = time.time()
    (meteo_, meteo_av, meteo_av30, meteo_mdy, meta_met, ts_length, ts_days,
     indx_day, indx_mdy, _) = mt.get_meteo(
         paths_, np.unique(X0_[:, :, I_fc]).squeeze(),
         KZone=inputs_['clim_zone'], site_num=simnum_, minLAImax=minLAImax,
         SMC_b=[Soil_LB[Soil_vars.index('SMC')],
                Soil_UB[Soil_vars.index('SMC')]])
    if inputs_['verbose']:
        print_et('\t\t', time.time() - t0)

    #  Generate 
    if inputs_['verbose']:
        print('\tGenerating the Scene...')
        t0 = time.time()
    (X_, meteo_, meteo_av, meteo_av30, sp_map, sp_pft, S_max, sp_ab, sp_id,
     sp_pft, PT_map_min, PT_map_max, PT_map_delta, num_dis, reco_P, GSI_all,
    GSI_wav, GSI_wav_param, GSI_rin, GSI_rin_param, GSI_tcol, GSI_tcol_param,
    GSI_twrm, GSI_twrm_param, PT_mean, PT_min, PT_max, local_av, local_LB,
    local_UB, coulds_map
    ) = scene_generator(
        simnum_, seednum_, inputs_, paths_, scsz_, X0_, all_vars, PT_vars,
        PT_LB, PT_UB, Soil_vars, Soil_LB, Soil_UB, I_cab, I_cs, I_lai, I_hc,
        I_vcmo, I_m, I_pt, I_gmm, I_rnd, I_sf, I_soil, I_smc, I_fc, I_met,
        I_rss, I_rssl, I_ang, veg_, P_pft, GM_T, reco_, M_rss, meteo_, meteo_av,
        meteo_av30, meteo_mdy, ts_days, indx_mdy, local_pft_lim=local_pft_lim)
    if inputs_['verbose']:
        print_et('\t\t', time.time() - t0)
    
    return(meteo_, meteo_av, meteo_av30, meteo_mdy, meta_met, ts_length,
           ts_days, indx_day, indx_mdy, X_, sp_map, sp_pft, S_max, sp_ab, sp_id,
           sp_pft, PT_map_min, PT_map_max, PT_map_delta, num_dis, reco_P,
           GSI_all, GSI_wav, GSI_wav_param, GSI_rin, GSI_rin_param, GSI_tcol,
           GSI_tcol_param, GSI_twrm, GSI_twrm_param, PT_mean, PT_min, PT_max,
           local_av, local_LB, local_UB, coulds_map)