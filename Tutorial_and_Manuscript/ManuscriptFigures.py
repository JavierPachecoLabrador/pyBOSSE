# %% 0) Imports ################################################################
import os
import sys
import pandas as pd
import numpy as np
import copy
import re
import time
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, norm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sn

# Import BOSSE and pyGNDIV
parent_foler = os.path.abspath(
    os.path.join(os.path.abspath(
        os.path.join(os.getcwd(), os.pardir)), os.pardir))

# Defione as global the pyBOSSE paths
path_bosse = (parent_foler + '//pyBOSSE//')
path_inputs= path_bosse + '//BOSSE_inputs//'
path_models= path_bosse + '//BOSSE_models//'

sys.path.insert(0, path_bosse)
from BOSSE.bosse import BosseModel
from BOSSE.helpers import set_up_paths_and_inputs, print_dict
from BOSSE.scsim_rtm import check_spatial_upscale, get_ndvi_nirv
from BOSSE.scsim_gsi_model import GSI_loop, gsi_smooth_input, get_water_avail

from pyGNDiv import pyGNDiv_imagery as gni


# %% 1) Functions ##############################################################
#  General functions -----------------------------------------------------------
def get_char_label(n_):
    lbl_ = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    return('(' + lbl_[n_] + ')')


def isacronym(str_):
    return(all([i_.isupper() for i_ in str_]))


def correct_var_names(leg_label):
    for i_, l_ in enumerate(leg_label):
        is_acr = isacronym(l_)
        # Remove $
        l_ = l_.replace('$', '')
        # Keep uds away if present
        uds = l_.split('(')
        if len(uds) < 2:
            uds = l_.split('[')
        if len(uds) == 2:
            l0_ = uds[0]
            uds = ' [' + uds[1].replace(')', ']')
        else:
            l0_ = copy.deepcopy(l_)
            uds = ''
        # Deal with underscores and uds
        if '_' in l_:
            p_ = re.split('_', l_, 1)
            leg_label[i_] = ('$' + p_[0][:] + '_{\\rm ' +
                             p_[1][:].replace('_', ',') + '}$' + uds)
        elif is_acr:
            leg_label[i_] = l0_ + uds
        # Concentrations
        elif (l0_[0] == 'C') and (isacronym(l0_[1:]) == False):
            leg_label[i_] = '$' + l0_[0] + '_{\\rm ' + l0_[1:] + '}$' + uds
        # fractions
        elif  (l0_[0] == 'f'):
             leg_label[i_] = '$' + l0_[0] + '_{\\rm ' + l0_[1:] + '}$' + uds
        else:
            leg_label[i_] = '$' + l0_ + '$' + uds

    return(leg_label)


# Figure 3 ---------------------------------------------------------------------
def show_GSI_relationship(axi, x_, GSI_cs_smp, veg_,
             sp_pft, xlab_, ylab_, sym_='-', inv_val=False):
    
    if np.ptp(x_) > .0:
        x_smp0 = np.arange(x_.min(), x_.max(), np.ptp(x_)/100)
    else:
        x_smp0 = np.array(x_[0])
                
    shp_ = GSI_cs_smp[0].shape
    GSI_delta_th = (np.inf * np.ones(shp_)).reshape(-1)
    y = GSI_loop(x_smp0, 0,
                 GSI_cs_smp[0].reshape(-1),
                 GSI_cs_smp[1].reshape(-1),
                 GSI_cs_smp[2].reshape(-1),
                 GSI_delta_th).reshape(-1, shp_[0], shp_[1])

    if inv_val:
        y = 1. - y

    # Plot
    axi.grid()
    for i_ in range(shp_[0]):
        col_ = veg_['pft_col'][veg_['pft_in'].index(sp_pft[i_])]
        axi.fill_between(x_smp0, y[:, i_].min(axis=1), y[:, i_].max(axis=1),
                           facecolor=col_, alpha=.5)
        axi.plot(x_smp0, y[:, i_].mean(axis=1), sym_, color=col_)
    axi.set_xlabel(xlab_)
    axi.set_ylabel(ylab_)
    

def show_GSI_all(axi, doy_, GSI_all, veg_, sp_pft, xlab_, ylab_,):
    shp_ = GSI_all.shape

    # Plot
    axi.grid()
    for i_ in range(shp_[1]):
        col_ = veg_['pft_col'][veg_['pft_in'].index(sp_pft[i_])]
        # axi.fill_between(doy_, GSI_all[:, i_].min(axis=1),
        #                  GSI_all[:, i_].max(axis=1),
        #                    facecolor=col_, alpha=.5)
        axi.plot(doy_, GSI_all[:, i_].mean(axis=1), color=col_)
    axi.set_xlabel(xlab_)
    axi.set_ylabel(ylab_)

    
def plot_GSI(fname_, bosse_M, sim_sel=0):
    # Initialize scene
    bosse_M.initialize_scene(0, seednum_=sim_sel)
    
    # Compute smoothed time series of meteorological variables    
    wav_ = gsi_smooth_input(get_water_avail(bosse_M.veg_, bosse_M.sp_pft,
                                            bosse_M.meteo_av30))
    Rin_ = gsi_smooth_input(bosse_M.meteo_av30['Rin'].values)
    Ta_ = gsi_smooth_input(bosse_M.meteo_av30['Ta'].values)
    
    plt.close()
    fig, ax = plt.subplots(2, 3, figsize=(10., 5.))

    show_GSI_relationship(ax[0, 0], wav_, bosse_M.GSI_wav_param, bosse_M.veg_,
                        bosse_M.sp_pft, '$W_{\\rm r}$', '$f$($W_{\\rm p}$) [-]')
    show_GSI_relationship(ax[0, 1], Rin_, bosse_M.GSI_rin_param, bosse_M.veg_,
                        bosse_M.sp_pft, bosse_M.get_variable_label('Rin'),
                        '$f$($R_{\\rm in}$) [-]')
    show_GSI_relationship(ax[1, 0], Ta_, bosse_M.GSI_tcol_param, bosse_M.veg_,
                        bosse_M.sp_pft, bosse_M.get_variable_label('Ta'),
                        '$f$($T_{\\rm a, cold}$) [-]', sym_='-')
    show_GSI_relationship(ax[1, 1], Ta_, bosse_M.GSI_twrm_param, bosse_M.veg_,
                        bosse_M.sp_pft, bosse_M.get_variable_label('Ta'),
                        '$f$($T_{\\rm a, warm}$) [-]', sym_='-', inv_val=True)
    doy_ = np.arange(1, bosse_M.meteo_av30.shape[0] + 1)

    ax[0, 2].grid()
    p0 = ax[0, 2].plot(doy_, bosse_M.GSI_wav.mean(axis=2), c='Navy',
                    label='$f$($W_{\\rm p}$)')
    p1 = ax[0, 2].plot(doy_, bosse_M.GSI_rin.mean(axis=2), c='DarkOrange',
                    label='$f$($R_{\\rm in}$)')
    p2 = ax[0, 2].plot(doy_, bosse_M.GSI_tcol.mean(axis=2), '--', c='SteelBlue',
                    label='$f$($T_{\\rm a, cold}$)')
    p3 = ax[0, 2].plot(doy_, bosse_M.GSI_twrm.mean(axis=2), '--', c='Salmon',
                    label='$f$($T_{\\rm a, warm}$)')
    ax[0, 2].set_xlabel('DoY')
    ax[0, 2].set_ylabel('$f_{\\rm GSI}$ [-]')

    show_GSI_all(ax[1, 2], doy_, bosse_M.GSI_all, bosse_M.veg_,
                bosse_M.sp_pft, 'DoY', 'GSI [-]')
    
    ax = ax.reshape(-1)
    for i_ in range(6):
        if i_ < 3:
            f_ = .88
        else:
            f_ = .05
        xl_ = ax[i_].get_xlim()
        lx_ = xl_[0] + f_*(xl_[1] - xl_[0])
        yl_ = ax[i_].get_ylim()
        ly_ = yl_[0] + .9*(yl_[1] - yl_[0])
        place_plot_label(ax[i_], lx_, ly_, i_)
        

    pfts_ = np.unique(bosse_M.sp_pft)
    num_pft = pfts_.shape[0]
    col_vars = ['Navy', 'DarkOrange', 'SteelBlue', 'Salmon']
    sym_vars = ['-'] * 2 + ['--'] * 2

    custom_lines = (
        [Line2D([0], [0], color=bosse_M.veg_['pft_col'][
            bosse_M.veg_['pft_in'].index(pfts_[i_])])
        for i_ in range(num_pft)] +
        [Line2D([0], [0], color=col_vars[i_], linestyle=sym_vars[i_])
        for i_ in range(num_pft)])
    labels = ([i_ for i_ in pfts_] + ['$f$($W_{\\rm p}$)', '$f$($R_{\\rm in}$)',
                                    '$f$($T_{\\rm a, cold}$)',
                                    '$f$($T_{\\rm a, warm}$)', 'GSI'])
    fig.legend(custom_lines, labels, loc='upper center', ncol=8,
               borderaxespad=0.1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.savefig(fname_, dpi=300)


# Figure 4 ---------------------------------------------------------------------
def plt_sp_patterns(paths_, inputs_, dest_, sim_sel=0, fontsize=12):
        
    fig, ax = plt.subplots(1, 3, figsize=[12, 3.5], sharex=True, sharey=True)
    plt.tight_layout()

    for i_, spt_ in enumerate(bosse_spatial_patterns):
        (inputs_, paths_) = set_up_paths_and_inputs(None, output_folder,
                                                    create_out_folder=False,
                                                    pth_root=path_bosse,
                                                    pth_inputs=path_inputs,
                                                    pth_models=path_models)
        inputs_['sp_pattern'] = spt_
        bosse_M = BosseModel(inputs_, paths_)
        bosse_M.initialize_scene(sim_sel)

        
        ax[i_].imshow(bosse_M.sp_map, cmap='tab20')
        set_map_lbls(ax[i_], 'Species map ($S$ = %d)' %  bosse_M.S_max,
                    fontsize=fontsize)    
        divider = make_axes_locatable(ax[i_])
        
        xl_ = ax[i_].get_xlim()
        delta_ = (xl_[1] - xl_[0]) 
        lx_ = xl_[0] + delta_ * (26 / 30)
        ly_ = xl_[0] + delta_ * (2.75 / 30)
        place_plot_label(ax[i_], lx_, ly_, i_)
    
    plt.tight_layout()
    fig.savefig(dest_, dpi=300)
    plt.close()


# Figure 5 ---------------------------------------------------------------------
def plt_trait_time_series(bosse_M, axi, kl_, title_in, vr_='LAI', uds_=''):
    df_PT_ = pd.DataFrame(
    data=((np.zeros((bosse_M.S_max * bosse_M.ts_days, 2 *
                        len(bosse_M.PT_vars) + 1))) * np.nan),
    columns=( ['SpID'] + ['mean_%s' % i_ for i_ in bosse_M.PT_vars] +
                ['std_%s' % i_ for i_ in bosse_M.PT_vars]))
    
    kk_ = 0
    # t_ = 0
    for t_ in range(bosse_M.ts_days):
        # # Set scene values -----------------------------------------
        X_ = bosse_M.pred_scene_timestamp(t_, 'meteo_mdy', 'indx_mdy',
                                          lai020_pt=True, sp_res=100)
        # Checking variables generated
        for id_ in bosse_M.sp_id:
            df_PT_.loc[kk_, :] = np.concatenate(
                (np.array([id_]),
                    np.mean(X_[bosse_M.sp_map == id_, :], axis=0)[bosse_M.I_pt],
                    np.std(X_[bosse_M.sp_map == id_, :], axis=0)[bosse_M.I_pt]))
            kk_ += 1
    
    x_ = np.arange(bosse_M.ts_days)
    ii_ = bosse_M.PT_vars.index(vr_)
    axi.grid()
    leg_el = [0] * len(bosse_M.veg_['pft_in'])
    for j_, id_ in enumerate(bosse_M.sp_id):
        Ipft = bosse_M.veg_['pft_in'].index(bosse_M.sp_pft[j_])
        col_ = bosse_M.veg_['pft_col'][Ipft]
        sym_= bosse_M.veg_['pft_sym'][Ipft]
        y_ = df_PT_.loc[df_PT_['SpID'] == id_, 'mean_%s' % vr_].values
        y_err = df_PT_.loc[df_PT_['SpID'] == id_, 'std_%s' % vr_].values
        axi.fill_between(x_, y_ - y_err, y_ + y_err,
                            facecolor=col_, alpha=.5)
        leg_el[Ipft] = axi.plot(x_, y_, ls=sym_, color=col_)
    axi.set_ylim([bosse_M.PT_LB[ii_], 1.2*bosse_M.PT_UB[ii_]])
    axi.set_xlabel('$t$ [days]')
    axi.set_ylabel(correct_var_names([vr_])[0] + ' ' + uds_)
    axi.set_title(title_in, fontsize=10)
    axi.text(650, axi.get_ylim()[1] * .9, get_char_label(kl_),
             fontweight='bold')
    l_hnd = [l_[0] for l_ in leg_el if l_ != 0]
    l_lbl = [l_ for j_, l_ in enumerate(
        bosse_M.veg_['pft_in']) if leg_el[j_] != 0]
    axi.legend(l_hnd, l_lbl, ncol=int(np.ceil(len(l_lbl) / 2)),
               loc='upper left', fontsize=8)


def subplot_var_time_series(ax, kl_, vr_, uds_, inputs_, paths_, sim_sel=0):
    # Initialize BOSSE class
    bosse_M = BosseModel(inputs_, paths_)
    
    # Get the BOSSE spatial patterns and climate zones
    bosse_spatial_patterns = bosse_M.get_input_descriptors('sp_pattern')
    bosse_climzone = bosse_M.get_input_descriptors('clim_zone')
    
    inputs_['clim_zone'] = bosse_climzone[0]
    i_, sp = 0, bosse_spatial_patterns[0]
    for i_, sp in enumerate(bosse_spatial_patterns):
        inputs_['sp_pattern'] = sp
        
        # Update BOSSE Model
        bosse_M = BosseModel(inputs_, paths_)
        bosse_M.initialize_scene(sim_sel)      
        plt_trait_time_series(
            bosse_M, ax[0, i_], kl_, '%s, %s ($S$ = %d)' %
            (inputs_['clim_zone'], sp, len(bosse_M.sp_pft)), vr_=vr_,
            uds_=uds_)
        kl_ += 1

    inputs_['sp_pattern'] = sp    
    for i_, kz in enumerate(bosse_climzone[1:]):
        inputs_['clim_zone'] = kz
        
        bosse_M = BosseModel(inputs_, paths_)
        bosse_M.initialize_scene(sim_sel)      
        plt_trait_time_series(
            bosse_M, ax[1, i_], kl_, '%s, %s ($S$ = %d)' %
            (kz, inputs_['sp_pattern'], len(bosse_M.sp_pft)),vr_=vr_,
            uds_=uds_)
        kl_ += 1


def plot_var_time_series(vr_, uds_, inputs_, paths_, dest_, sim_sel=0):
    
    fig, ax = plt.subplots(4, 3, figsize=[8.5, 9.5], sharex=False, sharey=False)
    for i_, vr_i in enumerate(vr_):
        subplot_var_time_series(ax[i_ * 2 :i_ * 2 + 2, :], i_ * 6, vr_i,
                                uds_[i_], inputs_, paths_, sim_sel=sim_sel)

    plt.tight_layout()
    fig.savefig(dest_, dpi=300)
    plt.close()


# Figure 6 ---------------------------------------------------------------------
def get_im_cb_limits(im_, th_=0.05, prec=2, lims_=None):
    im_ = im_.reshape(-1)
    if lims_==None:
        min_ = np.nanmin(im_)
        max_ = np.nanmax(im_)
    else:
        min_ = copy.deepcopy(lims_[0])
        max_ = copy.deepcopy(lims_[1])
        
    range_ = max_ - min_
    
    f_ = 10 ** prec
    min_out = np.floor((min_ - (th_ * range_)) * f_) / f_
    max_out = np.ceil((max_ + (th_ * range_)) * f_) / f_
    
    return(min_out, max_out)


def place_plot_label(ax, lx_, ly_, i_):
    t = ax.text(lx_, ly_, get_char_label(i_), fontsize=11, fontweight='bold')
    t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8))


def rgb_indices(RF, wvl):
    rgb = np.zeros((RF.shape[0], RF.shape[1], 3), dtype=int)
    srf_ = norm.pdf(np.linspace(-2.96, 2.96, 100))
    srf_ = np.repeat(
        np.repeat((srf_ / np.sum(srf_)).reshape(1, 1, -1), RF.shape[0], axis=0),
        RF.shape[0], axis=1)
    for i_, wl_i in enumerate([400, 500, 600]):
        I_ = (wvl >= wl_i) * (wvl < wl_i + 100)
        rgb[:, :, i_] = np.sum((RF[:, :, I_] * srf_) * 255, axis=2).astype(int) 

    ndvi_, nirv_ = get_ndvi_nirv(RF, wvl) 
    
    return(rgb, ndvi_, nirv_)


def stand_predobs(pred, obs):
    obs = obs.reshape(-1, 1)
    out_ = pd.DataFrame(data=np.zeros((obs.size, 2)),
                                      columns=['pred', 'obs'])
    
    sc = StandardScaler().fit(obs)
    out_['obs'] = (sc.transform(obs)).reshape(-1)
    out_['pred'] = (sc.transform(pred.reshape(-1, 1))).reshape(-1)
    
    return(out_)


def set_map_lbls(axi, title, fontsize=12):
    axi.set_xticks([])
    axi.set_xticklabels([])
    axi.set_yticks([])
    axi.set_yticklabels([])
    axi.set_xlabel('$x$ [pixels]', fontsize=fontsize)
    axi.set_ylabel('$y$ [pixels]', fontsize=fontsize)
    if title != '':
        axi.set_title(title, fontsize=fontsize)


def set_sizes(bosse_M, sp_res, scsz_):
    # Determine whether there will be spatial resampling and the new output size
    RaoQ_sz0 = getattr(bosse_M, 'RaoQ_sz')
    RaoQ_w = getattr(bosse_M, 'RaoQ_w')
    if sp_res == 100:
        RaoQ_sz = copy.deepcopy(RaoQ_sz0)
        out_sz = copy.deepcopy(scsz_)
    else:
        out_sz, valid = check_spatial_upscale(scsz_, sp_res)
        RaoQ_sz = int(out_sz / RaoQ_w) ** 2
    
    return(RaoQ_w, RaoQ_sz, out_sz)


def plot_maps(paths_, inputs_, dest_, t_=230, sim_sel=0, sp_res=100,
              fontsize=12, bosse_M=None, out_Bosse_M=False, lims_=None):
    if lims_ == None:
        lims_ = [None] * 12
        
    sppaterns_ = ['clustered', 'intermediate', 'even'] 
    climzone_ = ['Continental', 'Tropical', 'Dry', 'Temperate']
    
    inputs_['clim_zone'] = climzone_[0]
    if inputs_['sp_pattern'] == '':
        inputs_['sp_pattern'] = sppaterns_[1]
    if bosse_M == None:
        bosse_M = BosseModel(inputs_, paths_)
        bosse_M.initialize_scene(sim_sel)
    
    (RaoQ_w, RaoQ_sz, out_sz) = set_sizes(bosse_M, sp_res, bosse_M.scsz_)
    ones_layer = np.ones((out_sz * out_sz, 1))
    I_lai = bosse_M.PT_vars.index('LAI')
    I_cab = bosse_M.PT_vars.index('Cab')
    I687 = bosse_M.M_F['wl'] == 687
    I760 = bosse_M.M_F['wl'] == 760
    
    X_ = bosse_M.pred_scene_timestamp(t_, 'meteo_mdy', 'indx_mdy',
                                      lai020_pt=True, sp_res=100)    
    X_r = bosse_M.pred_scene_timestamp(t_, 'meteo_mdy', 'indx_mdy',
                                       lai020_pt=True, sp_res=sp_res)
    
    RF_ = bosse_M.pred_refl_factors(X_, bosse_M.scsz_, sp_res=sp_res)
    (rgb, ndvi_, nirv_) = rgb_indices(RF_, bosse_M.M_R['wl'])
    
    # # Retrieval of optical traits
    OT_ = bosse_M.pred_opt_traits(np.concatenate((
        RF_.reshape(out_sz * out_sz, -1), ones_layer *
        bosse_M.meteo_mdy.loc[t_, 'tts']), axis=1), out_sz)
    
    F_ = bosse_M.pred_fluorescence_rad(X_, bosse_M.scsz_, out_sz, sp_res=sp_res)

    LST_ = bosse_M.pred_landsurf_temp(X_, bosse_M.scsz_, sp_res=sp_res)


    plt.close()
    fig, ax = plt.subplots(3, 4, figsize=[12, 7.75], sharex=False, sharey=False)
    plt.tight_layout()
    
    axi_ = 0
    ax[0, 0].imshow(bosse_M.sp_map, cmap='tab20')
    set_map_lbls(ax[0, 0], 'Species map ($S$ = %d)' %  bosse_M.S_max,
                 fontsize=fontsize)    
    divider = make_axes_locatable(ax[0, 0])
    
    xl_ = ax[0, 0].get_xlim()
    delta_ = (xl_[1] - xl_[0]) 
    lx_ = xl_[0] + delta_ * (26 / 30)
    ly_ = xl_[0] + delta_ * (2.75 / 30)
    place_plot_label(ax[0, 0], lx_, ly_, axi_)
    t = ax[0, 0].text(xl_[0] - delta_ * .4, ly_ - delta_ * .25,
                      f'DOY {t_}', fontsize=10, fontweight='bold', c='DarkRed')
    # t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    axi_ += 1
    if sp_res == 100:
        pft_col = np.zeros((RF_.shape[0], RF_.shape[1], 3), dtype=int)
        # for pft_i in bosse_M.sp_pft
        for i_, spi_ in enumerate(bosse_M.sp_id):
            I_ = np.where(bosse_M.sp_map == spi_)
            Ipft = bosse_M.veg_['pft_in'].index(bosse_M.sp_pft[i_])
            pft_col[I_[0], I_[1], :] = (
                np.array(mcolors.to_rgb(bosse_M.veg_['pft_col'][Ipft])[:3]) * 255
                ).astype(int).reshape(1, 1, -1)
        im_ = ax[0, 1].imshow(pft_col)
        set_map_lbls(ax[0, 1], 'PFTs', fontsize=fontsize)
    else:
        im_ = ax[0, 1].imshow(rgb * 1)
        set_map_lbls(ax[0, 1], 'RGB', fontsize=fontsize)
    divider = make_axes_locatable(ax[0, 1])
    place_plot_label(ax[0, 1], lx_, ly_, axi_)

    axi_ += 1
    vmin_lai, vmax_lai = get_im_cb_limits(
        np.concatenate((X_[:, :, I_lai].reshape(-1),
                       OT_[:, :, I_lai].reshape(-1))), prec=0,
        lims_=lims_[axi_])
    im_ = ax[0, 2].imshow(X_[:, :, I_lai],cmap='YlGn',
                          vmin=vmin_lai, vmax=vmax_lai)
    set_map_lbls(ax[0, 2], 'LAI $\\rm [m^2 m^{-2}]$', fontsize=fontsize)
    divider = make_axes_locatable(ax[0, 2])
    place_plot_label(ax[0, 2], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    vmin_cab, vmax_cab = get_im_cb_limits(
        np.concatenate((X_[:, :, I_cab].reshape(-1),
                        OT_[:, :, I_cab].reshape(-1))), prec=0,
        lims_=lims_[axi_])
    im_ = ax[0, 3].imshow(X_[:, :, I_cab], cmap='YlGn',
                          vmin=vmin_cab, vmax=vmax_cab)
    set_map_lbls(ax[0, 3], '$C\\rm_{ab}$ $[\\mu g cm^{-2}]$', fontsize=fontsize)
    divider = make_axes_locatable(ax[0, 3])
    place_plot_label(ax[0, 3], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    vmin_, vmax_ = get_im_cb_limits(ndvi_, th_=.1, prec=1, lims_=lims_[axi_])
    im_ = ax[1, 0].imshow(ndvi_, cmap='YlGn', vmin=vmin_, vmax=vmax_)
    set_map_lbls(ax[1, 0], 'NDVI [-]', fontsize=fontsize)
    divider = make_axes_locatable(ax[1, 0])
    place_plot_label(ax[1, 0], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    vmin_, vmax_ = get_im_cb_limits(nirv_, th_=.1, prec=1, lims_=lims_[axi_])
    im_ = ax[1, 1].imshow(nirv_, cmap='YlGn', vmin=vmin_, vmax=vmax_)
    set_map_lbls(ax[1, 1], 'NIRV$_{\\rm v}$ [-]', fontsize=fontsize)
    divider = make_axes_locatable(ax[1, 1])
    place_plot_label(ax[1, 1], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    im_ = ax[1, 2].imshow(OT_[:, :, I_lai], cmap='YlGn',
                          vmin=vmin_lai, vmax=vmax_lai)
    set_map_lbls(ax[1, 2], 'OT$_{\\it LAI}$ $\\rm [m^2 m^{-2}]$',
                 fontsize=fontsize)
    divider = make_axes_locatable(ax[1, 2])
    place_plot_label(ax[1, 2], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    im_ = ax[1, 3].imshow(OT_[:, :, I_cab], cmap='YlGn',
                          vmin=vmin_cab, vmax=vmax_cab)
    set_map_lbls(ax[1, 3],
                 'OT$_{C_{\\rm ab}}$ $\\rm [\\mu g cm^{-2}]$',
                 fontsize=fontsize)
    divider = make_axes_locatable(ax[1, 3])
    place_plot_label(ax[1, 3], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    vmin_, vmax_ = get_im_cb_limits(F_[:, :, I687], th_=.1, prec=1,
                                    lims_=lims_[axi_])
    im_ = ax[2, 0].imshow(F_[:, :, I687], cmap='PuRd', vmin=vmin_, vmax=vmax_)
    set_map_lbls(ax[2, 0],
                 '$F_{\\rm 687 nm}$ $\\rm [mW m^{-2} sr^{-1} nm^{-1}]$',
                 fontsize=fontsize)
    divider = make_axes_locatable(ax[2, 0])
    place_plot_label(ax[2, 0], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)
    
    axi_ += 1
    vmin_, vmax_ = get_im_cb_limits(F_[:, :, I760], th_=.1, prec=1,
                                    lims_=lims_[axi_])
    im_ = ax[2, 1].imshow(F_[:, :, I760], cmap='PuRd', vmin=vmin_, vmax=vmax_)
    set_map_lbls(ax[2, 1],
                 '$F_{\\rm 760 nm}$ $\\rm [mW m^{-2} sr^{-1} nm^{-1}]$',
                 fontsize=fontsize)
    divider = make_axes_locatable(ax[2, 1])
    place_plot_label(ax[2, 1], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)
    
    axi_ += 1
    vmin_, vmax_ = get_im_cb_limits(LST_, th_=.05, prec=0,
                                    lims_=lims_[axi_])
    im_ = ax[2, 2].imshow(LST_, cmap='RdYlBu_r', vmin=vmin_, vmax=vmax_)
    set_map_lbls(ax[2, 2], 'LST $\\rm [K]$', fontsize=fontsize)
    divider = make_axes_locatable(ax[2, 2])
    place_plot_label(ax[2, 2], lx_, ly_, axi_)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    fig.colorbar(im_, cax=cax)

    axi_ += 1
    lai = stand_predobs(OT_[:, :, I_lai], X_r[:, :, I_lai])
    r2_lai = pearsonr(lai['pred'], lai['obs'])
    cab = stand_predobs(OT_[:, :, I_cab], X_r[:, :, I_cab])
    r2_cab = pearsonr(cab['pred'], cab['obs'])

    sn.regplot(cab, x='pred', y='obs', ax=ax[2, 3],
               label='$C_{\\rm ab}$; $r^2$=%.2f' % r2_cab[0] ** 2,
               color='OliveDrab', scatter_kws={'s':10},
               line_kws={'color':'DarkGreen'})
    sn.regplot(lai, x='pred', y='obs', ax=ax[2, 3],
               label='LAI; $r^2$=%.2f' % r2_lai[0] ** 2,
               color='SteelBlue', scatter_kws={'s':10},
               line_kws={'color':'Navy'})
    ax[2, 3].set_xlim([-2.75, 2.75])
    ax[2, 3].set_ylim([-2.75, 2.75])
    ax[2, 3].grid()
    ax[2, 3].plot([-2.5, 2.5], [-2.5, 2.5], '--k')
    ax[2, 3].legend(fontsize=9, loc='upper left', framealpha=.5, frameon=True)
    ax[2, 3].set_xlabel('Predicted (standardized)', fontsize=fontsize)
    ax[2, 3].set_ylabel('Simulated (standardized)', fontsize=fontsize)
    ax[2, 3].set_title('Retrieval evaluation', fontsize=fontsize)    
    place_plot_label(ax[2, 3], 3, 2.6, 11)
    
    plt.tight_layout()
    fig.savefig(dest_, dpi=300)
    
    if out_Bosse_M:
        return(bosse_M)


# Figure 7 ---------------------------------------------------------------------
def get_xlims(ax_):
    xl_ = ax_.get_xlim()
    lx_ = xl_[0] +  (xl_[1] - xl_[0]) * (1 / 30)
    ly_ = [xl_[0] +  (xl_[1] - xl_[0]) * (29.5 - 4.5) / 30,
            xl_[0] +  (xl_[1] - xl_[0]) * (29.5 - 1) / 30]
    lxl_ = xl_[0] +  (xl_[1] - xl_[0]) * (24 / 30)
    lyl_ = xl_[0] +  (xl_[1] - xl_[0]) * (3 / 30)
    
    return(lx_, ly_, lxl_, lyl_)


def plot_maps_sp_res(paths_, inputs_, fname_, t_=230, sim_sel=0):
    text_col2 = 'w'
    text_col = 'k'
    dt0_ = .2
    
    plt.close()
    fig, ax = plt.subplots(3, 5, figsize=(10, 6), sharex=False)  
    plt.tight_layout() 
    kl_ = 0 
    i_, sp_res = 0, 100
    for i_, sp_res in enumerate([100, 90, 60, 30, 10]):
        dt_ = dt0_ * sp_res / 100.
        
        inputs_['spat_res'] = sp_res
        bosse_M = BosseModel(inputs_, paths_)
        bosse_M.initialize_scene(sim_sel)
        I760 = np.where(bosse_M.M_F['wl'] == 760)[0][0]
    
        (RaoQ_w, RaoQ_sz, out_sz) = set_sizes(bosse_M, sp_res, bosse_M.scsz_)
        ones_layer = np.ones((out_sz * out_sz, 1))
    
        X_ = bosse_M.pred_scene_timestamp(t_, 'meteo_mdy', 'indx_mdy',
                                        lai020_pt=True, sp_res=100)    
    
        RF_ = bosse_M.pred_refl_factors(X_, bosse_M.scsz_, sp_res=sp_res)
        (rgb, ndvi_, nirv_) = rgb_indices(RF_, bosse_M.M_R['wl'])
        
        F_ = bosse_M.pred_fluorescence_rad(X_, bosse_M.scsz_, out_sz,
                                           sp_res=sp_res)

        LST_ = bosse_M.pred_landsurf_temp(X_, bosse_M.scsz_, sp_res=sp_res)
        
        (RaoQ_nirv, _) = gni.raoQ_grid(nirv_, wsz_=RaoQ_w)
        (RaoQ_f760, _) = gni.raoQ_grid(F_[:, :, [I760]], wsz_=RaoQ_w)
        (RaoQ_lst, _) = gni.raoQ_grid(LST_, wsz_=RaoQ_w)
        (_, _, _, f_alpha_nirv, _, _) = gni.varpart_grid(nirv_, wsz_=RaoQ_w)
        (_, _, _, f_alpha_f760, _, _) = gni.varpart_grid(F_[:, :, [I760]],
                                                         wsz_=RaoQ_w)
        (_, _, _, f_alpha_lst, _, _) = gni.varpart_grid(LST_, wsz_=RaoQ_w)
        
        if i_ == 0:
            vmin_lai_nirv, vmax_lai_nirv = get_im_cb_limits(
                nirv_, th_=.1, prec=1)
            vmin_lai_F, vmax_lai_F = get_im_cb_limits(
                F_[:, :, I760].reshape(-1), th_=.1, prec=1)
            vmin_lai_LST, vmax_lai_LST = get_im_cb_limits(
                LST_.reshape(-1), th_=.05, prec=0)
        
        # Reflectance
        im_ = ax[0, i_].imshow(nirv_, cmap='YlGn', vmin=vmin_lai_nirv,
                               vmax=vmax_lai_nirv)
        set_map_lbls(ax[0, i_], 'Spat. Res.: %d %%' % sp_res, fontsize=10)
        if sp_res == 10:
            cax = ax[0, i_].inset_axes([1.05, 0, 0.04, 1])
            fig.colorbar(im_, cax=cax, label='NIR$_{\\rm v}$ [-]')
            
        # Prepare label limits
        (lx_, ly_, lxl_, lyl_) = get_xlims(ax[0, i_])
        
        ax[0, i_].text(lx_ + dt_, ly_[0] + dt_,
                       '$\\bf \\it Q_{\\rm \\bf  Rao}$ = %.4f' %
                       RaoQ_nirv['RaoQ_mean'],
                       color=text_col2, fontsize=10, fontweight='bold')
        ax[0, i_].text(lx_, ly_[0],
                '$\\bf \\it Q_{\\rm \\bf  Rao}$ = %.4f' %
                RaoQ_nirv['RaoQ_mean'],
                color=text_col, fontsize=10, fontweight='bold')
        if RaoQ_sz > RaoQ_w:
            ax[0, i_].text(lx_ + dt_, ly_[1] + dt_,
                           '$\\bf \\it f_{\\alpha}$ = %.1f %%' %
                           f_alpha_nirv, color=text_col2, fontsize=10,
                           fontweight='bold')
            ax[0, i_].text(lx_, ly_[1],
                           '$\\bf \\it f_{\\alpha}$ = %.1f %%' %
                           f_alpha_nirv, color=text_col, fontsize=10,
                           fontweight='bold')
        ax[0, i_].text(lxl_+ dt_, lyl_+ dt_, get_char_label(kl_),
                       color=text_col2, fontweight='bold')
        ax[0, i_].text(lxl_, lyl_, get_char_label(kl_),
                       color=text_col, fontweight='bold')

        # Fluorescence
        im_ = ax[1, i_].imshow(F_[:, :, I760], cmap='PuRd', 
                               vmin=vmin_lai_F, vmax=vmax_lai_F)
        set_map_lbls(ax[1, i_], '', fontsize=10)
        if sp_res == 10:
            cax = ax[1, i_].inset_axes([1.05, 0, 0.04, 1])
            fig.colorbar(im_, cax=cax,
                         label='$F_{\\rm 760 nm}$ $\\rm [mW m^{-2} sr^{-1} nm^{-1}$')
                    
        ax[1, i_].text(lx_ + dt_, ly_[0] + dt_,
                       '$\\bf \\it Q_{\\rm \\bf  Rao}$ = %.4f' %
                       RaoQ_f760['RaoQ_mean'],
                       color=text_col2, fontsize=10, fontweight='bold')
        ax[1, i_].text(lx_, ly_[0],
                       '$\\bf \\it Q_{\\rm \\bf  Rao}$ = %.4f' %
                       RaoQ_f760['RaoQ_mean'],
                       color=text_col, fontsize=10, fontweight='bold')
        if RaoQ_sz > RaoQ_w:
            ax[1, i_].text(lx_ + dt_, ly_[1] + dt_,
                           '$\\bf \\it f_{\\alpha}$ = %.1f %%' %
                           f_alpha_f760, color=text_col2, fontsize=10,
                           fontweight='bold')
            ax[1, i_].text(lx_, ly_[1],
                           '$\\bf \\it f_{\\alpha}$ = %.1f %%' %
                           f_alpha_f760, color=text_col, fontsize=10,
                           fontweight='bold')
        ax[1, i_].text(lxl_ + dt_, lyl_ + dt_, get_char_label(kl_ + 5),
                       color=text_col2, fontweight='bold')
        ax[1, i_].text(lxl_, lyl_, get_char_label(kl_ + 5),
                       color=text_col, fontweight='bold')
        
        # LST
        im_ = ax[2, i_].imshow(LST_, cmap='RdYlBu_r',
                               vmin=vmin_lai_LST, vmax=vmax_lai_LST)
        set_map_lbls(ax[2, i_], '', fontsize=10)
        if sp_res == 10:
            cax = ax[2, i_].inset_axes([1.05, 0, 0.04, 1])
            fig.colorbar(im_, cax=cax, label='LST [K]')
         
        ax[2, i_].text(lx_ + dt_, ly_[0] + dt_,
                       '$\\bf \\it Q_{\\rm \\bf  Rao}$ = %.4f' %
                       RaoQ_lst['RaoQ_mean'],
                       color=text_col2, fontsize=10, fontweight='bold')
        ax[2, i_].text(lx_, ly_[0],
                       '$\\bf \\it Q_{\\rm \\bf  Rao}$ = %.4f' %
                       RaoQ_lst['RaoQ_mean'],
                       color=text_col, fontsize=10, fontweight='bold')
        if RaoQ_sz > RaoQ_w:
            ax[2, i_].text(lx_ + dt_, ly_[1] + dt_,
                           '$\\bf \\it f_{\\alpha}$ = %.1f %%' %
                           f_alpha_lst, color=text_col2, fontsize=10,
                           fontweight='bold')
            ax[2, i_].text(lx_, ly_[1],
                           '$\\bf \\it f_{\\alpha}$ = %.1f %%' %
                           f_alpha_lst, color=text_col, fontsize=10,
                           fontweight='bold')
        ax[2, i_].text(lxl_ + dt_, lyl_ + dt_, get_char_label(kl_ + 2 * 5),
                       color=text_col2, fontweight='bold')
        ax[2, i_].text(lxl_, lyl_, get_char_label(kl_ + 2 * 5),
                       color=text_col, fontweight='bold')
        
        kl_ += 1
    
    plt.tight_layout()
    fig.savefig(fname_, dpi=300)


# Figure 8 ---------------------------------------------------------------------
def compute_EF(inputs_, paths_, dest_csv, sim_sel=0):
    print('Computing hourly ecosystem functions, it can take ~50 min')
    t_start = time.time()
    
    bosse_M = BosseModel(inputs_, paths_)
    bosse_M.initialize_scene(sim_sel)
    
        
    ts_length = getattr(bosse_M, 'ts_length')
    
    # Preallocate output dataset
    columns_df = ['time', 'GPP', 'Rb', 'Rb_15C', 'NEP', 'LUE', 'LUEgreen',
                  'lE', 'T', 'H', 'Rn', 'G', 'ustar']
    df_eF = pd.DataFrame( data=(np.zeros((len(range(0, ts_length)),
                                          len(columns_df))) * np.nan),
                         columns=columns_df)
    
    # t_ = 5122
    for t_ in range(0, ts_length):
        # # Set scene values -----------------------------------------
        # Update plant traits daily only
        X_ = bosse_M.pred_scene_timestamp(
            t_, 'meteo_', 'indx_day', lai020_pt=True)

        # # Predict ecosystem function -------------------------------
        (df_eF.loc[t_]['time'], df_eF.loc[t_]['GPP'], df_eF.loc[t_]['Rb'],
         df_eF.loc[t_]['Rb_15C'], df_eF.loc[t_]['NEP'], df_eF.loc[t_]['LUE'],
         df_eF.loc[t_]['LUEgreen'], df_eF.loc[t_]['lE'], df_eF.loc[t_]['T'],
         df_eF.loc[t_]['H'], df_eF.loc[t_]['Rn'], df_eF.loc[t_]['G'],
         df_eF.loc[t_]['ustar']) = bosse_M.pred_ecosys_funct(t_, X_, 'meteo_')
    
    df_eF.to_csv(dest_csv, sep=';', index=False)
    print('\tElapsed time: %.2f min' % ((time.time() - t_start)/60))
    
    return(df_eF)


def plot_EF_time_series(spt_, kz_, inputs_, paths_, dest_, sim_sel=0):
    inputs_['sp_pattern'] = spt_
    inputs_['clim_zone'] = kz_
    
    dest_csv = dest_.replace('.png', '.csv')
    if os.path.exists(dest_csv):
        df_eF = pd.read_csv(dest_csv, sep=';')
    else:
        # This can take time
        df_eF = compute_EF(inputs_, paths_, dest_.replace('.png', '.csv'),
                           sim_sel=sim_sel)

    df_eF['DoY'] = df_eF['time'] / 24.
        
        
    plt.close()
    fig, ax = plt.subplots(4, 3, figsize=(8, 8), sharex=True) 
    ax = ax.ravel()
    plt.tight_layout()
    
    vars2plot = ['GPP', 'Rb', 'Rb_15C', 'NEP', 'LUE', 'LUEgreen', 'lE', 'T',
                 'H', 'Rn', 'G', 'ustar']
    
    col_ = ['steelblue', 'indigo']
    sym_ = ['-', '--']
    
    for i_, v_ in enumerate(vars2plot):
        print(v_)
        ax[i_].plot(df_eF['DoY'], df_eF[v_], '-', c='steelblue',
                    label=bosse_M.get_variable_symbol(v_), lw=.5)
        ax[i_].set_ylabel(bosse_M.get_variable_symbol(v_) + ' ' +
                            bosse_M.get_variable_units(v_), fontsize=10)
 
        xl_ = ax[i_].get_xlim()
        yl_ = ax[i_].get_ylim()
        ax[i_].set_xlabel('DoY [d]')
        ax[i_].text(xl_[0] + .85 * (xl_[1] - xl_[0]),
                    yl_[0] + .9 * (yl_[1] - yl_[0]),
                    get_char_label(i_))
        ax[i_].grid()
                
    plt.tight_layout()
    fig.savefig(dest_, dpi=300)
    plt.close()


# %% 3) Manuscript figures showing BOSSE simulations ###########################
# Define output folder
output_folder = parent_foler + '//Manuscript_bosse_v1_0_figures//'
if os.path.isdir(output_folder) == False:
    os.makedirs(output_folder)

# Generate standard set of input options and paths
(inputs_, paths_) = set_up_paths_and_inputs(None, output_folder,
                                            create_out_folder=False,
                                            pth_root=path_bosse,
                                            pth_inputs=path_inputs,
                                            pth_models=path_models)
print(f'The manuscript figures will be stored in {output_folder}')
# These are the default configuration options
print_dict(inputs_, 'inputs')

bosse_M = BosseModel(inputs_, paths_)
bosse_spatial_patterns = bosse_M.get_input_descriptors('sp_pattern')

# %% Figure 2
fig_name = 'Fig_2.png'
if os.path.isfile(output_folder + fig_name) is False:
    plot_GSI(output_folder + fig_name, bosse_M)


# %% Figure 3
fig_name = 'Fig_4.png'
if os.path.isfile(output_folder + fig_name) is False:
    plt_sp_patterns(paths_, inputs_, output_folder + fig_name)

# %% Figure 4
fig_name = 'Fig_5.png'
if os.path.isfile(output_folder + fig_name) is False:
    (inputs_, paths_) = set_up_paths_and_inputs(None, output_folder,
                                                create_out_folder=False,
                                                pth_root=path_bosse,
                                                pth_inputs=path_inputs,
                                                pth_models=path_models)

    plot_var_time_series(['LAI', 'Cab'],
                        ['$\\rm m^2 m^{-2}]$', '[$\\rm \\mu g cm^-2$]'],
                        inputs_, paths_, output_folder + fig_name)

# %% Figure 5
fig_name = 'Fig_6.png'
count_ = 1
if os.path.isfile(output_folder + fig_name) is False:
    for spt_ in bosse_spatial_patterns:        
        (inputs_, paths_) = set_up_paths_and_inputs(None, output_folder,
                                                    create_out_folder=False,
                                                    pth_root=path_bosse,
                                                    pth_inputs=path_inputs,
                                                    pth_models=path_models)
        inputs_['sp_pattern'] = spt_
        if spt_ == 'intermediate':
            fig_name_i = (output_folder + fig_name)
        else:
            fig_name_i = (output_folder + f'Fig_S10_{count_}.png')
            count_ += 1
            
        if os.path.isfile(fig_name_i) is False:
            plot_maps(paths_, inputs_, fig_name_i)

# %% Figure 6
fig_name = 'Fig_7.png'
count_ = 1
if os.path.isfile(output_folder + fig_name) is False:
    for spt_ in bosse_spatial_patterns: 
        (inputs_, paths_) = set_up_paths_and_inputs(None, output_folder,
                                                    create_out_folder=False,
                                                    pth_root=path_bosse,
                                                    pth_inputs=path_inputs,
                                                    pth_models=path_models)
        inputs_['sp_pattern'] = spt_
        if spt_ == 'intermediate':
            fig_name_i = (output_folder + fig_name)
        else:
            fig_name_i = (output_folder + f'Fig_S11_{count_}.png')
            count_ += 1
            
        if os.path.isfile(fig_name_i) is False:
            plot_maps_sp_res(paths_, inputs_, fig_name_i)

# %% Figure 7
sim_sel = 0
fig_name = 'Fig_8.png'
if os.path.isfile(output_folder + fig_name) is False:
    plot_EF_time_series('intermediate', 'Continental', inputs_, paths_,
                        output_folder + fig_name, sim_sel)

# %% Figure S12
fig_name = 'Fig_S12.png'
if os.path.isfile(output_folder + fig_name) is False:
    bosse_M.initialize_scene(sim_sel)
    bosse_M.plot_meteo_ts(fname_=output_folder + fig_name)
