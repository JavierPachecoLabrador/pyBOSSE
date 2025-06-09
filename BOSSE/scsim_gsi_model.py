# %% 0) Imports
import numpy as np
from scipy.interpolate import interp1d

import time as time
import rle

import matplotlib.pyplot as plt
import matplotlib as mplt

from BOSSE.helpers import div_zeros, zone_snum


# %% 1) GSI model (from Nuno) Forker et al. 2014
def get_GSI_ori_parameters():
    # Parameters from Table S5. Final parameters for LPJmL‐GSI (Suppl 3.1 for
    #   PFT codes definition)
    #   TrBE: EBF (evergreen broadleaved forest) AND tropical biome
    #   TrBR: DBF (deciduous broadleaved forest) AND tropical biome
    #   TeNE: ENF (evergreen needleleaved forest) AND temperate biome
    #   TeBE: EBF (evergreen broadleaved forest) AND temperate biome
    #   TeBS: DBF (deciduous broadleaved forest) AND temperate biome
    #   BoNE: ENF (evergreen needleleaved forest) AND boreal biome
    #   BoBS: DBF (deciduous broadleaved forest) AND boreal biome
    #   BoNS: DNF (deciduous needleleaved forest) AND boreal biome
    #   TrH: FPCherb AND tropical biome
    #   Old TeH: FPCherb AND temperate OR boreal biome.
    #       The TeH was further splitted in a new temperate herbaceous and a
    #       polar herbaceous PFT to separate between temperate grasslands and
    #       tundra:
    #   TeH (new): old TeH AND temperate OR boreal biome AND boreal trees < 0.3
    #   PoH: old TeH AND (boreal biome OR Koeppen‐Geiger E climate) AND boreal
    #       trees > 0.3
    ts5_ = dict()
    ts5_['pft_in'] = ['TrBE', 'TrBR', 'TeNE', 'TeBE', 'TeBS', 'BoNE', 'BoBS',
                      'BoNS', 'TrH', 'TeH', 'PoH']
    ts5_['sl_tmin'] = [1.01, 0.24, 0.22, 0.55, 0.26, 0.10, 0.22, 0.15, 0.91,
                       0.31, 0.13]
    ts5_['base_tmin'] = [8.30, 7.66, -7.81, -0.63, 13.69, -7.52, 2.05, -4.17,
                         6.42, 4.98, 2.79]
    ts5_['T_tmin'] = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
                      0.01, 0.20]
    ts5_['sl_heat'] = [1.86, 1.63, 1.83, 0.98, 1.74, 0.24, 1.74, 0.24, 1.47,
                       0.24, 0.24]
    ts5_['base_heat'] = [38.64, 38.64, 35.26, 41.12, 41.51, 27.32, 41.51,
                         44.60, 29.16, 32.04, 26.12]
    ts5_['T_heat'] = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
                      0.20, 0.20]
    ts5_['sl_light'] = [77.17, 23.00, 20.00, 18.83, 58.00, 14.00, 58.00, 95.00,
                        64.23, 23.00, 23.00]
    ts5_['base_light'] = [55.53, 13.01, 4.87, 39.32, 59.78, 3.04, 59.78, 130.1,
                          69.90, 75.94, 50.00]
    ts5_['T_light'] = [0.52, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
                       0.20, 0.38]
    ts5_['sl_water'] = [5.14, 7.97, 5.00, 5.00, 5.24, 5.00, 5.24, 5.00, 0.10,
                        0.52, 0.88]
    ts5_['base_water'] = [5.00, 22.21, 8.61, 8.82, 20.96, 0.01, 20.96, 2.34,
                          41.72, 53.07, 1.00]
    ts5_['T_water'] = [0.44, 0.13, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.17,
                       0.01, 0.94]

    # Emax needed to compute water availability. PFT parameters from Table 1
    # in Gerten et al., 2004
    emax_ = dict()
    # First column Tropical (mostly 7), second Temperate (mostly 5)
    emax_ = np.array([[7., 5.],  # DNF
                     [7., 5.],  # ENF
                     [7., 5.],  # DBF
                     [7., 5.],  # EBF
                     [7., 5.],  # SHR
                     [5., 5.],  # GRAC3
                     [7., 7.]])  # GRAC4

    return (ts5_, emax_)


def fun_GSI(x, pfGSI, slope, base, tau, delta_th, delta_x=1.):
    import warnings
    warnings.filterwarnings("error")
    
    if isinstance(delta_th, float):
        delta_th = delta_th * np.ones(pfGSI.shape[0])
        
    exp_ = -slope * (x - base)
    # Prevent np.exp() overflow. Hard-code the number to avoid calculation
    # np.log(np.finfo('d').max) = 709.782712893384
    exp_[exp_ > 709.782712893384] = 709.782712893384
    delta_ = ((1 / (1 + np.exp(exp_))) - pfGSI) * tau
    
    # Prevent extremely fast chantes
    th_ = delta_th * delta_x
    I_ = (delta_ > th_)
    delta_[I_] = th_[I_]
    I_ = (delta_ < -1. * th_)
    delta_[I_] = -1. * th_[I_]
    
    out_ = pfGSI + delta_
    warnings.resetwarnings()
    return (out_)


def GSI_loop(x, y0, slope, base, tau, delta_th, delta_x=1.):
    if len(np.array(slope).shape) == 0:
        y = np.zeros(x.shape[0]) * np.nan
    else:
        y = np.zeros(np.array([x.shape[0]] + [sz_ for sz_ in slope.shape]))

    y[0] = y0
    for i_, xi_ in enumerate(x[1:]):
        y[i_ + 1] = fun_GSI(xi_, y[i_], slope, base, tau, delta_th,
                            delta_x=delta_x)

    return (y)


def runGSI(x, y0, delta_th, slope=.5, base=.05, tau=.02, inv_val=False,
           delta_x=1.):
    # A way to circumvent the initial condition problem
    # y00 = GSI_loop(x, y0, slope, base, tau)
    # y = GSI_loop(x, y00[-1], slope, base, tau)
    # Also, reverse meteo so that initial conditions are coherent with
    # themselves not with the end of the timeseries conditions
    ext_ = 100
    rev_x = x[::-1, :]
    x_ext = np.concatenate((rev_x[-ext_:], x[1:, :]), axis=0)
    y00 = GSI_loop(x_ext, y0, slope, base, tau, delta_th, delta_x=delta_x)
    # Then, repeat the loop starting from x_ext[ext_ * 2 - 2], which features the same
    # meteorological conditions than x_ext[0]
    y = GSI_loop(x_ext, y00[int(ext_ * 2 - 2), :], slope, base, tau, delta_th,
                 delta_x=delta_x)
    y = y[-x.shape[0]:]
    if inv_val:
        y = 1. - y
    return (y)


def scale_var(x_):
    x_min = np.nanmin(x_)
    return(div_zeros(x_ - x_min, np.nanmax(x_) - x_min))


def scale_var_range(x_, x_min, x_max):
    x1 = (x_ - x_min)
    x1[x1 < 0.] = 0.
    out_ = x1 * div_zeros(x_max, x_max - x_min)
    return(out_)


def generate_sp_gsi_param_ranges(slope_b=[2, 30], base_b=[.4, .6],
                                 tau_b=[.01, 0.3]):
    ranges_ = np.sort(np.array(list(
        [np.random.uniform(slope_b[0], slope_b[1], 2),
         np.random.uniform(base_b[0], base_b[1], 2),
         np.random.uniform(tau_b[0], tau_b[1], 2)]), dtype=object), axis=1,
                      kind='mergesort')

    return (ranges_)


def generate_sp_gsi_params(veg_, sp_pft, vr_typ, x_, nPT, inv_val=False):
    GSI_params = np.zeros((3, len(sp_pft), nPT))
    delta_th = np.zeros(len(sp_pft))
    # GSI differ per species, according to its PFT, there's no intraspecific
    # variability; but a variabiliy per parameter
    i_, pft_ = 0, sp_pft[0]
    for i_, pft_ in enumerate(sp_pft):
        I_ = veg_['pft_in'].index(pft_)
        # Slope. 5 % variability between species and 5 % between treatments
        GSI_params[0, i_] = (veg_['sl_%s' % vr_typ][I_] *
                             np.random.normal(1, .05) *
                             np.random.normal(1, .05, size=(nPT)))
        # Base.
        # 5 % variability between species and 5 % between treatments
        GSI_params[1, i_] = (veg_['base_%s' % vr_typ][I_] *
                             np.random.normal(1, .05) *
                             np.random.normal(1, .05, size=(nPT)))
        # Tau
        # 1 % variability between species and .1 % between treatments.
        # Truncated to values < 1.
        GSI_params[2, i_] = (veg_['T_%s' % vr_typ][I_] *
                             np.abs((np.random.normal(1, .01) *
                             np.random.normal(1, .001, size=(nPT)))))
        
        # Define the limit at which a PFT GSI can change
        delta_th[i_] = veg_['gsi_lim'][I_]
    return (GSI_params, delta_th)


def show_GSI(x_smp, GSI_R, GSI_cs_smp, paths_, simnum_, veg_,
             sp_pft, kzone_, title='', vr_in='', x_smp0=np.arange(0., 1., .05),
             inv_val=False):
    shp_ = GSI_cs_smp[0].shape
    GSI_delta_th = (np.inf * np.ones(shp_)).reshape(-1)
    y = GSI_loop(x_smp0, 0,
                 GSI_cs_smp[0].reshape(-1),
                 GSI_cs_smp[1].reshape(-1),
                 GSI_cs_smp[2].reshape(-1),
                 GSI_delta_th).reshape(-1, shp_[0], shp_[1])

    if inv_val:
        y = 1. - y
    var_ = title.replace('f(', '').replace(')', '')
    x_ax1 = np.arange(1, GSI_R.shape[0] + 1)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8., 4.))
    ax[0].grid()
    for i_ in range(shp_[0]):
        col_ = veg_['pft_col'][veg_['pft_in'].index(sp_pft[i_])]
        ax[0].fill_between(x_smp0, y[:, i_].min(axis=1), y[:, i_].max(axis=1),
                           facecolor=col_, alpha=.5)
        ax[0].plot(x_smp0, y[:, i_].mean(axis=1), color=col_)
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$GSI$')
    ax[0].set_title(title)

    ax[1].grid()
    ax[1].plot(scale_var(x_smp), '--k', label=var_ + ('(scaled)'))
    for i_ in range(shp_[0]):
        col_ = veg_['pft_col'][veg_['pft_in'].index(sp_pft[i_])]
        ax[1].fill_between(x_ax1, GSI_R[:, i_, :].min(axis=1),
                           GSI_R[:, i_, :].max(axis=1),
                           facecolor=col_, alpha=.5)
        ax[1].plot(x_ax1, GSI_R[:, i_].mean(axis=1), color=col_)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$GSI$ or $x$')
    ax[1].set_title(title + ' time series')
    h_, l_ = ax[1].get_legend_handles_labels()
    ax[1].legend([h_[0]], [l_[0]])
    plt.savefig(paths_['2_out_folder_plots'] + 'GSI-sp_%s_%s.png' % (
            vr_in, zone_snum(simnum_, kzone_)), dpi=250)
    plt.close()


def smooth_gsi(GSI_, treshold_=0.02):
    x_ = np.arange(GSI_.shape[0])
    for i_ in range(GSI_.shape[1]):
        diff_ = np.concatenate((np.zeros((1)),
                               np.abs(GSI_[1:, i_] - GSI_[:-1, i_])))
        I_ = diff_ > treshold_
        if np.any(I_):
            intrp_ = interp1d(x_[~I_], GSI_[~I_, i_],
                              fill_value='extrapolate')
            GSI_[I_, i_] = intrp_(x_[I_])
    return (GSI_)


def get_GSI_and_param(x_, y0_, paths_, simnum_, veg_, sp_pft, vr_typ,
                      nPT, kzone_, inv_val=False, doplot=False, title='',
                      x_smp0=None, treshold_=None, var_in='', delta_x=1.):
    # Get the parameters
    # GSI_ranges = generate_sp_gsi_param_ranges()
    GSI_param, GSI_delta_th = generate_sp_gsi_params(veg_, sp_pft, vr_typ, x_,
                                                     nPT, inv_val=inv_val)

    # Get the vales
    GSI_ = np.zeros((x_.shape[0], len(sp_pft), nPT))
    for i_ in range(nPT):
        GSI_[:, :, i_] = runGSI(x_, y0_, GSI_delta_th,
                                slope=GSI_param[0][:, i_],
                                base=GSI_param[1][:, i_],
                                tau=GSI_param[2][:, i_],
                                inv_val=inv_val, delta_x=delta_x)

    # Smooth if needed
    if treshold_ != None:
        for i_ in range(GSI_.shape[1]):
            GSI_[:, i_, :] = smooth_gsi(GSI_[:, i_, :], treshold_=treshold_)

    # Plot
    if doplot:
        if x_smp0 == None:
            if np.ptp(x_) > .0:
                x_smp0 = np.arange(x_.min(), x_.max(), np.ptp(x_)/100)
            else:
                x_smp0 = np.array(x_[0])
            
        show_GSI(x_, GSI_, GSI_param, paths_, simnum_, veg_, 
                 sp_pft, kzone_, title=title, vr_in=var_in, x_smp0=x_smp0,
                 inv_val=inv_val)

    return (GSI_, GSI_param)


def get_water_avail(veg_, sp_pft, meteo_):
    # Supplementary 1.2 in Forkel et al 2014.
    # Water supply
    S_ = np.zeros((meteo_['SMC'].shape[0], len(sp_pft)))
    D_ = np.zeros((meteo_['SMC'].shape[0], len(sp_pft)))
    for i_, pft_ in enumerate(sp_pft):
        I_ = veg_['pft_in'].index(pft_)
        S_[:, i_] = veg_['Emax'][I_] * meteo_['wr'].values

        # Atmospheric water demand D_
        D_[:, i_] = (meteo_['PET'].values * 1.391 /
                     (1. +(3.26 / veg_['gpot'][I_])))
    D_[D_ < 0.] = 0. 
    
    W_ = 100 *div_zeros(S_, D_, out_=1.)
    W_[W_ > 100.] = 100.
    W_[W_ < 0.] = 0.

    return (W_)


def gsi_smooth_input(x_):
    # Apply monthly averages as in Forkel et al. 2014
    if x_.ndim == 1:
        x_ = x_.reshape(-1, 1)
    y_ = np.zeros(x_.shape)
    n_dates = x_.shape[0]
    for i_ in range(n_dates):
        I_ = np.arange(max(0, i_-15), min(i_+15, n_dates))
        y_[i_, :] = np.nanmean(x_[I_, :], axis=0)
    return (y_)


def get_GSI_Cs(GSI_all, PT_vars):
    # Redefine GSI_all for Cs, so that these are only produced during senescence
    Ics = PT_vars.index('Cs')
    # Get the derivaive
    dif_GSI_sp =  -1* np.round(np.minimum(0., (GSI_all[1:, :, Ics] -
                                               GSI_all[:-1, :, Ics])), 2)
    dif_GSI_sp = np.concatenate((
        dif_GSI_sp[np.array([0]), :], dif_GSI_sp[:, :]), axis=0) 
    dif_GSI_sd = np.round(np.maximum(0.03, (GSI_all[1:, :, Ics] -
                                          GSI_all[:-1, :, Ics])), 2)   
    dif_GSI_sd = np.concatenate((
        dif_GSI_sd[np.array([0]), :], dif_GSI_sd[:, :]), axis=0) 
      
    GSI_cs = np.zeros(dif_GSI_sd.shape)    
    for j_ in range(dif_GSI_sd.shape[1]):
        (event, counts0) = rle.encode(dif_GSI_sp[:, j_] > 0.)
        counts = np.concatenate((np.zeros(1, dtype=int), np.cumsum(counts0)))
        
        for i_, ev_ in enumerate(event):
            if ev_ == True:
                # print(i_, ev_, counts[i_ ], counts[i_+1])
                # Pigments grow during senescence
                GSI_cs[counts[i_ ]:counts[i_+1], j_] = (
                    GSI_cs[counts[i_ ]:counts[i_ + 1], j_] +
                    np.cumsum(dif_GSI_sp[counts[i_ ]:counts[i_ + 1], j_]) -
                    dif_GSI_sp[counts[i_ ], j_])
                # Make them disappear as a funciton of GSI, wiht a minimum rate,
                # which might be dominiating
                if counts[i_+1] < counts[-1]:
                    GSI_cs[counts[i_ + 1]:counts[i_ + 2], j_] = np.maximum(
                        0, (GSI_cs[counts[i_ + 1] - 1, j_] - 1 *
                        (np.cumsum(
                            dif_GSI_sd[counts[i_ + 1]:counts[i_ + 2], j_]) -
                         dif_GSI_sd[counts[i_ + 1], 0])))
        # Scale to the maximum value of the original GSI
        GSI_cs[:, j_] = (GSI_cs[:, j_] *
                         div_zeros(np.max(GSI_all[:, j_, Ics]),
                                   np.max(GSI_cs[:, j_]), 1.)[0][0])

    GSI_all[:, :, Ics] = 1. - GSI_cs
    return(GSI_all)


def get_decidious_period(gsi, perc_down_val=25., perc_up_val = 40.,
                         min_period=65.):
    perc_down = np.percentile(gsi, perc_down_val)
    perc_up = np.percentile(gsi, perc_up_val)
    
    # Select the decidious period, removing short declines
    I_ = np.zeros(gsi.shape, dtype=bool)
    (event, counts0) = rle.encode(gsi <= perc_up)
    counts = np.concatenate((np.zeros(1, dtype=int), np.cumsum(counts0)))
    for ii_, ev_ in enumerate(event):
        if (ev_ == True) and (counts0[ii_] > min_period):
            I_[counts[ii_]:counts[ii_ + 1]] = True
    
    return(I_, perc_down, perc_up)


def set_decidious_GSI(GSI_all, PT_vars, sp_pft):
    
    # Force decidious PFTs to reach 0. To do so, set the GSI < percentile 25 to 
    # 0, and scale linearly to percentile 50 to make it less abrupt, but only for
    # periods long enough (45 / 70 days for grasses or decidious)
    Iv_ = [PT_vars.index('Cab'), PT_vars.index('Cca'), PT_vars.index('Cant'),
           PT_vars.index('LAI')]
    min_period = {'GRAC3': 45., 'GRAC4': 45., 'DBF': 70., 'DNF': 70.}
    for i_, id_ in enumerate(sp_pft):
        if id_ in ['GRAC3', 'GRAC4', 'DBF', 'DNF']:
            for j_ in Iv_:
                I_, prc_down, prc_up = get_decidious_period(
                    GSI_all[:, i_, j_],  min_period=min_period[id_])
                
                GSI_all[I_, i_, j_] = scale_var_range(GSI_all[I_, i_, j_],
                                                      prc_down, prc_up)

    return(GSI_all)


def generate_GSI(meteo_, paths_, simnum_, veg_, sp_pft, PT_vars, kzone_,
                 doplot=False):
    # Get meteo conditions
    wav_ = gsi_smooth_input(get_water_avail(veg_, sp_pft, meteo_))
    Rin_ = gsi_smooth_input(meteo_['Rin'].values)
    Ta_ = gsi_smooth_input(meteo_['Ta'].values)
    
    # Plant development to foliar content to SMC. Since calculating water
    # availability requires variables that must be predicted and makes things
    # complex, I'll rather use the simpler model of Bayat et al.
    nPT = len(PT_vars)
    GSI_wav, GSI_wav_param = get_GSI_and_param(
        wav_, 0, paths_, simnum_, veg_, sp_pft, 'water', nPT, kzone_,
        doplot=doplot, title='f($W_{\\rm p}$)', var_in='Wavail')

    # Plant development to foliar content to Rin
    GSI_rin, GSI_rin_param = get_GSI_and_param(
        Rin_, 0, paths_, simnum_,  veg_, sp_pft, 'light', nPT, kzone_,
        doplot=doplot, title='f($R_{\\rm in}$)', var_in='Rin')

    # Plant development cold temperatures. Set lim 30 deg
    GSI_tcol, GSI_tcol_param = get_GSI_and_param(
        Ta_, 0, paths_, simnum_,  veg_, sp_pft, 'tmin', nPT, kzone_,
        doplot=doplot, title='f($T_{\\rm a, cold}$)', var_in='Tcold')

    # Plant development warm temperatures. Set lim 30 deg
    GSI_twrm, GSI_twrm_param = get_GSI_and_param(
        Ta_, 1, paths_, simnum_,  veg_, sp_pft, 'heat', nPT, kzone_,
        doplot=doplot, inv_val=True, title='f($T_{\\rm a, warm}$)',
        var_in='Twarm')

    # GSI_all = scale_var(GSI_wav * GSI_rin * GSI_tcol * GSI_twrm)
    GSI_all_raw = GSI_wav * GSI_rin * GSI_tcol * GSI_twrm

    # Force decidious PFTs to reach 0 for some plant traits
    GSI_all = set_decidious_GSI(GSI_all_raw, PT_vars, sp_pft)

    if doplot:
        plot_generate_GSI(wav_, Rin_, Ta_, GSI_all, GSI_wav, GSI_wav_param,
                          GSI_rin, GSI_rin_param, GSI_tcol, GSI_tcol_param,
                          GSI_twrm, GSI_twrm_param, simnum_, sp_pft, veg_,
                          kzone_, paths_)

    # Redefine GSI_all for Cs, so that these are only produced during
    # senescence. This is done after ploting to get clearer plots
    GSI_all = get_GSI_Cs(GSI_all, PT_vars)

    return(GSI_all, GSI_wav, GSI_wav_param, GSI_rin, GSI_rin_param, GSI_tcol,
           GSI_tcol_param, GSI_twrm, GSI_twrm_param)


def generate_GSI_test(GSI_wav_param, GSI_rin_param, GSI_tcol_param,
                      GSI_twrm_param, GSI_delta_th, sp_pft, nPT):
    # Set excesively wide ranges to capture the full GSI variabilty range
    wav_test = np.repeat(np.linspace(-50, 150, 500).reshape(-1, 1),
                         GSI_wav_param.shape[1], axis=1)
    Rin_test =  np.repeat(np.linspace(0, 2500, 500).reshape(-1, 1),
                          GSI_rin_param.shape[1], axis=1)
    Ta_test = np.repeat(np.linspace(-60, 60, 500).reshape(-1, 1),
                        GSI_twrm_param.shape[1], axis=1)
    
    # Water availability
    GSI_wav_test = np.zeros((wav_test.shape[0], len(sp_pft), nPT))
    for i_ in range(nPT):
        GSI_wav_test[:, :, i_] = runGSI(
            wav_test, 0, GSI_delta_th, slope=GSI_wav_param[0][:, i_],
            base=GSI_wav_param[1][:, i_], tau=GSI_wav_param[2][:, i_],
            inv_val=False)

    # Incoming radiation
    GSI_rin_test = np.zeros((Rin_test.shape[0], len(sp_pft), nPT))
    for i_ in range(nPT):
        GSI_rin_test[:, :, i_] = runGSI(
            Rin_test, 0,GSI_delta_th,  slope=GSI_rin_param[0][:, i_],
            base=GSI_rin_param[1][:, i_], tau=GSI_rin_param[2][:, i_],
            inv_val=False)
    
    # Cold temperature
    GSI_tcol_test = np.zeros((Ta_test.shape[0], len(sp_pft), nPT))
    for i_ in range(nPT):
        GSI_tcol_test[:, :, i_] = runGSI(
            Ta_test, 0, GSI_delta_th, slope=GSI_tcol_param[0][:, i_],
            base=GSI_tcol_param[1][:, i_], tau=GSI_tcol_param[2][:, i_],
            inv_val=False)
    
    # Warm temperature
    GSI_twrm_test = np.zeros((Ta_test.shape[0], len(sp_pft), nPT))
    for i_ in range(nPT):
        GSI_twrm_test[:, :, i_] = runGSI(
            Ta_test, 0, GSI_delta_th, slope=GSI_twrm_param[0][:, i_],
            base=GSI_twrm_param[1][:, i_], tau=GSI_twrm_param[2][:, i_],
            inv_val=True)
    
    return(wav_test, Rin_test, Ta_test, GSI_wav_test, GSI_rin_test,
           GSI_tcol_test, GSI_twrm_test)


def plot_generate_GSI(wav_, Rin_, Ta_, GSI_all, GSI_wav, GSI_wav_param, GSI_rin,
                      GSI_rin_param, GSI_tcol, GSI_tcol_param, GSI_twrm,
                      GSI_twrm_param, simnum_, sp_pft, veg_, kzone_, paths_):

    cmap = mplt.colormaps.get_cmap('tab20')
    x_ax1 = np.arange(1, GSI_all.shape[0] + 1)
    shp_ = GSI_all.shape
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].grid()
    p1_ = ax[0].plot(GSI_wav.mean(axis=2), 'b',
                        label='f($W_{\\rm avail}$)')
    p2_ = ax[0].plot(GSI_rin.mean(axis=2), 'y', label='f($R_{\\rm in}$)')
    p3_ = ax[0].plot(GSI_tcol.mean(axis=2), 'c',
                        label='f($T_{\\rm a, cold}$)')
    p4_ = ax[0].plot(GSI_twrm.mean(axis=2), 'r',
                        label='f($T_{\\rm a, warm}$)')
    ax[0].set_title('$GSI$ for Simulation #%2d' % simnum_)
    ax[0].set_ylim([0, 1.3])
    ax[0].set_ylabel('$GSI_{\\rm components, averaged}$ [-]')
    ax[0].legend(handles=(p1_[0], p2_[0], p3_[0], p4_[0]),
                    ncol=5, loc='upper center', fontsize=10)
    ax[1].grid()
    for i_ in range(shp_[1]):
        col_ = veg_['pft_col'][veg_['pft_in'].index(sp_pft[i_])]
        ax[1].fill_between(x_ax1, GSI_all[:, i_, :].min(axis=1),
                            GSI_all[:, i_, :].max(axis=1),
                            facecolor=col_, alpha=.5)
        ax[1].plot(x_ax1, GSI_all[:, i_].mean(axis=1),
                    color=col_)
    ax[1].plot()
    ax[1].set_xlabel('$t$ [days]')
    ax[1].set_ylabel('$GSI_{\\rm all}$ [-]')
    ax[1].set_ylim([0, 1.3])

    upft = np.unique(sp_pft).tolist()
    for i_, pfti in enumerate(upft):
        col_ = veg_['pft_col'][veg_['pft_in'].index(pfti)]
        ax[1].plot(1, -1, marker='s', color=col_, label=pfti)
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, ncol=len(upft),
                    loc='upper center', fontsize=10)
    plt.savefig(paths_['2_out_folder_plots'] + 'GSI-sp_%s.png' % (
        zone_snum(simnum_, kzone_)), dpi=250)
    plt.close()

    ylb_ = ['Slope', 'Base', '$\\tau$']
    x_ax = np.arange(1, GSI_all.shape[1] + 1)
    fig, ax = plt.subplots(3, 4, figsize=(9, 8), sharex=True)
    for i_ in range(3):
        ax[i_, 0].grid()
        ax[i_, 0].plot(x_ax, GSI_wav_param[i_], '.b')
        ax[i_, 0].set_ylabel(ylb_[i_])
    ax[2, 0].set_xlabel('Species ID')
    ax[0, 0].set_title('f($W_{\\rm avail}$)')

    for i_ in range(3):
        ax[i_, 1].grid()
        ax[i_, 1].plot(x_ax, GSI_rin_param[i_], '.y')
        ax[i_, 1].set_ylabel(ylb_[i_])
    ax[2, 1].set_xlabel('Species ID')
    ax[0, 1].set_title('f($R_{\\rm in}$)')

    for i_ in range(3):
        ax[i_, 2].grid()
        ax[i_, 2].plot(x_ax, GSI_tcol_param[i_], '.c')
        ax[i_, 2].set_ylabel(ylb_[i_])
    ax[2, 2].set_xlabel('Species ID')
    ax[0, 2].set_title('f($T_{\\rm a, cold}$)')

    for i_ in range(3):
        ax[i_, 3].grid()
        ax[i_, 3].plot(x_ax, GSI_twrm_param[i_], '.r')
        ax[i_, 3].set_ylabel(ylb_[i_])
    ax[2, 3].set_xlabel('Species ID')
    ax[0, 3].set_title('f($T_{\\rm a, warm}$)')
    fig.tight_layout()
    plt.savefig(paths_['2_out_folder_plots'] + 'GSI-params-sp_%s.png' % (
        zone_snum(simnum_, kzone_)), dpi=250)
    plt.close()