# %% 0) Imports
import copy

import numpy as np
import time as time

from BOSSE.scsim_emulation import truncatedGMM
from BOSSE import scsim_gsi_model as gsi
from BOSSE import scsim_rtm as rtm
from BOSSE.helpers import div_zeros


# %% 1) Species simulation
def generate_all_vars():
    all_vars = ['N', 'Cab', 'Cca', 'Cant', 'Cs', 'Cw', 'Cdm', 'LIDFa', 'LIDFb',
                'LAI', 'hc', 'leafwidth', 'Vcmo', 'm', 'kV', 'Rdparam',
                'Type', 'stressfactor', 'BSMBrightness', 'BSMlat', 'BSMlon',
                'SMC', 'FC', 'tts', 'tto', 'psi', 'Gfrac', 'rss', 'rss_level',
                'Rin', 'Rli', 'Ta', 'p', 'ea', 'u']
    all_vals = np.array([1.5, 50., 17.5, 0., 0., .001, .001, -.35, .5,
                         3., 10., .01, 60., 8., .6396, .015,
                         0, 1., .8, 50., 55.,
                         .5, 50., 30., 0., 0., .35, 1E3, .4,
                         1.2E3, 400., 25., 1E3, 25., 20.])

    return(all_vars, all_vals)


def set_sim_vars(scsz_):
    # Set all the variables and predefined values
    (all_vars, all_vals) = generate_all_vars()

    X_ = np.repeat(np.repeat(all_vals.reshape(1, 1, -1), scsz_, axis=0),
                   scsz_, axis=1)

    # Follow the order for the Neural Networks
    PT_vars = ['N', 'Cab', 'Cca', 'Cant', 'Cs', 'Cw', 'Cdm', 'LIDFa', 'LIDFb',
               'LAI', 'hc', 'leafwidth', 'Vcmo', 'm']
    PT_LB = np.asarray([1., 0., 0., 0., 0., 4.0e-03, 1.9e-03, -1., -1.,
                        0., 0.1, 0.01, 0., 1.])
    PT_UB = np.asarray([3.6, 100., 40., 10., 3., 8.1E-02, 3.0E-02, 1., 1.,
                        15., 30., .1, 200., 25.])

    Soil_vars = ['BSMBrightness', 'BSMlat',  'BSMlon', 'SMC', 'FC']
    Soil_LB = np.asarray([.5, 20., 45., 5., 5.])
    Soil_UB = np.asarray([1., 40., 65., 70., 75.])

    # Indices locating the different variables
    I_cab = PT_vars.index('Cab')
    I_cs = PT_vars.index('Cs')
    I_lai = PT_vars.index('LAI')
    I_hc = PT_vars.index('hc')
    I_vcmo = PT_vars.index('Vcmo')
    I_m = PT_vars.index('m')
    I_pt = [all_vars.index(i_) for i_ in PT_vars]
    I_gmm = [all_vars.index(i_) for i_ in PT_vars[:7]]
    I_rnd = [all_vars.index(i_) for i_ in PT_vars[7:] if i_ not in ['hc',
                                                                    'Vcmo']]
    I_sf = all_vars.index('stressfactor')
    I_soil = [all_vars.index(i_) for i_ in ['BSMBrightness', 'BSMlat',
                                            'BSMlon', 'SMC', 'FC']]
    I_smc_ = all_vars.index('SMC')
    I_fc = all_vars.index('FC')
    I_met = [all_vars.index(i_) for i_ in ['Rin', 'Rli', 'Ta', 'p', 'ea', 'u']]
    I_rss = all_vars.index('rss')
    I_rssl = all_vars.index('rss_level')
    I_ang = [all_vars.index(i_) for i_ in ['tts', 'tto', 'psi']]
    I_rin = all_vars.index('Rin')

    return(X_, all_vars, all_vals, PT_vars, PT_LB, PT_UB, Soil_vars, Soil_LB,
           Soil_UB, I_cab, I_cs, I_lai, I_hc, I_vcmo, I_m, I_pt, I_gmm, I_rnd,
           I_sf, I_soil, I_smc_, I_fc, I_met, I_rss, I_rssl, I_ang, I_rin)


def get_VcmoCab_Luo():
    # Vcmax model as a function of Cab. Table 1 from Luo et al 2019. GCB
    Luo_coeffs = np.asarray(list([[1.3, 3.72]] * 3 + [[0.66, 6.99]] +
                                 [[0.95, 14.71]] + 2 * [[1.98, 12.5]]),
                            dtype=object)
    Luo_sd = 5

    return (Luo_coeffs, Luo_sd)


def get_mBB_Miner():
    # Constatns to simulate m_BallBerry. Fig 1 from Miner et al. 2017
    # Assign ENF to DNF since there is no data
    miner_ori = {'CROC3': [13.3, 9.4], 'GRAC3': [13.4, 3.2],
                 'HerbPerenn': [13.2, 4.5], 'HerbAnnual': [10.4, 5.1],
                 'EBF': [9.8, 4.6], 'DBF': [8.7, 5.1], 'SHB': [7.4, 3.9],
                 'ENF': [6.8, 2.5], 'CROC4': [5.8, 3.8], 'GRAC4': [9.8, 1.0]}
    miner_eq = ['ENF', 'ENF', 'DBF', 'EBF', 'SHB', 'GRAC3', 'GRAC4']
    miner_mean = [0]*len(miner_eq)
    miner_std = [0]*len(miner_eq)
    
    for i_, eq_ in enumerate(miner_eq):
        miner_mean[i_] = miner_ori[eq_][0]
        miner_std[i_] = miner_ori[eq_][1]

    return (miner_mean, miner_std)


def get_gpot():
    # Potential canopy conductance estimated by Leuning et al. 2008
    # https://doi.org/10.1029/2007WR006562
    T2 = {'wsa': np.mean([3., 6.3]),
          'dec': np.mean([6.7, 3.8, 3.6, 4.6]),
          'con': np.mean([2., 3.9, 8.5, 0.27]),
          'gra': np.mean([4.8]),
          'cro': np.mean([5.3]),
          'ebf': np.mean([4.7, 7.6]),
          'wet': np.mean([4.8])}
    
    # ['DNF', 'ENF', 'DBF', 'EBF', 'SHB', 'GRAC3', 'GRAC4']
    gpot = [T2['con'], T2['con'], T2['dec'], T2['ebf'], T2['gra'],
            T2['gra'], T2['gra']]
    
    return(gpot)


def get_LAImax_Asner(clim_zone):
    # LAImax from Asner et al. 2003.  Table 2 after IQR analysis
    # Assign ENF to DNF since there is no data
    # ['DNF', 'ENF', 'DBF', 'EBF', 'SHB', 'GRAC3', 'GRAC4']
    if clim_zone == 'Tropical':
        LAImax = [15., 15., 8.9, 8.0, 4.5, 5.0, 5.0]
    else:
        LAImax = [15., 15., 8.8, 11.6, 4.5, 5.0, 5.0]

    return (LAImax)


def get_GSI_parameters(clim_zone, veg_0):
    veg_ = copy.deepcopy(veg_0)
    # Parameters of the GSM model from Forkel et al. 2014
    gsiS5, emax_ = gsi.get_GSI_ori_parameters()

    # Select Emax by climate
    if clim_zone == 'Tropical':
        veg_['Emax'] = emax_[:, 0].tolist()
    else:
        veg_['Emax'] = emax_[:, 1].tolist()

    # Select GSI params by climate
    # Assign ENF to DNF since there is no data
    if clim_zone == 'Tropical':
        gsi_eq = ['TeNE', 'TeNE', 'TrBR', 'TrBE', 'TrH', 'TrH', 'TrH']
    else:
        gsi_eq = ['TeNE', 'TeNE', 'TeBR', 'TeBE', 'TeH', 'TeH', 'TeH']

    gsi_var = list(gsiS5.keys())[1:]
    for i_, ky_ in enumerate(gsi_var):
        veg_[ky_] = np.zeros(len(veg_['pft_in']))
        for j_, eq_ in enumerate(gsi_eq):
            veg_[ky_][j_] = gsiS5[ky_][j_]

    return (veg_, gsi_var)


def set_plants(P_pft, clim_zone='Continental'):
    """
    Following the PFTs defined by Luo et al., 2019 for models predicting
        Vcmax from Cab. Use the Vcmo C4/C3 ratio 25/90 from Niu et al. 2006
        for GRA and from Zhang et al. 2018 for CRO
        Get data from Miner et al., 2017 distirbutions for Ball-berry slopes
    """
    veg_ = dict()
    # Define plant functinal types grouped Luo et al., 2019
    veg_['pft_in'] = ['DNF', 'ENF', 'DBF', 'EBF',
                      'SHB', 'GRAC3', 'GRAC4']
    veg_['pft_col'] = ['indigo', 'darkgreen', 'mediumpurple', 'forestgreen',
                       'olive', 'gold', 'khaki']
    veg_['pft_sym'] = ['--', '-', '--', '-',
                       ':', '-', '--']

    # Constants to simulate Vcmax
    veg_['Vcmo_Luo'], veg_['Vcmo_noise_SD'] = get_VcmoCab_Luo()

    # Constatns to simulate m_BallBerry. Fig 1 from Miner et al. 2017
    veg_['m_BB_Miner_mean'], veg_['m_BB_Miner_std'] = get_mBB_Miner()
    
    # Potential canopy conductance, from Leuning et al 2008
    veg_['gpot'] =  get_gpot()
    
    # LAImax from Asner et al. 2003.  Table 2 after IQR analysis
    veg_['LAI_max'] = get_LAImax_Asner(clim_zone)
    
    # Cab ranges from Croft et al. 2017. Fig. 7 expanding the 1STD to 3STD
    # Assume ENF=DNF and GRA3=GRA4
    veg_['Cab_min'] = [ 0.,  0., 0., 15.,  0.,  0.,  0.]
    veg_['Cab_max'] = [55., 55., 75., 75., 30., 40., 40.]
    
    # Canopy heght is defined as a funciton of the allometric relation described
    # by Jones 1998, as in Jules. Each PFT has associated a given scaling factor
    # to convert h_i where hc = h_i * LAI ^ (2/3). Values come from Table B3 in
    # Wiltshire et al. 2020.
    veg_['hc_factor'] = [6.5, 6.5, 6.5, 6.5, 1., .5, .5]

    # GSI phenological model parameters from Forkel et al. 2014 PFTs
    veg_, gsi_var = get_GSI_parameters(clim_zone, veg_)

    # PFTs not included in the Climatic Zone
    if P_pft is not None:
        I_ = [veg_['pft_in'].index(k_) for k_ in P_pft.iloc[:, 0]]
    else:
        I_ = [i_ for i_, k_ in enumerate(veg_['pft_in'])]
        
    # Get GSI change limit
    veg_['gsi_lim'] = [.015, .008, .015, .008, .015, .025, .025]
    
    # Remove values for the PFTs that are not presen in the climatic zone if
    # provided
    for ky_ in veg_:
        if isinstance(veg_[ky_], list) or isinstance(veg_[ky_], np.ndarray):
            veg_[ky_] = [veg_[ky_][i_] for i_ in I_]
    
    return (veg_)


def set_reco(P_pft, clim_zone='Continental'):
    # Define plant functinal types grouped Luo et al., 2019
    # veg_['pft_in'] = ['DBF', 'ENF', 'MF', 'SAV', 'WET', 'CROC3', 'CROC4',
    #                   'GRAC3', 'GRAC4', 'SH', 'EBF']
    # Ecosystem respiration parameters
    # Table 5. Migliavacca et al. 2010.
    # Correspondences: MF, EBF, MF, SAV, WET, CRO, CRO, GRA, GRA, SHB, EBF
    # Assign ENF to DNF since there is no data. Same parameters for GRAC3/C4
    reco_ = dict()
    reco_['pft_in'] = ['DNF', 'ENF', 'DBF', 'EBF', 'SHB', 'GRAC3', 'GRAC4']
    reco_['Rlai0'] = [1.02, 1.02, 1.27, -0.47, 0.42, 0.41, 0.41]
    reco_['alai'] = [0.42, 0.42, 0.34, 0.82, 0.57, 1.14, 1.14]
    reco_['LAI_max'] = get_LAImax_Asner(clim_zone)
    reco_['k2'] = [0.478, 0.478, 0.247, 0.602, 0.354, 0.578, 0.578]
    reco_['E0'] = [124.833, 124.833, 87.655, 52.753, 156.746, 101.181, 101.181]
    reco_['alpha'] = [0.604, 0.604, 0.796, 0.593, 0.850, 0.670, 0.670]
    reco_['K'] = [0.222, 0.222, 0.184, 2.019, 0.097, 0.765, 0.765]

    reco_['Rlai0_sd'] = [0.42, 0.42, 0.50, 0.50, 0.39, 0.71, 0.71]
    reco_['alai_sd'] = [0.08, 0.08, 0.10, 0.13, 0.17, 0.33, 0.33]
    reco_['LAI_max_sd'] = [0.05 * i_ for i_ in reco_['LAI_max']]
    reco_['k2_sd'] = [0.013, 0.013, 0.009, 0.044, 0.021, 0.062, 0.062]
    reco_['E0_sd'] = [4.656, 4.656, 4.405, 4.351, 8.222, 6.362, 6.362]
    reco_['alpha_sd'] = [0.065, 0.065, 0.031, 0.032, 0.070, 0.052, 0.052]
    reco_['K_sd'] = [0.070, 0.070, 0.064, 1.052, 1.304, 1.589, 1.589]

    # Filter out PFTs non included in the climatic zone
    I_ = [reco_['pft_in'].index(k_) for k_ in P_pft.iloc[:, 0]]
    reco_out = dict()
    for key_ in reco_.keys():
        if key_ == 'pft_in':
            reco_out['pft_in'] = [k_ for k_ in P_pft.iloc[:, 0]]
        else:
            reco_out[key_] =  np.array([reco_[key_][i_] for i_ in I_])

    return (reco_out)


def prune_PT(PT_, LB, UB):
    for i_ in range(PT_.shape[1]):
        PT_[PT_[:, i_] < LB[i_], i_] = LB[i_]
        PT_[PT_[:, i_] > UB[i_], i_] = UB[i_]
    return (PT_)


def prune_PTmap(PT_, LB, UB):
    if PT_.ndim == 3:
        for i_ in range(PT_.shape[2]):
            if i_ != 1:
                I_ = PT_[:, :, i_] < LB[i_]
                PT_[I_, i_] = LB[i_]
                I_ = PT_[:, :, i_] > UB[i_]
                PT_[I_, i_] = UB[i_]
    elif PT_.ndim == 2:
        for i_ in range(PT_.shape[1]):
            if i_ != 1:
                I_ = PT_[:, i_] < LB[i_]
                PT_[I_, i_] = LB[i_]
                I_ = PT_[:, i_] > UB[i_]
                PT_[I_, i_] = UB[i_]
    return (PT_)


def generate_PT(GM_T, sz_, LB, UB, PT_LBg, PT_UBg, veg_, pft_pos, I_cab, I_lai,
                I_hc, I_vcmo, I_gmm, I_rnd, overs_=20, rseed_=None):
    PT_ = np.zeros((sz_, LB.shape[0]))
    # Leaf biophysical
    PT_[:, I_gmm] = truncatedGMM(sz_, GM_T['GMMtraits'], GM_T['trans_'],
                                 GM_T['lambda_'], LB[I_gmm], UB[I_gmm],
                                 PT_LBg[I_gmm].reshape(1, -1),
                                 PT_UBg[I_gmm].reshape(1, -1), oversmp_=overs_,
                                 rseed_=rseed_)
    # Structural biophysical
    PT_[:, I_rnd] = LB[I_rnd] + np.random.random(
        (sz_, len(I_rnd))) * (UB[I_rnd] - LB[I_rnd])
    # Compute hc and add a 5 % random noise
    PT_[:, I_hc] = (compute_hc(veg_, pft_pos, PT_[:, I_lai]) *
                  np.random.normal(1, .05, size=PT_[:, I_lai].shape))
    # Compute Vcmax and add a 5 % random noise
    PT_[:, I_vcmo] = (compute_Vcmo(veg_, pft_pos, PT_[:, I_cab]) *
                  np.random.normal(1, .05, size=PT_[:, I_cab].shape))
    PT_ = prune_PT(PT_, LB, UB)

    return (PT_)


def generate_PT_rel_noise(GM_T, sz_, PT_in, LB, UB, PT_LBg, PT_UBg, rng_sd,
                          rel_intrasv, I_gmm, overs_=20, rseed_=None):
    # Generate relative noise with a fiven relative std
    PTn_ = np.zeros((sz_, 12))
    
    # Leaf biophysical
    PTn_0 = truncatedGMM(max(30, sz_), GM_T['GMMtraits'], GM_T['trans_'],
                         GM_T['lambda_'], LB[I_gmm], UB[I_gmm], PT_LBg, PT_UBg,
                         oversmp_=overs_, rseed_=rseed_)
    PTn_mean = np.abs(PTn_0.mean(axis=0, keepdims=True))
    PTn_cv = (PTn_0.std(axis=0, keepdims=True) / PTn_mean)
    PTn_[:, I_gmm] = (PTn_0[:sz_, :] - PTn_mean) * (rel_intrasv[:, I_gmm] /
                                                    PTn_cv)
    
    # Structural biophysical
    I_ = np.arange(I_gmm[-1] + 1, PTn_.shape[2] + 1)
    PTn_[:, I_] = (PT_in[:, I_] - (PT_in[:, I_] * rng_sd.normal(
        1, rel_intrasv[:, I_], size=PTn_[:, I_].shape)))
    
    # Center the noise
    PTn_out = PTn_ - np.mean(PTn_, axis=0, keepdims=True).repeat(sz_, axis=0)

    return (PTn_out)


def get_intraspvariab(interspecific_variability, rng_sd, LB_pft, UB_pft, PT_sp):
    # Get the intraspecific variance and variability assuming that it represents
    # around 30 % of the interspecific variance (Albert et al. 2010, Fig. 1),
    # ranging between 20 % and 40 % for within + between populations
    riv = 2.96 * (interspecific_variability *
           rng_sd.uniform(.2, .4, size=interspecific_variability.shape))

    riv_lb = np.maximum(PT_sp - riv, LB_pft)
    riv_ub = np.minimum(PT_sp + riv, UB_pft)

    return (riv_lb, riv_ub)


def keep_noise_in_bounds(PT_I0, PT_noise, LB_ns, UB_ns, ax_=0):
    PT_I = PT_I0 + PT_noise
    In_ = np.logical_or(PT_I < LB_ns,  PT_I > UB_ns)
    f_ = 1.
    while (In_).any() and f_ >= -1.:
        PT_In_i = copy.deepcopy(PT_noise)
        PT_In_i[In_] = PT_In_i[In_] * f_ 
        PT_I = PT_I0 + PT_In_i
        In_ = np.logical_or(PT_I < LB_ns,  PT_I > UB_ns)
        f_ -= .1
    return(PT_I)


def compute_Vcmo(veg_, spf_pos, Cab):
    # Equation from Luo et al. (2019)
    Vcmo = veg_['Vcmo_Luo'][spf_pos][0] * Cab + veg_['Vcmo_Luo'][spf_pos][1]
    
    return(Vcmo)


def compute_hc(veg_, spf_pos, LAI):
    # Equation from Jones 1998
    hc = veg_['hc_factor'][spf_pos] * LAI ** (2./3.)
    
    return(hc)


def center_noise(sigma_in, ax_=None):
    # Sustract the mean from a intra-specific variability to prevent biasing
    # the species plant traits
    if ax_ == 0:
        mean_ = np.nanmean(sigma_in, axis=0, keepdims=True).repeat(
            sigma_in.shape[0], axis=0)
    elif ax_ == 1:
        mean_ = np.nanmean(sigma_in, axis=1, keepdims=True).repeat(
            sigma_in.shape[1], axis=1)
    else:
        mean_ = np.nanmean(sigma_in)

    return(sigma_in - mean_)
    

def increase_cs(cs_in, mag, rng_mean, rng_sd, invtra_var):
    if invtra_var:
        isv_ = center_noise(np.abs(rng_sd.normal(
            0.,rng_sd.uniform(0., mag[1]), size=cs_in.shape[0])))
    else:
        isv_ = 0.
    cs_out = np.maximum(0., cs_in + rng_mean.uniform(0., mag[0]) + isv_) 
    
    return(cs_out)


def get_interspecific_variabily(PT_, sp_ab):
    mu = np.repeat(np.average(PT_, weights=sp_ab, axis=0, keepdims=True),
                   PT_.shape[0], axis=0)
    sigma_w = np.sqrt(np.average((PT_ - mu) ** 2, weights=sp_ab, axis=0))
    
    return (sigma_w)


def generate_correlated_plant_tratis(GM_T, num_samples, unique_pft, PT_LB,
                                     PT_UB, PT_LBg, PT_UBg, PT_vars, sp_pft,
                                     veg_, I_cab, I_lai, I_hc, I_vcmo, I_gmm,
                                     I_rnd, shrink_factor=1., overs_=300,
                                     type_='max'):
    if isinstance(sp_pft, (list, tuple, np.ndarray)) == False:
        sp_pft = np.array([sp_pft])

    # Generate the species mean trait values
    out_ = np.zeros((num_samples, len(PT_vars)))
    for pfti_ in unique_pft:
        I_ = np.where(sp_pft == pfti_)[0]
        if I_.shape[0] > 0:
            Iv_ = veg_['pft_in'].index(pfti_)
            range_ = PT_UB[pfti_] - PT_LB[pfti_]
            if type_ == 'max':
                lbi_ = copy.deepcopy(PT_LB[pfti_])
                lui_ = PT_LB[pfti_] + range_ * shrink_factor
            elif type_ == 'min':
                lbi_ = PT_UB[pfti_] - range_ * shrink_factor
                lui_ = copy.deepcopy(PT_UB[pfti_])

            out_[I_, :] = generate_PT(GM_T, len(I_), lbi_, lui_, PT_LBg, PT_UBg,
                                      veg_, Iv_, I_cab, I_lai, I_hc, I_vcmo,
                                      I_gmm, I_rnd, overs_==overs_)

    return(out_)


def generate_plant_traits(sp_map, sp_id, S_max, sp_pft, unique_pft, veg_, GM_T,
                          PT_vars, PT_LBg, PT_UBg, PT_LB, PT_UB, I_cab, I_cs,
                          I_lai, I_hc, I_vcmo, I_gmm, I_rnd, rng_mean, rng_sd,
                          sp_ab, num_dis=-1, inc_cs=False, invtra_var=True,
                          type_='max'):
    # Fraction of dissimilar species
    if num_dis == -1:
        num_dis = int(np.random.random(1)[0] * S_max)
    num_sim = S_max - num_dis

    # Preallocate traits matrix
    PT_ = np.zeros((S_max, len(PT_vars)))

    # Generate dissimilar species mean trait values (except Vcmax or mBB)
    if num_dis > 0:
        PT_[:num_dis, :] = generate_correlated_plant_tratis(
            GM_T, num_dis, unique_pft, PT_LB, PT_UB, PT_LBg, PT_UBg, PT_vars,
            sp_pft[:num_dis], veg_, I_cab, I_lai, I_hc, I_vcmo, I_gmm, I_rnd,
            shrink_factor=1., overs_=30, type_=type_)
        
    # Generate similar species mean trait values  (except Vcmax or mBB)
    if num_sim > 0:
        PT_[num_dis:, :] = generate_correlated_plant_tratis(
            GM_T, num_sim, unique_pft, PT_LB, PT_UB, PT_LBg, PT_UBg, PT_vars,
            sp_pft[num_dis:], veg_, I_cab, I_lai, I_hc, I_vcmo, I_gmm, I_rnd,
            shrink_factor=.15, overs_=100, type_=type_)

    # Determine the maximum LAI of each PFT
    pft_LAImax = np.zeros(S_max)
    for spfi_ in unique_pft:
        Iv_ = veg_['pft_in'] .index(spfi_)
        pft_LAImax[sp_pft == spfi_] = veg_['LAI_max'][Iv_]
        
    # Assign the largest LAI o the PFTs with the largest maximum LAI
    # Use mergesort to preserve the relative order of equal valuesensure and
    # ensure reproducibility between different machines
    ord_pft_LAImax = np.argsort(pft_LAImax, kind='mergesort')
    ord_sp_LAImax = np.argsort(PT_[:, I_lai], kind='mergesort')
    PT_[ord_pft_LAImax, I_lai] = PT_[ord_sp_LAImax, I_lai]
   
    # Determine the maximum LAI of the species in the scene. Give some
    # margin for the intraspecific variability
    scLAImax = np.min((PT_UBg[I_lai] * 1.1, np.max(PT_[:, I_lai]) * 1.2))
            
    # PT map with intraspecific variability of foliar and structural traits
    # Do not check on variables where input limits should not macth the
    # simulated values
    # Compute intraspecific variabiliry
    interspecific_variability = get_interspecific_variabily(PT_, sp_ab)
    
    # Preallocate traits map
    PT_map = np.zeros((sp_map.shape[0], sp_map.shape[1], len(PT_vars)))

    # i_, id_ = 0, 0
    for i_, id_ in enumerate(sp_id):
        # Identify the species in the map
        I_ = sp_map == id_
        sp_npx = int(np.sum(I_))
        
        # Identify the specie's PFT
        pfti_ = sp_pft[i_]
        Iv_ = veg_['pft_in'].index(pfti_)
        
        # Set the PFT limits to determine the intraspecific variability. Allow
        # individuals to exceed in a 5 % the PFT traits range as long as
        # these do not overpass the global tresholds
        range_pft = PT_UB[pfti_] - PT_LB[pfti_]
        LB_pft = np.maximum(PT_LB[pfti_] - range_pft * .05, PT_LBg)
        UB_pft = np.minimum(PT_UB[pfti_] + range_pft * .05, PT_UBg)
        
        # Compute hc bounds as a function of LAI bounds. Account for random
        # noise added during generation
        LB_pft[I_hc] = compute_hc(veg_, Iv_, LB_pft[I_lai])
        UB_pft[I_hc] = compute_hc(veg_, Iv_, UB_pft[I_lai])
        # Compute Vcmo bounds as a function of Cab bounds. Account for random
        # noise added during generation
        LB_pft[I_vcmo] = compute_Vcmo(veg_, Iv_, LB_pft[I_cab])
        UB_pft[I_vcmo] = compute_Vcmo(veg_, Iv_, UB_pft[I_cab])

        # Add noise and keep data within the bounds
        if invtra_var is True:
            # Generate the intraspecific variability for the species and 
            # truncate it with the PFT-dependent bounds
            # POR AQUI. CHECK THAT THE VARIABILITY STAYS WITHIN THE PFT-BOUNDS IF THIS IS NOT DONE LATER
            intra_LB, intra_UB = get_intraspvariab(
                interspecific_variability, rng_sd, LB_pft,
                UB_pft, PT_[i_, :])
            
            # Generate the noise for that species. This process can make the
            # the specie's mean to deviate from the initially set mean value.
            PT_intra = generate_PT(GM_T, sp_npx, intra_LB, intra_UB, PT_LBg,
                                   PT_UBg, veg_, Iv_, I_cab, I_lai, I_hc,
                                   I_vcmo, I_gmm, I_rnd, overs_=20)
            PT_I0 = np.repeat(PT_[i_, :].reshape(1, -1), sp_npx, axis=0)
            
            # Just in case. Prevent values outside bounds
            PT_map[I_, :] = keep_noise_in_bounds(
                PT_I0, PT_intra - PT_I0, LB_pft, UB_pft)
            
        else:
            PT_map[I_, :] = copy.deepcopy(PT_[i_, :])
        
        # Increase max Cs content during dry season
        if inc_cs:            
            if sp_pft[i_] in ['GRAC3', 'GRAC4']:
                PT_map[I_, I_cs] = (increase_cs(
                    PT_map[I_, I_cs], [1., .01], rng_mean, rng_sd,
                    invtra_var))
            else:
                PT_map[I_, I_cs] = (increase_cs(
                    PT_map[I_, I_cs], [.2, .002], rng_mean, rng_sd,
                    invtra_var))

    # Correct LDIFa
    a_ind = PT_vars.index('LIDFa')
    b_ind = PT_vars.index('LIDFb')
    PT_map[:, :, a_ind],  PT_map[:, :, b_ind] = rtm.lidf_back_transform(
        PT_map[:, :, a_ind],  PT_map[:, :, b_ind])

    return (PT_map, scLAImax)
    

def generate_plant_trait_limits(sp_map, sp_id, sp_ab, S_max, sp_pft, veg_, GM_T,
                                PT_vars, PT_LB0, PT_UB0, I_cab, I_cs, I_lai,
                                I_hc, I_vcmo, I_m, I_gmm, I_rnd, rng_mean,
                                rng_sd, invtra_var=True):
    """Generate two dataset, one with minimum values, by truncating the
    distributions within a certain fracion of the bounds range and then
    a second with the remaining fraction of the variability range.
    Then for some variables, consider which variables should have max values
    at the moment of maximum development, and which should be minimum. For
    those unclear, which one is selected randomly.
    """
    # For each PFT, select produce an averaged value and a range of variability
    unique_pft = np.unique(sp_pft)
    local_av = dict()
    local_LB = dict() 
    local_UB = dict()    
    spfi_ = unique_pft[0]
    for spfi_ in unique_pft:
        # Set limits
        Iv_ = veg_['pft_in'] .index(spfi_)
        PT_LB_tmp = copy.deepcopy(PT_LB0)
        PT_UB_tmp = copy.deepcopy(PT_UB0)
        # Set Cab PFT-dependent bounds
        PT_LB_tmp[I_cab] = veg_['Cab_min'][Iv_]        
        PT_UB_tmp[I_cab] = veg_['Cab_max'][Iv_]
        # Set LAI PFT-dependent bounds        
        PT_UB_tmp[I_lai] = veg_['LAI_max'][Iv_]
        # Compute hc bounds as a function of LAI bounds
        PT_LB_tmp[I_hc] = compute_hc(veg_, Iv_, PT_LB_tmp[I_lai])
        PT_UB_tmp[I_hc] = compute_hc(veg_, Iv_, PT_UB_tmp[I_lai]) 
        # Compute Vcmo bounds as a function of Cab bounds
        PT_LB_tmp[I_vcmo] = compute_Vcmo(veg_, Iv_, PT_LB_tmp[I_cab])
        PT_UB_tmp[I_vcmo] = compute_Vcmo(veg_, Iv_, PT_UB_tmp[I_cab])
        # Set m PFT-dependent bounds
        PT_LB_tmp[I_m] = np.maximum(veg_['m_BB_Miner_mean'][Iv_] - 2.96 *
                                    veg_['m_BB_Miner_std'][Iv_], PT_LB0[I_m])
        PT_UB_tmp[I_m] = np.minimum(veg_['m_BB_Miner_mean'][Iv_] + 2.96 *
                                    veg_['m_BB_Miner_std'][Iv_], PT_UB0[I_m])
        # Set a random PFT mean value within the bounds. Get the average of 
        # prevent extreme values while keeping some variability
        local_av[spfi_] = np.mean(generate_PT(
            GM_T, 100, PT_LB_tmp, PT_UB_tmp, PT_LB0, PT_UB0, veg_,
            Iv_, I_cab, I_lai, I_hc, I_vcmo, I_gmm, I_rnd, overs_=5
            )[np.random.randint(0, 100, 3)], axis=0)
        
        # Ensure a relatively centered LAI value for the treshold
        local_av[spfi_][I_lai] = np.min((np.max((1.5, local_av[spfi_][I_lai])),
                                         local_av[spfi_][I_lai] * .75))
        
        # Generate range of variability around it
        FB_ = np.random.uniform(.1, .4, 1)[0] * (PT_UB_tmp - PT_LB_tmp)
        local_LB[spfi_] = np.max((PT_LB_tmp, local_av[spfi_] - FB_), axis=0)
        local_UB[spfi_] = np.min((PT_UB_tmp, local_av[spfi_] + FB_), axis=0)
        
    # Set number of dissimilar species so that is the same for both min and max
    # values
    num_dis = int(np.random.random(1)[0] * S_max)

    # Generate the maps with variables above and below the averaged value
    # Upper bound

    PT_map_1, scLAImax = generate_plant_traits(
        sp_map, sp_id, S_max, sp_pft, unique_pft, veg_, GM_T, PT_vars,
        PT_LB0, PT_UB0, local_av, local_UB, I_cab, I_cs, I_lai, I_hc, I_vcmo,
        I_gmm, I_rnd, rng_mean, rng_sd, sp_ab, num_dis=num_dis,
        invtra_var=invtra_var, inc_cs=True, type_='max')
    PT_UB0[I_lai] = copy.deepcopy(scLAImax)
    # Prune to keep all values within the global limits
    PT_map_1 = prune_PTmap(PT_map_1, PT_LB0, PT_UB0)

    # Lower bound
    PT_map_2, _ = generate_plant_traits(
        sp_map, sp_id, S_max, sp_pft, unique_pft, veg_, GM_T, PT_vars,
        PT_LB0, PT_UB0, local_LB, local_av, I_cab, I_cs, I_lai, I_hc, I_vcmo,
        I_gmm, I_rnd, rng_mean, rng_sd, sp_ab, num_dis=num_dis,
        invtra_var=invtra_var, inc_cs=False, type_='min')
    # Prune to keep all values within the global limits
    PT_map_2 = prune_PTmap(PT_map_2, PT_LB0, PT_UB0)

    # Select the values for the top and the bottom of the seasonal cycle
    # This process can produce desagreements between the runs of different
    # spatial patterns, particularly when values are close, which can make that
    # uncertainties bias the averaged value slightly
    PT_map_min = np.zeros(PT_map_1.shape)
    PT_map_max = np.zeros(PT_map_1.shape)
    for i_, var_ in enumerate(PT_vars):
        # Variables whose maximum value happes at the phenological maximum
        if var_ in ['N', 'Cab', 'Cca', 'Cw', 'Cdm', 'hc', 'LAI', 'leafwidth',
                    'Vcmo']:
            tmp_ = np.stack((PT_map_1[:, :, i_], PT_map_2[:, :, i_]))
            PT_map_min[:, :, i_] = np.min(tmp_, axis=0)
            PT_map_max[:, :, i_] = np.max(tmp_, axis=0)
        
            if var_ == 'hc':
                tmp_ = np.stack((PT_map_1[:, :, i_], PT_map_2[:, :, i_]))
                tmp_min = np.min(tmp_, axis=0)
                tmp_max = np.max(tmp_, axis=0)
                # i_, id_ = 0, sp_id[0]
                for j_, id_ in enumerate(sp_id):
                    I_ = sp_map == id_
                    # Allow variable hc for herbaceous species. Fix it for woody
                    # assuming woody vegetation canopy height does not change
                    # across seasons.
                    if sp_pft[j_] in ['GRAC3', 'GRAC4']:
                        PT_map_min[I_, i_] = tmp_min[I_]
                        PT_map_max[I_, i_] = tmp_max[I_]
                    else:
                        PT_map_min[I_, i_] = tmp_max[I_]
                        PT_map_max[I_, i_] = tmp_max[I_]
        # Variables whose maximum value happens at the phenological minimum      
        elif var_ == 'Cs':
            tmp_ = np.stack((PT_map_1[:, :, i_], PT_map_2[:, :, i_]))
            PT_map_min[:, :, i_] = np.max(tmp_, axis=0)
            # Reduce Cs at the green peak to prevent unrealistic simulations
            PT_map_max[:, :, i_] = np.min(tmp_, axis=0) * .05
        # Variables whose maximum value can happen at any of the extremes  
        else:
            # Decide this randomly but without using the RGN to prevent
            # differences between simulations with different spatial patterns
            if int(PT_map_1[:, :, i_].sum()) % 2 == 0:
                PT_map_min[:, :, i_] = PT_map_1[:, :, i_]
                PT_map_max[:, :, i_] = PT_map_2[:, :, i_]
            else:
                PT_map_min[:, :, i_] = PT_map_2[:, :, i_]
                PT_map_max[:, :, i_] = PT_map_1[:, :, i_]
    
    # Reduce the range of variability for evergreen species and set LAI and
    # photosynthetic pigments to 0 in decidious grasslands
    PT_map_max_out = copy.deepcopy(PT_map_max)
    PT_map_min_out = copy.deepcopy(PT_map_min)
    PT_map_delta = PT_map_max - PT_map_min
    
    i_, id_ = 0, 1
    Iv_ = [PT_vars.index('Cab'), PT_vars.index('Cca'), PT_vars.index('Cant'),
           I_lai]
    traits_dec = np.ones((PT_map_max_out.shape))
    traits_dec[:, :, Iv_] = 0.
    for i_, id_ in enumerate(sp_id):
        # Reduce the range of variability of all traits for evergreen species
        if sp_pft[i_] in ['ENF', 'EBF']:
            repeat_delta = True
            range_ = rng_mean.uniform(.5, .9)
            I_ = sp_map == id_
            # Decide this randomly but without using the RGN to prevent
            # differences between simulations with different spatial patterns
            if S_max % 2 == 0:
                PT_map_max_out[I_, :] = (PT_map_max[I_, :] -
                                         (range_ * PT_map_delta[I_, :]))
            else:
                PT_map_min_out[I_, :] = (PT_map_min[I_, :] +
                                         (range_ * PT_map_delta[I_, :]))
        elif sp_pft[i_] in ['GRAC3', 'GRAC4', 'DBF', 'DNF']:
            # Set LAI and photosynthetic pigments to 0
            I_ = sp_map == id_
            PT_map_min_out[I_] = PT_map_min[I_] * traits_dec[I_]
            # print(np.max(np.abs(PT_map_min_out[I_] - PT_map_min[I_])))
    PT_map_delta_out = PT_map_max_out - PT_map_min_out

    return (PT_map_min_out, PT_map_max_out, PT_map_delta_out, num_dis)


def truncate_gauss_noise(x_, sigma_, mu_=1., lb=-np.inf, ub=np.inf):
    y_ = np.zeros((x_.shape))
    I_ = np.ones(x_.shape, dtype=bool)
    counter_ = 0
    if np.isclose(mu_, 1.):
        while (I_.any()) and (counter_ <= 20):
            y_[I_] = x_[I_] * (np.random.normal(mu_, sigma_, size=I_.sum()))
            I_ = y_ < lb or y_ > ub
            counter_ += 1
    elif np.isclose(mu_, 0.):
        while (I_.any()) and (counter_ <= 20):
            y_[I_] = x_ [I_] + (np.random.normal(mu_, sigma_, size=I_.sum()))
            I_ = y_ < lb or y_ > ub
            counter_ += 1
    else:
        raise ValueError('The mean provided should be 0 or 1 to determine the' +
                         'type of noise (additive or multiplicative)')
        
    if counter_ >= 20:
        y_ = np.maximum(np.minimum(y_, ub), lb)

    return(y_)


def generate_Reco_params(sp_map, sp_id, S_max, sp_pft, reco_, LAImax,
                         reshape2D=True):
    # Rlai0, alai, LAImax, k2, E0, GPP, Tref, T0, Tair, alpha_, k_, P
    vrs_ = [i_ for i_ in reco_.keys()]
    sz_ = int(len(vrs_[1:])/2)
    reco_param = np.zeros((sp_map.shape[0], sp_map.shape[1], sz_))
    i_, id_ = 0, 1
    # First generate interspecific variability based on the std provided by
    # Migliavacca et al. 2011
    i_, pf_ = 0, sp_pft[0]
    for pf_ in np.unique(sp_pft):
        J_ = reco_['pft_in'].index(pf_)
        Ip_ = np.where(sp_pft == pf_)[0]
        # Get the species mean values for the PFT
        sp_mean = np.zeros((sz_, Ip_.shape[0]))
        sd_inter_sp = np.zeros((sz_))
        for j_, k_ in enumerate(vrs_[1:]):
            if k_ != 'LAI_max' and ('_sd' not in k_):
                mu_ = reco_[k_][J_]
                sd_inter_sp[j_] = copy.deepcopy(reco_[k_ + '_sd'][J_])
                # sd_inter_sp[j_] = np.abs(sd_ / mu_)
                # Since only Rlai 0 seems to present negative values in the
                # manuscript, distribitions are truncated for the rest
                if k_ == 'Rlai0':
                    sp_mean[j_, :] = (mu_ + (
                        np.random.normal(0., sd_inter_sp[j_],
                                         size=Ip_.shape[0])))
                elif k_ == 'alpha':
                    sp_mean[j_, :] = truncate_gauss_noise(
                        mu_, sd_inter_sp[j_], mu_= 0., lb=.05, ub=.95)
                else:
                    sp_mean[j_, :] = np.abs(mu_ * (
                        np.random.normal(1., sd_inter_sp[j_],
                                         size=Ip_.shape[0])))
        
        # Assign values to the species including intraspecific variability
        # according to Albert et al. (2010)
        for c_, i_ in enumerate(Ip_):
            I_ = sp_map == sp_id[i_]
            # Assign the PFT value and add intraspecific variability
            for j_, k_ in enumerate(vrs_[1:]):
                if k_ != 'LAI_max' and ('_sd' not in k_):
                    cv_intra_sp = np.random.uniform(.2, .4) * sd_inter_sp[j_]
                    # Since only Rlai 0 seems to present negative values in the
                    # manuscript, distribitions are truncated for the rest
                    if k_ == 'Rlai0':
                        reco_param[I_, j_] = (sp_mean[j_, c_] + (
                            np.random.normal(0., cv_intra_sp, size=I_.sum())))
                    elif k_ == 'alpha':
                        reco_param[I_, j_] = truncate_gauss_noise(
                            sp_mean[j_, c_], cv_intra_sp, mu_= 0., lb=.05,
                            ub=.95)
                    else:
                        reco_param[I_, j_] = np.abs(
                            (sp_mean[j_, c_] + (np.random.normal(
                                0., cv_intra_sp, size=I_.sum()))))

    # Add species LAI max already generated outside
    reco_param[:, :, vrs_[1:].index('LAI_max')] = LAImax

    if reshape2D:
        reco_param = reco_param.reshape(-1, reco_param.shape[2])

    return (reco_param)