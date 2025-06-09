import os
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import time as time

from BOSSE import scsim_scene as sc
from BOSSE import scsim_species as sp
from BOSSE import scsim_rtm as rtm
from BOSSE.helpers import print_et
import BOSSE.plotter as pl


# %% Class BOSSE MODEL
class BosseModel:
    # Class Initialization -----------------------------------------------------
    def __init__(self, inputs_, paths_, RaoQ_w=3):
        self.inputs_ = inputs_
        self.paths_ = paths_
        
        # Define some variables with names easier to handle
        self.scsz_ = inputs_['scene_sz']
        
        # Set RaoQ window size
        self.RaoQ_w = RaoQ_w
        self.RaoQ_len = int(self.scsz_ / self.RaoQ_w)
        self.RaoQ_sz = self.RaoQ_len ** 2

        # Load constants and generate the scene matrix (X_)
        (self.X0_, self.all_vars, self.all_vals, self.PT_vars, self.PT_LB,
         self.PT_UB, self.Soil_vars, self.Soil_LB, self.Soil_UB, self.I_cab,
         self.I_cs, self.I_lai, self.I_hc, self.I_vcmo, self.I_m, self.I_pt,
         self.I_gmm, self.I_rnd, self.I_sf, self.I_soil, self.I_smc, self.I_fc,
         self.I_met, self.I_rss, self.I_rssl, self.I_ang, self.I_rin
         ) = sp.set_sim_vars(self.scsz_)

        # Load models
        self.get_bosse_models(self.paths_, self.all_vars,
                              self.inputs_['clim_zone'])

    # Method to load models
    def get_bosse_models(self, paths_, all_vars, clim_zone, sensor_='_Hy',
                         no_crop=True):
        # Get PFTs of the climatic zone
        self.P_pft = pd.read_csv((paths_['1_dest_PFTdist_folder'] + clim_zone +
                             '_freq.csv'), sep=';')
        if no_crop:
            self.P_pft = self.P_pft[(self.P_pft.iloc[:, 0].values != 'CROC3') &
                                    (self.P_pft.iloc[:, 0].values != 'CROC4')]

        # Load veg constants and filter PFTs
        self.veg_ = sp.set_plants(self.P_pft, clim_zone=clim_zone)

        # Load ecosystem respiration parameters and filter PFTs
        self.reco_ = sp.set_reco(self.P_pft, clim_zone=clim_zone)

        # GMM foliar traits
        self.GM_T = joblib.load(paths_['1_dest_GMMtraits_file_joblib'])
        with open(paths_['1_dest_GMMtraits_file_pkl'], 'rb') as f:
            [self.GM_T['trans_'], self.GM_T['lambda_']] = pickle.load(f)

        # Reflectance Emulator
        # Here use the 2-layers NN with trainned with extra Soil LUT data
        self.M_R = joblib.load(
            paths_['1_dest_NNR_file_joblib'].replace(
                'NNR_nlyr1_Hy', 'NNR_nlyr2_Hy').replace(
                    '.joblib', '_Slut_LAI020.joblib'))
        self.M_R['I_'] = [all_vars.index(i_) for i_ in self.M_R['feat_sel']]
        self.M_R['nI'] = len(self.M_R['I_'])
        # print(self.M_R['feat_sel'])

        # Fluorescence Emulator
        self.M_F = joblib.load(paths_['1_dest_NNF_file_joblib'].replace(
            'NNF_nlyr1', 'NNF_nlyr2').replace(
            '.joblib', '_Slut_LAI020.joblib'))
        self.M_F['I_'] = [all_vars.index(i_) for i_ in self.M_F['feat_sel']]
        self.M_F['nI'] = len(self.M_F['I_'])
        # print(self.M_F['feat_sel'])
        
        # LST Emulator
        self.M_T = joblib.load(paths_['1_dest_NNT_file_joblib'].replace(
            'NNT_nlyr1', 'NNT_nlyr2').replace(
                '.joblib', '_Slut_LAI020.joblib'))
        self.M_T['I_'] = [all_vars.index(i_) for i_ in self.M_T['feat_sel']]
        self.M_T['nI'] = len(self.M_T['I_'])
        # print(self.M_T['feat_sel'])

        # Ecosystem Functions Emulator
        self.M_eF = joblib.load(paths_['1_dest_NNeF_file_joblib'].replace(
            'NNeF_nlyr1', 'NNeF_nlyr2').replace(
                '.joblib', '_Slut_LAI020.joblib'))
        self.M_eF['I_'] = [all_vars.index(i_) for i_ in self.M_eF['feat_sel']]
        self.M_eF['nI'] = len(self.M_eF['I_'])
        # print(self.M_eF['feat_sel'])

        # Reflectance inversion emulator
        self.M_Rinv = joblib.load(paths_['1_dest_NNRinv_file_joblib'].replace(
            'NNRinv_nlyr1_Hy', 'NNRinv_nlyr2%s' % sensor_).replace(
                '.joblib', '_Slut_LAI020.joblib'))
        self.M_Rinv['pred_vars'] = [i_.replace('$', '') for i_ in
                                    self.M_Rinv['input_MPL']['pred_vars']]
        self.M_Rinv['Ip_'] = [all_vars.index(i_) for i_ in
                              self.M_Rinv['pred_vars']]
        self.M_Rinv['I_'] = range(len(self.M_Rinv['feat_sel']))
        self.M_Rinv['nI'] = len(self.M_Rinv['I_'])
        # print(M_Rinv['pred_vars'])

        # rss model (from SMC and level)
        mat_ = loadmat(paths_['1_ori_Irss_file'])
        self.M_rss = RegularGridInterpolator((mat_['X'][0], mat_['Y'][:, 0]),
                                             mat_['Z'].T)        
        # Plot the model
        plt_fname = paths_['1_ori_Irss_file'].replace('.mat', '.png')
        if (os.path.isfile(plt_fname) is False):
            self.plot_rss_model(mat_, cmap='viridis', plt_show=False,
                                fname=plt_fname)        

    # Scene Initializaton ------------------------------------------------------
    # Method prepare the Scence
    def initialize_scene(self, simnum_, seednum_=None, minLAImax=1.,
                         verbose=None, inspect=None, local_pft_lim=None):
        # Update verbose and inpsect options if necessary
        if verbose != None:
            self.inputs_['verbose'] = verbose
        if inspect != None:
            self.inputs_['inspect'] = inspect
        
        if self.inputs_['verbose']:
            print('#' * 72)
            print(f'Initializing scene {simnum_} ...')
            t0 = time.time()
   
        (self.meteo_, self.meteo_av, self.meteo_av30, self.meteo_mdy,
         self.meta_met, self.ts_length, self.ts_days, self.indx_day,
         self.indx_mdy, self.X_, self.sp_map, self.sp_pft, self.S_max,
         self.sp_ab, self.sp_id, self.sp_pft, self.PT_map_min, self.PT_map_max,
         self.PT_map_delta, self.num_dis, self.reco_P, self.GSI_all,
         self.GSI_wav, self.GSI_wav_param, self.GSI_rin, self.GSI_rin_param,
         self.GSI_tcol, self.GSI_tcol_param, self.GSI_twrm, self.GSI_twrm_param,
         self.PT_mean, self.PT_min, self.PT_max, self.local_av, self.local_LB,
         self.local_UB, self.coulds_map) = (
             sc.create_scene_data(
                 self.paths_, self.inputs_, simnum_, seednum_, self.scsz_,
                 self.X0_, self.all_vars, self.PT_vars, self.PT_LB, self.PT_UB,
                 self.Soil_vars, self.Soil_LB, self.Soil_UB, self.I_cab,
                 self.I_cs, self.I_lai, self.I_hc, self.I_vcmo, self.I_m,
                 self.I_pt, self.I_gmm, self.I_rnd, self.I_sf, self.I_soil,
                 self.I_smc, self.I_fc, self.I_met, self.I_rss, self.I_rssl,
                 self.I_ang, self.veg_, self.P_pft, self.GM_T, self.reco_,
                 self.M_rss, minLAImax=minLAImax, local_pft_lim=local_pft_lim))
         
        if self.inputs_['verbose']:
            print_et('Total ', time.time() - t0)
    
    # Scene Simulation ---------------------------------------------------------
    # Method to retrieve the possible input option values or formats
    def get_input_descriptors(self, key):
        self.input_info = {
            'rseed_num': 'integer',
            'subfolder_out': 'string',
            'scene_sz': 'integer',
            'S_max': 'integer',
            'sensor': ['Hy', 'EnMAP', 'DESIS', 'S2'],
            'spat_res': ['integer between 1 and 100'],
            'inspect': [True, False],
            'sp_pattern': ['clustered', 'intermediate', 'even'],
            'clim_zone': ['Continental', 'Tropical', 'Dry', 'Temperate']}
        
        return(self.input_info[key])
    
    # Method to set scene at a given timestamp
    def pred_scene_timestamp(self, t_, meteo_input, indx_input, lai020_pt=True,
                             lai020_ang=False, lai020_met=False, sp_res=100):
        X_, PT_mean, PT_min, PT_max = sc.update_scene_timestamp(
            t_, getattr(self, indx_input), self.PT_map_min, self.PT_map_max,
            self.PT_map_delta, self.sp_map, self.sp_id, self.GSI_all,
            self.GSI_wav, self.X_, self.PT_mean, self.PT_min, self.PT_max,
            getattr(self, meteo_input), self.M_rss, self.Soil_UB, self.I_pt,
            self.I_sf, self.I_smc, self.I_fc, self.I_rss, self.I_rssl,
            self.I_lai, self.I_ang, self.I_met, self.all_vars,
            lai020_pt=lai020_pt, lai020_ang=lai020_ang, lai020_met=lai020_met)
        
        if sp_res != 100:
            X_ = rtm.spatial_upscale(X_, sp_res, type_='Mean')
        
        return(X_)
    
    # Method to transform 3D matrix into 2D matrix
    def transf3D_2_2D(self, M3d):
        sz_ = M3d.shape
        M2d = M3d.reshape(sz_[0] * sz_[1], sz_[2])
        
        return(M2d)
        
    # Method to transform 2D matrix into 3D squared matrix
    def transf2D_2_3Dsq(self, M2d):
        sz_ = M2d.shape
        sz_sq = int(float(sz_[0]) ** .5)
        M3d = M2d.reshape(sz_sq, sz_sq, sz_[1])
        
        return(M3d)
    
    # Method to ensure that the inputs provided to the emulators generating
    # spectral signals get the inputs properly
    def check_emulator_inputs(self, M_):
        (all_vars, all_vals) = sp.generate_all_vars()
        for i_, iv_ in enumerate(M_['I_']):
            print(i_, iv_, all_vars[iv_], M_['feat_sel'][i_])
            if all_vars[iv_] != M_['feat_sel'][i_]:
                raise Warning(all_vars[iv_] + 'and' + M_['feat_sel'][i_] +
                      'do not match')
    
    # Generate relative random noise
    def add_random_noise(self, signal_, rel_rand_noise, abs_rand_noise,
                         rand_seed):
        # Seed for reproducibility
        if rand_seed != None:            
            rng = np.random.default_rng(seed=rand_seed)
        else:            
            rng = np.random.default_rng(seed=np.random.randint(1))
        
        # Determine whcih type of uncertainty is provided
        do_rel = np.any(rel_rand_noise > 0.)
        do_abs = np.all(abs_rand_noise > 0.)
        
        # Generate noise from a single or a spectrally variable value
        if do_rel and do_abs:
            raise ValueError(
                'either relative and absolute random noise must be 0.')

        # Generate noise from relative values
        elif do_rel:
            sz_ = signal_.shape
            if isinstance(rel_rand_noise, np.ndarray):
                rel_rand_noise = rel_rand_noise.reshape(1, 1, -1)
            elif isinstance(rel_rand_noise, list):
                rel_rand_noise = np.array(rel_rand_noise).reshape(1, 1, -1)
            elif( isinstance(rel_rand_noise, float) or
                 isinstance(rel_rand_noise, float)):
                rel_rand_noise = np.full_like(np.zeros((1, 1, sz_[2])),
                                              rel_rand_noise, dtype=float)
            mu_ = np.zeros(sz_)
            sigma_ = np.repeat(
                np.repeat(rel_rand_noise, sz_[0], axis=0),
                sz_[1], axis=1)
            
            noise_ = (rng.normal(mu_, sigma_, size=sz_) * signal_)
        # Generate noise from absolute values
        elif do_abs:
            sz_ = signal_.shape
            if isinstance(abs_rand_noise, np.ndarray):
                abs_rand_noise = abs_rand_noise.reshape(1, 1, -1)
            elif isinstance(abs_rand_noise, list):
                abs_rand_noise = np.array(abs_rand_noise).reshape(1, 1, -1)
            elif( isinstance(abs_rand_noise, float) or
                 isinstance(abs_rand_noise, float)):
                abs_rand_noise = np.full_like(np.zeros((1, 1, sz_[2])),
                                              abs_rand_noise, dtype=float)
            mu_ = np.zeros(sz_)
            sigma_ = np.repeat(
                np.repeat(abs_rand_noise, sz_[0], axis=0),
                sz_[1], axis=1)

            noise_ = (rng.normal(mu_, sigma_, size=sz_))
        
        else:
            noise_ = 0.

        return(signal_ + noise_)

    # Method to predict reflectance factors
    def pred_refl_factors(self, X_, scsz_, sp_res=100, rel_rand_noise=0.,
                          abs_rand_noise=0., rand_seed=None):
        RF_ = rtm.spectral_pred(
            self.M_R, X_[:, :, self.M_R['I_']].reshape(-1, self.M_R['nI']),
            check_input=False, out_shape=(scsz_, scsz_, len(self.M_R['wl'])),
            sp_res=sp_res)

        if np.any(rel_rand_noise > 0.) or np.any(abs_rand_noise > 0.):
            RF_ = self.add_random_noise(RF_, rel_rand_noise, abs_rand_noise,
                                        rand_seed)

        return(RF_)
    
    # Method to predict optical traits from reflectance factors
    def pred_opt_traits(self, RFx_, out_sz, rel_rand_noise=0., 
                        abs_rand_noise=0., rand_seed=None):
        if isinstance(out_sz, (list, tuple, np.ndarray)) and len(out_sz) == 2:
            OT_ = rtm.optical_trait_ret(
                self.M_Rinv, RFx_, out_shape=(out_sz[0], out_sz[1], -1),
                force_baresoil=True)
        else:
            OT_ = rtm.optical_trait_ret(
                self.M_Rinv, RFx_, out_shape=(out_sz, out_sz, -1),
                force_baresoil=True)
            
        if np.any(rel_rand_noise > 0.) or np.all(abs_rand_noise > 0.):
            OT_ = self.add_random_noise(OT_, rel_rand_noise, abs_rand_noise,
                                        rand_seed)
        
        return(OT_)
    
    # Method to predict sun-induced chlorophyll fluorescence
    def pred_fluorescence_rad(self, X_, scsz_, out_sz, sp_res=100,
                              rel_rand_noise=0.,  abs_rand_noise=0.,
                              rand_seed=None):
        if np.any(X_[:, :, self.I_rin] > 0):
            F_ = rtm.spectral_pred(
            self.M_F, X_[:, :, self.M_F['I_']].reshape(-1, self.M_F['nI']),
                check_input=False, out_shape=(scsz_, scsz_,len(self.M_F['wl'])),
                sp_res=sp_res)
                # Avoid fluorescence when there is no light or LAI
            F_[F_ < 0.] = 0.
        else: 
            F_ = np.zeros((out_sz, out_sz, len(self.M_F['wl'])))

        if np.any(rel_rand_noise > 0.) or np.all(abs_rand_noise > 0.):
            F_ = self.add_random_noise(F_, rel_rand_noise, abs_rand_noise,
                                        rand_seed)
        
        return(F_)
    
    # Method to predict land surface temperature
    def pred_landsurf_temp(self, X_, scsz_, sp_res=100, rel_rand_noise=0.,
                            abs_rand_noise=0., rand_seed=None):
        LST_ = rtm.spectral_pred(
            self.M_T, X_[:, :, self.M_T['I_']].reshape(-1, self.M_T['nI']),
            check_input=False, out_shape=(scsz_, scsz_, len(self.M_T['wl'])),
            sp_res=sp_res)

        if np.any(rel_rand_noise > 0.) or np.all(abs_rand_noise > 0.):
            LST_ = self.add_random_noise(LST_, rel_rand_noise, abs_rand_noise,
                                        rand_seed)
                  
        return(LST_)
    
    # Method to predict ecosystem functions
    def pred_ecosys_funct(self, t_, X_, meteo_input, rel_rand_noise=0.,
                            abs_rand_noise=0., rand_seed=None,
                            output_map=False):
        # Compute the fluxes
        efun = list(sc.update_ecosys_funct(
            t_, X_, self.all_vars, self.M_eF, self.reco_P,
            getattr(self, meteo_input),
            self.meteo_av30.loc[int(np.floor(t_ / 24)), 'tp'], self.I_met,
            output_map=output_map))
        
        # Add noise
        num_fun = len(efun)
        if np.any(rel_rand_noise > 0.) or  np.any(abs_rand_noise > 0.):
            if isinstance(rel_rand_noise, float):
                rel_rand_noise = np.ones(num_fun) * rel_rand_noise

            if isinstance(abs_rand_noise, float):
                abs_rand_noise = np.ones(num_fun) * abs_rand_noise
            
            if output_map:
                for i_, funct in enumerate(efun[1:]):
                    if (rel_rand_noise[i_] > 0.) or (abs_rand_noise[i_] > 0.):
                        if rand_seed != None:
                            rseed_i = rand_seed + 1500 * (i_ + 1)
                        else:
                            rseed_i = None
                             
                        efun[i_ + 1] = self.add_random_noise(
                            np.atleast_3d(funct), rel_rand_noise[i_ + 1],
                            abs_rand_noise[i_ + 1], rseed_i).squeeze()
            else:
                for i_, funct in enumerate(efun[1:]):
                    if (rel_rand_noise[i_] > 0.) or (abs_rand_noise[i_] > 0.):
                        if rand_seed != None:
                            rseed_i = rand_seed + 50 * (i_ + 1)
                        else:
                            rseed_i = None
                            
                        efun[i_ + 1] = self.add_random_noise(
                            np.atleast_3d(funct), rel_rand_noise[i_ + 1],
                            abs_rand_noise[i_ + 1], rseed_i).ravel()[0]
         
        (time_out, GPP, Rb, Rb_15C, NEP, LUE, LUEgreen, lE, T, H, Rn, G,
         ustar) = efun
        
        return(time_out, GPP, Rb, Rb_15C, NEP, LUE, LUEgreen, lE, T, H, Rn, G,
         ustar)
        

    # Methods to plot data -----------------------------------------------------
    # Methods to get variable's symbols and units, and label
    def get_variable_symbol(self, var_name, add_brackets=False, subscript=None):
        var_sym = pl.get_variable_symbol(var_name, add_brackets=add_brackets,
                                         subscript=subscript)
        
        return(var_sym)
        
    def get_variable_units(self, var_name):
        var_uds = pl.get_variable_units(var_name)
        
        return(var_uds)
        
    def get_variable_label(self, var_name, subscript=None):
        var_lbl = pl.get_variable_label(var_name, subscript=subscript)
        
        return(var_lbl)
        
    # Plot meteorological data time series
    def plot_meteo_ts(self, plt_show=False, fname_=None):
        pl.do_plot_meteo_ts(self.meteo_, plt_show=plt_show, fname_=fname_)
        
    # Plot the soil resistance to evaporation from the pore space model
    def plot_rss_model(self, mat_, cmap='viridis', plt_show=False, fname=None):
        pl.do_plot_rss_model(mat_, cmap=cmap, plt_show=plt_show, fname=fname)
    
    # Plot any BOSSE 2D Scene map
    def show_bosse_map(self, im_, title_lb='BOSSE simulation', xlb='x [pixel]',
                      ylb='y [pixel]', add_colorbar=True, cmap='viridis',
                      plt_show=False, fname=None, ):
        pl.do_show_bosse_map(im_, title_lb=title_lb, xlb=xlb, ylb=ylb,
                             add_colorbar=add_colorbar, cmap=cmap,
                             plt_show=plt_show, fname=fname)

    def show_pft_map(self, title_lb='BOSSE Plant Functional Types map',
                     xlb='x [pixel]', ylb='y [pixel]', fname=None,
                     plt_show=False):
        pl.do_plot_pft_map(self.sp_map, self.veg_, title_lb=title_lb,
                           xlb=xlb, ylb=ylb, add_colorbar=False,
                           plt_show=plt_show, fname=fname)

    def show_species_map(self, title_lb='BOSSE Species map',
                         xlb='x [pixel]', ylb='y [pixel]', add_colorbar=True,
                         cmap='tab20', fname=None, plt_show=False):
        pl.do_show_bosse_map(self.sp_map, title_lb=title_lb, xlb=xlb, ylb=ylb,
                           add_colorbar=add_colorbar, cmap=cmap,
                           plt_show=plt_show, fname=fname)
    
    # Plot the spectra of each pixel colored by species
    def plot_species_spectra(self, wvl, X_, ylbl, cmp_=None, plt_show=False,
                             fname=None):        
        pl.do_plot_species_spectra(self.sp_map, self.sp_id, wvl, X_, ylbl,
                                   cmp_=cmp_, plt_show=plt_show, fname=fname)
