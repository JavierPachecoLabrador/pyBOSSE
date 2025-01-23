# %% 0) Imports
import os
import copy
import pandas as pd
import numpy as np
from scipy.stats import norm
import sklearn.metrics as metrics
from scipy.stats import (zscore, spearmanr, pearsonr, linregress)


# %% 1) IO
def set_out_path(in_, path_outputs, subfolder_out, sp_pattern, pth_sensors='',
                 pth_inputs='', pth_meteo='', pth_models='',
                 create_out_folder=True):
    # Update input's structure
    in_['sp_pattern'] = sp_pattern
    in_['subfolder_out'] = (subfolder_out + '//' + sp_pattern + '//')
    
    # Generate simulation paths
    paths_out = set_bosse_paths(pth_out=path_outputs, sensor_='Hy',
                                pth_inputs=pth_inputs, pth_meteo=pth_meteo,
                                pth_sensors=pth_sensors, pth_models=pth_models,
                                sf_out=in_['subfolder_out'])
    
    # Create output folder
    if (create_out_folder and
        (os.path.isdir(paths_out['2_out_folder_plots']) is False)):
        os.makedirs(paths_out['2_out_folder_plots'])

    return(paths_out, in_)


def set_folder(path_folder):
    if os.path.isdir(path_folder) is False:
        # In case there are paralell jobs
        os.makedirs(path_folder, exist_ok=True)
    return(path_folder)


def clear_paths(paths_):
    for k_ in paths_.keys():
        paths_[k_] = paths_[k_].replace('////', '//').replace('///', '//')

    return(paths_)


def zone_snum(simnum_, kzone_):
    return('%s_sim%02d' % (kzone_[:3], simnum_))


def set_bosse_paths(pth_out='', sf_out='', sensor_='Hy', pth_inputs='',
                    pth_meteo='', pth_sensors='', pth_models='',
                    n_hidden_lyrs=2, mang_=False, sims_v=12):
    ori_ = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    paths_ = dict()
    # General paths -----------------------------------------------------------        
    paths_['0_root'] = ori_ + '//'
    paths_['0_bosse'] = ori_ + '//pyBOSSE//'
    paths_['0_outputs'] = pth_out + '//'

    # Ancillary inputs ---------------------------------------------------------
    if pth_inputs == '':
        paths_['0_inputs'] = paths_['0_bosse'] + 'BOSSE_inputs//'
    else:
        paths_['0_inputs'] = copy.deepcopy(pth_inputs)

    if pth_sensors == '':
        paths_['0_sensors'] = paths_['0_inputs'] + 'Sensors//'
    else:
        paths_['0_sensors'] = copy.deepcopy(pth_sensors)
    
    if pth_meteo == '':
        paths_['0_meteo'] = paths_['0_inputs'] + 'Meteo_ERA5Land//'
    else:
        paths_['0_meteo'] = copy.deepcopy(pth_meteo)
        
    # BOSSE Models --------------------------------------------------------------
    if pth_models == '':
        paths_['0_bosse_models'] = paths_['0_bosse'] + '//BOSSE_models//'
    else:
        paths_['0_bosse_models'] = copy.deepcopy(pth_models)
    
    paths_['1_simulations'] = (ori_.split('3_BOSSE')[0] + '3_Simulations//')
    paths_['1_simulations_sbfldr'] = ((paths_['1_simulations'] + '1_LUTs_V%d//')
                                      % sims_v)

    # GMM plant traits
    paths_['1_ori_GMMtraits_file'] = (paths_['1_simulations'] +
                                      'Trait_GMD//leaf_rtm_db.csv')
    paths_['1_dest_GMMtraits_folder'] = (paths_['0_root'] +
                                         'BOSSE_models//GMMtraits//')
    paths_['1_dest_GMMtraits_file_pkl'] = (paths_['1_dest_GMMtraits_folder'] +
                                           'GMMtraits.pkl')
    paths_['1_dest_GMMtraits_file_joblib'] = (
        paths_['1_dest_GMMtraits_folder'] + 'GMMtraits.joblib')
    
    # GMM PFTs
    paths_['1_dest_PFTdist_folder'] = (paths_['0_root'] +
                                       'BOSSE_models//PFTdist//')
    
    # R emulator. Inputs.
    paths_['1_ori_NNR_file_meta'] = ((
        paths_['1_simulations_sbfldr'] +
        '1_LUTsV%d_SCOPE_n10000_ERA5_EcoStr_1_' +
        'ltlhs_dT0_fix-fqe-tto-Gfrac-kV-Rdparam//' +
        'Dataset_PSI0_BSMsoil_10000.mat') % sims_v)
    paths_['1_ori_NNR_file_train'] = ((
        paths_['1_simulations_sbfldr'] +
        '1_LUTsV%d_SCOPE_n10000_ERA5_EcoStr_1_' +
        'ltlhs_dT0_fix-fqe-tto-Gfrac-kV-Rdparam//' +
        'Test_t_PSI0_BSMsoil_10000.mat') % sims_v)
    paths_['1_ori_NNR_file_test'] = ((
        paths_['1_simulations_sbfldr'] +
        '1_LUTsV%d_SCOPE_n10000_ERA5_EcoStr_1_' +
        'ltlhs_dT0_fix-fqe-tto-Gfrac-kV-Rdparam//' +
        'Test_v_PSI0_BSMsoil_10000.mat') % sims_v)
    
    # Configure whether data are multi-angular or not
    if mang_ == True:
        paths_['1_ori_NNR_file_meta'] = (
            paths_['1_ori_NNR_file_meta'].replace('-tto', ''))
        paths_['1_ori_NNR_file_train'] = (
            paths_['1_ori_NNR_file_train'].replace('-tto', ''))
        paths_['1_ori_NNR_file_test'] = (
            paths_['1_ori_NNR_file_test'].replace('-tto', ''))
        mglb = '_mang'
    else:
        mglb = ''
    
    # Models. Always for Hy
    paths_['1_dest_NNR_folder'] = (
        paths_['0_root'] + 'BOSSE_models//NNR_nlyr%d_Hy%s//' %
        (n_hidden_lyrs, mglb))
    paths_['1_dest_NNR_file_pkl'] = (paths_['1_dest_NNR_folder'] + 'NNR.pkl')
    paths_['1_dest_NNR_file_joblib'] = (paths_['1_dest_NNR_folder'] +
                                        'NNR.joblib')
    # F emulator. Inputs
    paths_['1_ori_NNF_file_meta'] = copy.deepcopy(
        paths_['1_ori_NNR_file_meta'])
    paths_['1_ori_NNF_file_train'] = copy.deepcopy(
        paths_['1_ori_NNR_file_train'])
    paths_['1_ori_NNF_file_test'] = copy.deepcopy(
        paths_['1_ori_NNR_file_test'])
    # F emulator. Models
    paths_['1_dest_NNF_folder'] = (
        paths_['0_root'] + 'BOSSE_models//NNF_nlyr%d%s//' %
        (n_hidden_lyrs, mglb))
    paths_['1_dest_NNF_file_pkl'] = (paths_['1_dest_NNF_folder'] + 'NNF.pkl')
    paths_['1_dest_NNF_file_joblib'] = (paths_['1_dest_NNF_folder'] +
                                        'NNF.joblib')
    
    # LST emulator. Inputs
    paths_['1_ori_NNT_file_meta'] = copy.deepcopy(
        paths_['1_ori_NNR_file_meta'])
    paths_['1_ori_NNT_file_train'] = copy.deepcopy(
        paths_['1_ori_NNR_file_train'])
    paths_['1_ori_NNT_file_test'] = copy.deepcopy(
        paths_['1_ori_NNR_file_test'])
    # LST emulator. Models
    paths_['1_dest_NNT_folder'] = (
        paths_['0_root'] + 'BOSSE_models//NNT_nlyr%d%s//' %
        (n_hidden_lyrs, mglb))
    paths_['1_dest_NNT_file_pkl'] = (paths_['1_dest_NNT_folder'] + 'NNT.pkl')
    paths_['1_dest_NNT_file_joblib'] = (paths_['1_dest_NNT_folder'] +
                                        'NNT.joblib')
    
    # Ecosystem functions emulator. Inputs
    paths_['1_ori_NNeF_file_meta'] = copy.deepcopy(
        paths_['1_ori_NNR_file_meta'])
    paths_['1_ori_NNeF_file_train'] = copy.deepcopy(
        paths_['1_ori_NNR_file_train'])
    paths_['1_ori_NNeF_file_test'] = copy.deepcopy(
        paths_['1_ori_NNR_file_test'])
    # Ecosystem functions emulator. Models
    paths_['1_dest_NNeF_folder'] = (
        paths_['0_root'] + 'BOSSE_models//NNeF_nlyr%d//' % (n_hidden_lyrs))
    paths_['1_dest_NNeF_file_pkl'] = (paths_['1_dest_NNeF_folder'] +
                                      'NNef.pkl')
    paths_['1_dest_NNeF_file_joblib'] = (paths_['1_dest_NNeF_folder'] +
                                         'NNef.joblib')

    # Rinv emulator. For different sensors. Here use a different LUT than for
    # the R emulator to make the retrieval more independent
    paths_['1_ori_NNRinv_file_meta'] = paths_['1_ori_NNR_file_meta'].replace(
        'EcoStr_1_', 'EcoStr_2_')        
    paths_['1_ori_NNRinv_file_train'] = paths_['1_ori_NNR_file_train'].replace(
        'EcoStr_1_', 'EcoStr_2_')
    paths_['1_ori_NNRinv_file_test'] = paths_['1_ori_NNR_file_test'].replace(
        'EcoStr_1_', 'EcoStr_2_')
    # Rinv emulator. Models, for different sensors
    paths_['1_dest_NNRinv_folder'] = (
        paths_['0_root'] + 'BOSSE_models//NNRinv_nlyr%d_%s%s//' %
        (n_hidden_lyrs, sensor_, mglb))
    paths_['1_dest_NNRinv_file_pkl'] = (paths_['1_dest_NNRinv_folder'] +
                                        'NNRinv.pkl')
    paths_['1_dest_NNRinv_file_joblib'] = (paths_['1_dest_NNRinv_folder'] +
                                        'NNRinv.joblib')
    
    # Soil LUTs
    paths_['1_ori_soilLUT_train_full'] = (paths_['1_ori_NNR_file_train'].replace(
        'ltlhs_dT0_fix', 'ltlhs_dT0_BS_fix')).replace('10000', '2000')
    paths_['1_ori_soilLUT_test_full'] = (paths_['1_ori_NNR_file_test'].replace(
        'ltlhs_dT0_fix', 'ltlhs_dT0_BS_fix')).replace('10000', '2000')
        

    # rss model (from SMp and level)
    paths_['1_ori_Irss_file'] = (paths_['0_root'] +
                                 '//BOSSE_models//Interp_rss//rss_smp_mod.mat')
    
    # Scene simulator ----------------------------------------------------------
    paths_['2_out_folder'] = paths_['0_outputs'] + sf_out + '//'
    paths_['2_out_folder_plots'] = paths_['2_out_folder'] + 'Plots//'

    # Remove excesive slashes
    paths_ = clear_paths(paths_)

    return(paths_)


def set_up_paths_and_inputs(options_, path_outputs, create_out_folder=True,
                            pth_inputs='', pth_meteo='', pth_sensors='',
                            pth_models=''):
    """_simulation_options_

    options:
        scene_sz: in, length, in pixels, of the simulated scene
        S_max: int, maximum number of species that can be generated.
                The selection is random, can be lower.
        sp_pattern: Spatial pattern to be used for the distribution of the
                species. Can be: ['clustered', 'intermediate', 'even']
        clim_zone: Climatic zone where meteorological data and plant functional
                types are selected. Can be: 'Continental', 'Tropical', 'Dry',
                'Temperate']
        'spat_res': int in [1, 100], ratio plant to pixel size. 100 means only 
                one plant or group of identical plants are represented per
                pixel. Lower values mean 100 % resolution is simulated, but
                the model outputs resolution is degraded, mixing several species
                in one pixel.
        sensor: sensor simulated or used to retrieve optical traits, can be
                ['Hy', 'EnMAP', 'DESIS', 'S2']
        rseed_num: int, seed for the randon mumber generators
        subfolder_out: subfolder where the outputs will be stored
        inspect: bool, if True, produce some plots when initialize the BOSSE 
                model class.
        verbose: bool, if True, print information regarding the simulations
    
    path_outputs: str, folder where the outputs will be stored
    path_bosse: str, path to the pyBOSSE folder, included

    """

    # Take the input arguments, if None or empty, take default values
    if (options_ == None) or (len(options_) == 0):
        inputs_ = {'rseed_num': 100,
                   'subfolder_out': 'bosse_outputs',
                   'scene_sz': 60,
                   'S_max': 40,
                   'sensor': '_Hy',                      
                   'spat_res': 100,               
                   'sp_pattern': 'intermediate',
                   'clim_zone': 'continental',
                   'inspect': False,
                   'verbose': False}
    else:
        inputs_ = {'rseed_num': int(options_[0]),
                   'subfolder_out': options_[1],
                   'scene_sz': int(options_[2]),
                   'S_max': int(options_[3]),
                   'sensor': ('_' + options_[4]),
                   'spat_res': float(options_[5]),
                   'sp_pattern': options_[6],
                   'clim_zone': options_[7],
                   'inspect': bool(int(options_[8])),
                   'verbose': bool(int(options_[9]))} 
    
    # Redefine inputs and set up the corresponding paths
    paths_, inputs_ = set_out_path(inputs_,
                                   path_outputs,
                                   inputs_['subfolder_out'],
                                   inputs_['sp_pattern'],
                                   create_out_folder=create_out_folder,
                                   pth_inputs=pth_inputs,
                                   pth_meteo=pth_meteo,
                                   pth_sensors=pth_sensors,
                                   pth_models=pth_models)


    # Check if the spatial resolution provides at least a 3 x 3 pixels scene
    if int(float(inputs_['scene_sz']) * float(inputs_['spat_res']) / 100.) < 3:
        raise ValueError((
            'The requested spatial resolution requested does not allow ' +
            'producing scenes of at least three pixels. Increase the spatial ' +
            'resolution or the scene size'
        ))

    return(inputs_, paths_)


def create_empty_csv(fname_):
    df = pd.DataFrame(list())
    df.to_csv(fname_)


def preallocate_in_cluster(fname_, location):
    if location == 'cluster':
        print('    ...preallocating %s' % fname_)
        create_empty_csv(fname_)


def missing_files(fnames_):
    exist_ = [os.path.isfile(i_) for i_ in fnames_]
    
    return(all(exist_) is False)


def missing_folders(folders_):
    exist_ = [os.path.isdir(i_) for i_ in folders_]
    
    return(all(exist_) is False)


def check_non_empty_file(file_path, min_size=5, unit='kb'):
    """ Check whether the file does not exists and if it larger than a given size. This
    avoid trying to open preallocated files"""
    out_bool = [False, False]
    if os.path.isfile(file_path):
        out_bool[0] = True
        file_size = os.path.getsize(file_path)
        exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
        if unit not in exponents_map:
            raise ValueError("'unit' must be ['bytes', 'kb', 'mb', 'gb']")
        else:
            size = file_size / 1024 ** exponents_map[unit]
        if size > min_size:
            out_bool[1] = True

    return(out_bool)


# %% 2) Operations
def div_zeros(a, b, out_=0.):
    if isinstance(a, float):
        if isinstance(b, float):
            a = np.array(a).reshape((1, 1))
        else:
            a = a * np.ones(b.shape)
    if isinstance(b, float):
        if isinstance(a, float):
            b = np.array(b).reshape((1, 1))
        else:
            b = b * np.ones(a.shape)

    if a.shape != b.shape:
        if (a.shape[0] == b.shape[0]) and (a.shape[1] != b.shape[1]):
            if a.shape[1] == 1:
                b = np.repeat(a, b.shape[1], axis=1)
            elif b.shape[1] == 1:
                b = np.repeat(b, a.shape[1], axis=1)
        if (a.shape[1] == b.shape[1]) and (a.shape[0] != b.shape[0]):
            if a.shape[0] == 1:
                a = np.repeat(a, b.shape[0], axis=0)
            elif b.shape[0] == 1:
                b = np.repeat(b, a.shape[0], axis=0)
            
    return(np.divide(a, b, out=out_ * np.ones_like(a), where=b != 0.))


def nanmean_emp(X):
    if (np.size(X) > 0) and np.any(~np.isnan(X)):
        Y = np.nanmean(X)
    else:
        Y = np.nan

    return(Y)


def nanmedian_emp(X):
    if (np.size(X) > 0) and np.any(~np.isnan(X)):
        Y = np.nanmedian(X)
    else:
        Y = np.nan

    return(Y)


def get_predobs_stats(pred_in, obs_in, get_lm=False):
    """
    pred_ in the x axis, obs_ in the y axis
    """
    # Preallocate
    if get_lm is True:
        stats = ['R2', 'R2_adj', 'r_pearson', 'pval_pearson', 'r_spearman',
                'pval_spearman', 'RMSE', 'MAE', 'MSE', 'RRMSE', 'NRMSE',
                'slope', 'intercept']
    else:
        stats = ['R2', 'R2_adj', 'r_pearson', 'pval_pearson', 'r_spearman',
                'pval_spearman', 'RMSE', 'MAE', 'MSE', 'RRMSE', 'NRMSE']
    vals = np.zeros(len(stats)) * np.nan
    
    if ((pred_in is not None) and (obs_in is not None)):
        # Keep only finite data
        I_ = np.isfinite(pred_in + obs_in)
        pred_ = pred_in[I_]
        obs_ = obs_in[I_]

        try:
            n_ = pred_.shape[0]
        except:
            n_ = len(pred_)
            
        # Compute statistics
        if n_ > 0:           
            p_ = 1.
            nup_ = np.unique(pred_).shape[0]
            nuo_ = np.unique(obs_).shape[0]
            
            # Compute the errors, that can be got even with one datapoint
            vals[stats.index('RMSE')] = np.sqrt(
                metrics.mean_squared_error(obs_, pred_))
            vals[stats.index('MAE')] = metrics.mean_absolute_error(obs_, pred_)
            vals[stats.index('MSE')] = metrics.mean_squared_error(obs_, pred_)
            vals[stats.index('RRMSE')] = (100 *
                                          div_zeros(vals[stats.index('RMSE')],
                                                    np.nanmean(pred_)))

            # Get the scores and correlation coefficients if there are more
            # than 2 datapoints 
            if n_ > 2:
                vals[stats.index('R2')] = metrics.r2_score(obs_, pred_)
                vals[stats.index('R2_adj')] = (
                    1 - (1 -vals[stats.index('R2')]) *
                    ((n_ - 1.) / (n_ - p_ - 1)))
                
                if (nup_ > 1) and (nuo_ > 1):
                    (vals[stats.index('r_pearson')],
                    vals[stats.index('pval_pearson')]) = pearsonr(obs_, pred_)
                    (vals[stats.index('r_spearman')],
                    vals[stats.index('pval_spearman')]) = spearmanr(obs_, pred_)

                Iz_ = (np.abs(zscore(obs_)) < 3)
                if Iz_.sum() > 1:
                    range_y = np.abs(np.max(obs_[Iz_]) - np.min(obs_[Iz_]))
                    vals[stats.index('NRMSE')] = (
                        100 * np.sqrt(div_zeros(vals[stats.index('MSE')],
                                                np.abs(range_y))))
            
            # Get the model, if requested and enough data
            if (get_lm is True) and ((nup_ > 1) and (nuo_ > 1)):
                try:
                    lm_ = linregress(pred_, obs_)
                    vals[stats.index('slope')] = lm_.slope
                    vals[stats.index('intercept')] = lm_.intercept
                except:
                    pass

    return(vals, stats)


def rmv_nan(var):
    if isinstance(var, pd.DataFrame):
        I_ = np.where(np.isnan(var.values) == False)[0]
    else:
        I_ = np.where(np.isnan(var) == False)[0]
        
    return(var[I_])


# %% 3) Messaging
def print_et(string_, delta_time, add2statement=False):
    if delta_time < 60.:
        str_delta = 'elapsed time %.2f seconds.' % delta_time
    elif delta_time <= 3600.:
        min_ = np.floor(delta_time / 60.)
        sec_ = delta_time - 60. * min_
        str_delta = 'elapsed time %d min, %.2f s.' % (min_, sec_)
    elif delta_time > 3600.:
        hr_ = np.floor(delta_time / 3600.)
        tmp_ = delta_time - 3600. * hr_
        min_ = np.floor(tmp_ / 60.)
        sec_ = tmp_ - 60. * min_
        str_delta = 'elapsed time %d h, %d min, %.2f s.' % (hr_, min_,
                                                                sec_)
    if add2statement:
        str_delta = '. E' + str_delta[1:]
        
    print(string_ + str_delta)
    

def print_dict(dict_, dict_name):
    print(dict_name)
    for key, value in zip(dict_.keys(), dict_.values()):
        print(f"\t{key}: {value}")    


# %% 4) Sensors
def remove_WV_bands(R_, wl):
    # Bands selected from https://doi.org/10.3390/rs70912009
    I_ = np.where((wl < 1330) | ((wl > 1490) & (wl < 1770)) | (wl > 1990))[0]
    if wl.shape[0] == R_.shape[1]:
        R_ = R_[:, I_]
    elif wl.shape[0] == R_.shape[0]:
        R_ = R_[I_]
    wl = wl[I_]
    
    return(R_, wl)


def get_RSensor_bands(sensor_, path_sens, rmWVb=False):
    """
    Generate the spectral bands of different sensors
    """
    if (sensor_ == 'Hy'):
        RSensor_bands = []
        wl = np.r_[400:2401]

    elif sensor_ == 'S2':
        df = pd.read_csv(path_sens + 'Sentinel2A_MSI_SRF.txt', sep="\t")
        df_ = df.loc[(df.SR_WL >= 400) & (df.SR_WL <= 2400)]
        # Remove 60 m bands (https://earth.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial):
        for b in ['SR_AV_B1','SR_AV_B9','SR_AV_B10']:
            df_.__delitem__(b)
        RSensor_bands = (df_.to_numpy())
        RSensor_bands = RSensor_bands[:, 1:]
        # Get S2 selected wavelengths
        wl = np.array([490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190])

    elif sensor_ == 'DESIS':
        df = np.loadtxt(path_sens + 'DESIS_SpectralFeat.txt')
        RSensor_bands = np.zeros((2001, df.shape[0]))
        wl0 = np.arange(400, 2401)
        wl = df[:, 0]
        for i_ in range(df.shape[0]):
            RSensor_bands[:, i_] = norm.pdf(
                wl0, loc=df[i_, 0], scale=df[i_, 1])
        #     plt.plot(wl0, RSensor_bands[:, i_])
        # plt.show()
    elif sensor_ == 'EnMAP':
        # https://www.enmap.org/data/doc/EnMAP_Spectral_Bands_update.xlsx
        # Mixed VNIR and SWIR, using SWIR above 900 nm as reported in
        # EnMAP specifications document
        df = pd.read_csv(path_sens + 'EnMAP_SpectralFeatures.txt', sep=";")
        RSensor_bands = np.zeros((2001, df.shape[0]))
        wl0 = np.arange(400, 2401)
        wl = df['CW'].values
        for i_ in range(df.shape[0]):
            RSensor_bands[:, i_] = norm.pdf(
                wl0, loc=df.iloc[i_, 0], scale=df.iloc[i_, 1])
        #     plt.plot(wl0, RSensor_bands[:, i_])
        # plt.show()

    elif sensor_ == 'QB2':
        df = pd.read_csv(path_sens + 'QuickBird2_SRF.txt', sep="\t")
        df_ = df.loc[(df.SR_WL >= 400) & (df.SR_WL <= 2400)]
        RSensor_bands = (df_.to_numpy())
        RSensor_bands = RSensor_bands[:, 1:] / np.sum(RSensor_bands[:, 1:], axis=0)
        wl = df_.SR_WL.to_numpy() @ RSensor_bands

    # Makes sure that the sensor is normalized by the band integral
    if (sensor_ != 'Hy'):
        SRFint_ = np.sum(RSensor_bands, axis=0).reshape(1, -1)
        RSensor_bands = (RSensor_bands / np.repeat(SRFint_, [2001], axis=0))
        # Check:  (.5*np.ones((2,2001))) @ RSensor_bands

    # Remove water vapour sensor bands
    if rmWVb is True:
        RSensor_bands, wl = remove_WV_bands(RSensor_bands, wl)

    return(RSensor_bands, wl)


def convolve_sensor(sensor_, path_sens, R_hy, rmWVb=False):
    if sensor_ != 'Hy':
        SRF, wl_sen = get_RSensor_bands(sensor_, path_sens, rmWVb=rmWVb)
        R_sen = R_hy @ SRF
        return(R_sen, wl_sen)
    else:
        wl_sen = np.arange(400, 2401)
        return(R_hy, wl_sen)
    

# 5) Variable labels
def var2label(vr_):
    chk = vr_.split('_')
    if len(chk) == 1:
        lbl = r'$%s$' % vr_
    elif len(chk) == 2:
         lbl = r'$%s_{\rm %s}$' % (chk[0], chk[1])  
    elif len(chk) == 3:
         lbl = r'$%s_{\rm %s, %s}$' % (chk[0], chk[1], chk[2])
    elif len(chk) == 4:
         lbl = r'$%s_{\rm %s, %s, %s}$' % (chk[0], chk[1], chk[2], chk[3])
    elif len(chk) == 5:
         lbl = r'$%s_{\rm %s, %s, %s, %s}$' % (chk[0], chk[1], chk[2],
                                               chk[3], chk[4])
    return(lbl)
