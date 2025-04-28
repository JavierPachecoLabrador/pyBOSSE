# %% 0) Imports
import copy

import numpy as np
import pandas as pd
import xarray as xr
import math
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import time as time
import datetime
import matplotlib.pyplot as plt
import pyet


# %% 1) Functions    
def plot_pyet_input(vr_, I_, Inan_):
    plt.clf()
    plt.plot(I_, vr_[I_], '.')
    plt.plot(Inan_, Inan_ * 0, '.')


def daily_average(meteo_, day_, uday_, pet_df):
    Int_ = [i_ for i_, c_ in enumerate(meteo_.columns) if c_
            not in ['time', 'date', 'tp']]
    
    # For precipitation, perfomr the daily accumulated rain
    Itp_ = list(meteo_.keys()).index('tp')
    
    meteo_av = pd.DataFrame(data=np.zeros((len(uday_), meteo_.shape[1])),
                            columns=meteo_.columns)
    meteo_av['time'] = meteo_av['time'].astype(meteo_['time'].dtype)
    meteo_av['DoY'] = meteo_av['DoY'].astype(int)

    for i_, ud_ in enumerate(uday_):
        I_ = day_ == ud_
        Imd_ = I_ & (meteo_['hour'] == 12)
        meteo_.loc[I_, 'PET'] = pet_df[i_]
        meteo_av.iloc[i_, Int_] = np.nanmean(
            meteo_.iloc[I_, Int_].values, axis=0)
        # meteo_av.iloc[i_, Is_] = np.max(meteo_.iloc[I_, Is_].values, axis=0)
        meteo_av.iloc[i_, Itp_] = np.nansum(meteo_.iloc[I_, Itp_].values)
        meteo_av.loc[i_, 'time'] = meteo_.loc[Imd_, 'time'].values
    
    return(meteo_, meteo_av)


def monthly_average(meteo_av, uday_):
    Int_ = [i_ for i_, c_ in enumerate(meteo_av.columns) if c_
        not in ['time', 'DoY', 'hour']]
    meteo_av30 = pd.DataFrame(data=np.zeros((len(uday_), meteo_av.shape[1])),
                            columns=meteo_av.columns)
    meteo_av30['time'] = copy.deepcopy(meteo_av30['time'])
    meteo_av30['DoY'] = copy.deepcopy(meteo_av30['DoY'])
    meteo_av30['hour'] = copy.deepcopy(meteo_av30['hour'])
    
    len_uday = len(uday_) - 1
    for i_ in range(len(uday_)):
        I_ = i_ + np.arange(-15, 16)
        I_[I_ < 0] = I_[I_ < 0] + 15
        I_[I_ > len_uday] = I_[I_ > len_uday] - 15
        meteo_av30.iloc[i_, Int_] = (np.mean(meteo_av.iloc[I_, Int_].values,
                                            axis=0))
    return(meteo_av30)


def get_pet_metav(meteo_, yr_num, lat=45., elevation=300):
    # Calculate PET using pyet
    day_ = yr_num * 365 + meteo_['DoY'].values
    uday_ = np.unique(day_).astype(int)

    # Get timestamps, needed for pyet
    time_tmp = [0] * meteo_.shape[0]
    for i_ in range(meteo_.shape[0]):
        time_tmp[i_] = (datetime.datetime(int(meteo_.loc[i_, 'Yr']), 1, 1) +
                        datetime.timedelta(int(meteo_.loc[i_, 'DoY']) - 1)
                        ).replace(hour=int(meteo_.loc[i_, 'hour']))
    # Preallocate
    meteo_tmp = pd.DataFrame(
        data=np.zeros((len(uday_), 10)) * np.nan,
        columns=['tmean', 'tmin', 'tmax', 'rh', 'rhmin', 'rhmax', 'wind',
                 'rs', 'ea', 'pressure'])

    # Get daily input variables
    for i_, ud_ in enumerate(uday_):
        I_ = day_ == ud_
        Iw_ = np.where(day_ == ud_)[0][0]

        if any(I_):
            meteo_tmp.iloc[i_] = (
                meteo_.loc[I_, 'Ta'].values.mean(),
                meteo_.loc[I_, 'Ta'].values.max(),
                meteo_.loc[I_, 'Ta'].values.min(),
                meteo_.loc[I_, 'rH'].values.mean(),
                meteo_.loc[I_, 'rH'].values.min(),
                meteo_.loc[I_, 'rH'].values.max(),
                meteo_.loc[I_, 'u'].values.mean(),
                np.sum((meteo_.loc[I_, 'Rin'].values +
                        meteo_.loc[I_, 'Rli'].values)) * (3600. * 1E-6),
                meteo_.loc[I_, 'ea'].values.mean() / 10.,
                meteo_.loc[I_, 'p'].values.mean() / 10.)

    # Set datetime index    
    meteo_tmp['Datetime'] = [time_tmp[i_] for i_ in range(meteo_.shape[0])
                             if np.isclose(meteo_.loc[i_, 'hour'], 12)]
    meteo_tmp = meteo_tmp.set_index('Datetime')

    # Extract the variables and compute evapotranspiration according to
    # Penman-Monteith 1965 (John L Monteith. Evaporation and environment.
    # In Symposia of the society for experimental biology, volume 19,
    # 205–234. Cambridge University Press (CUP) Cambridge, 1965.)
    pet_df = pyet.pm(tmean=meteo_tmp['tmean'], tmin=meteo_tmp['tmin'],
                     tmax=meteo_tmp['tmax'], rh=meteo_tmp['rh'],
                     rhmin=meteo_tmp['rhmin'], rhmax=meteo_tmp['rhmax'],
                     wind=meteo_tmp['wind'], rs=meteo_tmp['rs'],
                     pressure=meteo_tmp['pressure'], ea=meteo_tmp['ea'], 
                     elevation=elevation, lat=math.radians(lat)).values

    # Remove NaN in PET
    if np.any(np.isnan(pet_df)):
        Inan_ =  np.where(np.isnan(pet_df))[0]
        I_ =  np.where(~np.isnan(pet_df))[0]
        
        mod_ = LinearRegression().fit(meteo_tmp.iloc[I_, :].values,
                                      pet_df[I_].reshape(-1, 1))
        pet_df[Inan_] = mod_.predict(meteo_tmp.iloc[Inan_, :].values
                                     ).reshape(-1)

    # Assign PET and compute daily averages or accumulations
    meteo_['PET'] = 0.
    meteo_, meteo_av = daily_average(meteo_, day_, uday_, pet_df)

    # Compute monthly daily averages
    meteo_av30 = monthly_average(meteo_av, uday_)
        
    return(meteo_, meteo_av, meteo_av30)


def interp_lt_thr(x_, y_, th_=0., rp_val=0.):
    shp_ = y_.shape
    I_ = y_ < th_
    if np.any(I_):
        if len(shp_) < 2:
            y_ = y_.reshape(-1, 1)
        # Interpolate
        for i_ in range(y_.shape[1]):
            Ii_ = y_[:, i_] < th_
            if np.any(Ii_):
                intrp_ = interp1d(x_[~Ii_], y_[~Ii_, i_],
                                  fill_value='extrapolate')
                y_[Ii_, i_] = intrp_(x_[Ii_])
        # If there are still values, below the threshold, assign output value
        y_[y_ < th_] = rp_val
        if len(shp_) < 2:
            y_ = y_.reshape(shp_)
    return (y_)


def get_vpd(meteo_):
    T_ = meteo_['Ta'].values
    es = 6.107 * 10 * np.exp(7.5 * T_ / (237.3 + T_))
    vpd = es - meteo_['ea'].values
    return (vpd, es)


def get_meteo(paths_, FC_, KZone='Continental', site_num=0, minLAImax=1.,
              SMC_b=None):
    # Load meteo nc file
    meteo_, meta_met, lai_max = get_site_meteo(
        paths_, KZone=KZone, site_num=site_num, minLAImax=minLAImax)
    
    # Truncate SMC. Check wether SMC comes as a percentaje.
    # The emulators use it as a fraction of SMC; however, it should be
    # kept as percentaje in the meteo structure    
    if np.nanmean(meteo_['SMC']) <= 1.:
        meteo_['SMC'] = 100 * meteo_['SMC']
    if SMC_b is not None:
        meteo_.loc[meteo_['SMC'] < SMC_b[0], 'SMC'] = SMC_b[0]
        meteo_.loc[meteo_['SMC'] > SMC_b[1], 'SMC'] = SMC_b[1]
        
    # Compute relative soil moisture content
    # The relative soil moisture, which is defined as the ratio (in
    # percentage) of water in weight to the field capacity in weight of
    # the measured soil layer. Here used as a fraction. Will be updated later
    meteo_['wr'] = (meteo_['SMC'] / FC_)

    if np.any(np.isnan(meteo_.iloc[:, 1:].values.mean(axis=0))):
        print('\tNaN values in meteo data')
        
    # Replace Doy, Yr, and Hour; which correspondbto "local time"
    meteo_['Yr'] = meteo_['Yr'].astype(int)

    # For simplicity, remove leap years' extra days
    meteo_ = (meteo_.loc[~np.isclose(meteo_['DoY'], 366), :])
    meteo_ = meteo_.reset_index(drop=True)
    
    # Other time-related variables
    ts_days = (meteo_.groupby(['Yr', 'DoY']).size()).shape[0]
    yr_num = meteo_['Yr'].values - min(meteo_['Yr'].values)
    dec_doy = meteo_['DoY'].values + meteo_['hour'].values / 24.
    indx_day_raw = yr_num * 365 + dec_doy
    indx_day = indx_day_raw - min(indx_day_raw)
    
    # Get VPD
    meteo_['VPD'], es = get_vpd(meteo_)

    # Calculate PET if not provided and generate averaged meteo values
    meteo_, meteo_av, meteo_av30 = get_pet_metav(
        meteo_, yr_num, lat=meta_met['latitude'], elevation=300)

    # Keep night time for computing respiration
    meteo_['DayTime'] = (meteo_['Rin'] > 20.)
    ts_length = meteo_.shape[0]

    # Generate a meteo dataset with midday data for the GSI
    meteo_mdy = meteo_.loc[np.isclose(meteo_['hour'], 12), :]
    meteo_mdy = meteo_mdy.reset_index(drop=True)
    indx_mdy = np.arange(meteo_mdy.shape[0], dtype=int)
    if meteo_mdy.shape[0] != ts_days:
        raise ValueError('meteo: The number of days and middays do not match')

    return(meteo_, meteo_av, meteo_av30, meteo_mdy, meta_met, ts_length,
           ts_days, indx_day, indx_mdy, lai_max)


# %% 2) Produce/Load meteo netcdf datasets
def time_shift(lon):
    # Provide the time shift in hours respect to the UTC
    return(-int(lon/15))


def rad_acc2inst(Rin):
    #  Accumulates radiation daily and then uses the difference to get
    # hourly values. 
    # Converts from J m-2 h-1 to J m-2 s-1 = W m-2 dividing by 3600 s/h
    Rin = Rin / 3600
    for i_ in range(1, Rin.shape[0]-23, 24):
        Rin[i_ : i_+24] = Rin[i_ : i_+24] + Rin[i_-1]
    Rout = np.concatenate((np.zeros(1), Rin[1:] - Rin[:-1])) 
    return(Rout)


def preallocate_ds(time_, lat, lon):
    ds_ = xr.Dataset(
        data_vars=None,
        coords=dict(time = time_),
        attrs = dict(latitude = lat, longitude = lon,
                     description=("BOSSE meteo inputs. v1 stands for data already " +
                                  "available for ERA5-Land 0d10 or for ERA5 0d25 " +
                                  "(if not in Land) in the data_BGC folder")))
    return(ds_)


def satvap(T):
    es = 6.107 * (10**(7.5 * T / (237.3 + T)))
    return(es)


def correctLAImax(lmx):
    # Correct LAImax underestimation using the regression model found in Fig 3b 
    # of Li et al. 2022. https://doi.org/10.1016/j.srs.2022.100066
    lmx_corr = max(lmx, (lmx - .39) / .83)
    
    return(lmx_corr)


def load_meteo_nc(fname_nc):
    # Load file
    ds_ = xr.open_dataset(fname_nc)
    
    # Rename variables
    ds_ = ds_.rename_vars(name_dict={'t2m': 'Ta', 'sp': 'p'})
    
    # Transform variable units
    ds_['Ta'] = ds_['Ta'] - 273.15
    ds_['p'] = ds_['p'] / 100.
    ds_['tp'] = ds_['tp'] * 1000
    # Compute variables
    ds_['ea'] = satvap(ds_['Ta']) * ds_['rH']  / 100.
    ds_['u'] = (ds_['u10']**2 + ds_['v10']**2)**.5
    ds_['SMC'] = (ds_['swvl1']*.25 + ds_['swvl2']*.25 +
                    ds_['swvl3']*.25 + ds_['swvl4']*.25) * 100.

    # Correct LAImax underestimation
    lai_max = correctLAImax(
        np.max(ds_['lai_hv'].values + ds_['lai_lv'].values))
    
    # Remove negative precipitation data
    Ing_ = np.where(ds_['tp'].values < 0)[0]
    if np.any(Ing_):
        ds_['tp'][Ing_] = 0.
   
    # Drop variables
    try:
        ds_ = ds_.drop_vars(['u10', 'v10', 'swvl1',
                             'swvl2', 'swvl3', 'swvl4','lai_hv', 'lai_lv'])
    except:
        pass
    return(ds_, lai_max)


def get_site_meteo(paths_, KZone='Continental', site_num=0, minLAImax=1.):
    fname_nc = (paths_['0_meteo'] +
                'BOSSE_meteo_%s_%d.nc' % (KZone, site_num))
    ds_, lai_max = load_meteo_nc(fname_nc)
    if minLAImax != None:
        lai_max = max((lai_max, minLAImax))
        
    dsf_ = (ds_.to_dataframe()).reset_index()
    # Convert  float32 to float64
    for i_ in dsf_.keys():
       if dsf_[i_].dtype == np.dtype('float32'):
           dsf_[i_] = dsf_[i_].astype('float64')
   
    # Artificially expand the first and last day to make them 24h
    t_ = dsf_.loc[0, 'time'].hour
    dsf_before = dsf_.loc[24-t_:24-1].reset_index(drop=True)
    for i_ in range(dsf_before.shape[0]):
        dsf_before.loc[i_, 'time'] = dsf_before.loc[i_, 'time'].replace(
            day=dsf_.loc[0, 'time'].day)
    t_ = dsf_.loc[dsf_.shape[0]-1, 'time'].hour
    dsf_after = dsf_.loc[dsf_.shape[0]-24:
        dsf_.shape[0]-t_-2].reset_index(drop=True)
    for i_ in range(dsf_after.shape[0]):
        dsf_after.loc[i_, 'time'] = dsf_after.loc[i_, 'time'].replace(
            day=dsf_.loc[dsf_.shape[0]-1, 'time'].day)
    dsf_out = pd.concat([dsf_before, dsf_, dsf_after], ignore_index=True)

    # Correct times to local
    dsf_out['Yr'] = np.array([dsf_out.loc[i_, 'time'].year
                       for i_ in range(dsf_out.shape[0])]).astype(int)
    dsf_out['hour'] = [dsf_out.loc[i_, 'time'].hour
                       for i_ in range(dsf_out.shape[0])]
    dsf_out['DoY'] = [dsf_out.loc[i_, 'time'].day_of_year
                       for i_ in range(dsf_out.shape[0])]
    
    # Remove some negative values if happen. First interpolate, and if there are
    # negative values left, these are truncated. This avoids sharp changes
    indx_ = np.arange(dsf_out.shape[0])
    for v_ in ['p', 'tp', 'rH', 'Rin', 'Rli', 'ea', 'u', 'SMC']:
        if v_ != 'Rli':
            dsf_out[v_] = interp_lt_thr(indx_, dsf_out[v_].values)
        else:
            # Set a larger treshold for longwave incoming radiation
            dsf_out[v_] = interp_lt_thr(indx_, dsf_out[v_].values, th_=10,
                                        rp_val=50)
            
    # Round up some variables that lead to slight differences between machines
    dsf_out['ea'] = np.round(dsf_out['ea'], 5)

    return(dsf_out, ds_.attrs, lai_max)
