# %% 0) Imports
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps as mcmaps
from matplotlib import cm

# %% 1) Support functions
# Add subplot label
def add_subplot_label(ax, fx, fy, lbl):
    xl_ = ax.get_xlim()
    xrange = xl_[1] - xl_[0]
    yl_ = ax.get_ylim()
    yrange = yl_[1] - yl_[0]
    
    t_ = ax.text(xl_[0] + fx * xrange, yl_[0] + fy * yrange, lbl)
    
    return(t_)


# %% 2) Function to get BOSSE variables symbols and units
def get_var_dict():
    var_dict = {
        'N': ['$N$', '-'], 
        'Cab': ['$C_{\\rm ab}$', '$\\rm \mu mol cm^{-2}$'],
        'Cca': ['$C_{\\rm ca}$', '$\\rm \mu mol cm^{-2}$'], 
        'Cant': ['$C_{\\rm ant}$', '$\\rm \mu mol cm^{-2}$'], 
        'Cs': ['$C_{\\rm s}$', 'a.u.'], 
        'Cw': ['$C_{\\rm w}$', '$\\rm g cm^{-2}$'], 
        'Cdm': ['$C_{\\rm dm}$', '$\\rm g cm^{-2}$'],
        'LIDFa': ['LIDF$_{\\rm a}$', '-'],
        'LIDFb': ['LIDF$_{\\rm b}$', '-'], 
        'LAI': ['LAI', '$\\rm m^{2} m^{-2}$'],
        'hc': ['$h_{\\rm c}$', 'm'],
        'lw': ['$l_{\\rm w}$', 'm'],
        'Vcmo': ['$V_{\\rm cmo}$', '$\\rm \mu molC cm^{-2} s^{-1}$'],
        'm': ['$b_{\\rm BB}$', '-'],
        'kV': ['$k_{\\rm V}$', '-'],
        'Rdparam': ['$R_{\\rm d}$', '-'],
        'Type': ['Type', ''],
        'stressfactor': ['$f_{\\rm stress}$', '-'],
        'BSMBrightness': ['$B_{\\rm BSM}$', '-'],
        'BSMlat': ['$\phi_{\\rm BSM}$', '$\degree$'],
        'BSMlon': ['$\lambda_{\\rm BSM}$', '$\degree$'],
        'SMC': ['SMC', '%'],
        'FC': ['$\\theta _{\\rm c}$', '%'],
        'tts': ['$\\theta _{\\rm sun}$', '$\\degree$'],
        'tto': ['$\\theta_ {\\rm view}$', '$\\degree$'],
        'Gfrac': ['$f_{\\rm G}$', '-'],
        'rss': ['$r_{\\rm ss}$', '$\\rm s m^{-1}$'],
        'rss': ['$f_{r_{\\rm ss}}$', '-'],
        'psi': ['$\\phi_{\\rm sun-view}$', 'deg'],
        'Rin': ['$R_{\\rm in}$', '$\\rm W m^{-2}$'],
        'Rli': ['$R_{\\rm li}$', '$\\rm W m^{-2}$'],
        'Ta': ['$T_{\\rm air}$', '$\degree$C'],
        'p': ['$P_{\\rm atm}$', 'hPA'],
        'ea': ['$e_{\\rm a}$', 'hPA'],
        'u': ['$u$', '$\\rm m s^{-1}$'],
        'GPP': ['GPP', '$\\rm \mu molC m^{-2} s^{-1}$'],
        'Rb': ['$R_{\\rm ECO}$', '$\\rm \mu molC m^{-2} s^{-1}$'],
        'Rb_15C': ['$R_{\\rm ECO,15C}$', '$\\rm \mu molC m^{-2} s^{-1}$'],
        'NEP': ['NEP', '$\\rm \mu molC m^{-2} s^{-1}$'],
        'LUE': ['LUE', '$\\rm \mu molC \mu mol^{-1}$'],
        'LUEgreen': ['LUE$_{\\rm green}$', '$\\rm \mu molC \mu mol^{-1}$'],
        'lE': ['$\lambda \\rm E$', '$\\rm W m^{-2}$'],
        'T': ['$T$', '$\\rm W m^{-2}$'],
        'H': ['$H$', '$\\rm W m^{-2}$'],
        'Rn': ['$R_{\\rm n}$', '$\\rm W m^{-2}$'],
        'G': ['$G$', '$\\rm W m^{-2}$'],
        'ustar': ['$u^{\\rm *}$', '$\\rm m s^{-1}$'],
        'RF': ['$R$', '-'],
        'F': ['$F$', '$\\rm mW m^{-2} s^{-1}$'],
        'LST': ['LST', 'K'],
        'wr': ['$W_{\\rm r}$', '%'],
        'VPD': ['VPD', 'hPa'],
        'PET': ['PET', 'mm d$^{-1}$']
        }
    
    return(var_dict)


def read_var_dict(var_name, col=1, add_brackets=True):
    var_dict = get_var_dict()
    
    if var_name in var_dict.keys():
        if add_brackets:
            return('[%s]' % var_dict[var_name][col])
        else:
            return(var_dict[var_name][col])
    else:
        if add_brackets:
            return('[]')
        else:
            return('')


def get_variable_symbol(var_name, add_brackets=False, subscript=None):
    var_sym = read_var_dict(var_name, col=0, add_brackets=add_brackets)
    
    # Add subscript labels
    if subscript != None:
        # If there are already subscript labels
        if var_sym[-2] == '}':
            var_sym = var_sym[:-2] + ',' + subscript + var_sym[-2:]
        # New subscript label
        elif var_sym[-1] == '$':
            var_sym = var_sym[:-1] + '_{' + subscript + '}' + var_sym[-1] 
        elif var_sym[-1] != '$':
            var_sym = var_sym + '$_{' + subscript + '}$'
    
    return(var_sym)


def get_variable_units(var_name, add_brackets=True):
    var_uds = read_var_dict(var_name, col=1, add_brackets=add_brackets)
    
    return(var_uds)


def get_variable_label(var_name, subscript=None):
    var_lab = (get_variable_symbol(var_name, subscript=subscript) + ' ' +
               get_variable_units(var_name))
    
    return(var_lab)


# %% 3) Plot meteorological data
def do_plot_rss_model(mat_, cmap='viridis', fname=None, plt_show=False):
    fig = plt.figure(figsize=(7, 4.8))
    ax = plt.axes(projection ="3d")
    ax.plot_surface(mat_['X'], mat_['Y'], mat_['Z'], cmap=cmap)   
    ax.set_xlabel('$SM_{\\rm rel}$ [-]')
    ax.set_ylabel('$r_{\\rm ss, factor}$ [-]')
    ax.set_zlabel('$r_{\\rm ss} [s m^{-1}]$')
    ax.view_init(elev=10., azim=40)
    plt.tight_layout()
    
    if fname != None:
        fig.savefig(fname, dpi=300)

    if plt_show:
        plt.show(block=False)
    else:
        plt.close()


def do_plot_meteo_ts(meteo_, plt_show=False, fname_=None):
    yr_num = meteo_['Yr'].values - min( meteo_['Yr'].values)
    dec_doy =  meteo_['DoY'].values +  meteo_['hour'].values / 24.
    indx_day_raw = yr_num * 365 + dec_doy
    indx_day = indx_day_raw - min(indx_day_raw)
    
    fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True)
    plt.tight_layout()
    ax = ax.reshape(-1)
    ax[0].grid()
    ax[0].plot(indx_day, meteo_['Rin'],
                label=get_variable_symbol('Rin'))
    ax[0].plot(indx_day, meteo_['Rli'], c='indigo',
                label=get_variable_symbol('Rli'))
    ax[0].set_ylabel(get_variable_symbol('Rin') + ' or ' +
                        get_variable_label('Rin'))
    ax[0].legend(loc="upper left", ncol=2)
    yl_ = ax[0].get_ylim()
    ax[0].set_ylim(yl_[0], yl_[1] + .1 * (yl_[1] - yl_[0]))
    lbl = add_subplot_label(ax[0], .9, .91, '(a)')
    
    ax[1].grid()
    ax[1].plot(indx_day, meteo_['Ta'])
    ax[1].set_ylabel(get_variable_label('Ta'))
    ax[1].set_ylabel('$T_{\\rm air}$ [$\\degree$ C]')
    lbl = add_subplot_label(ax[1], .9, .91, '(b)')
    
    ax[2].grid()
    ax[2].plot(indx_day, meteo_['tts'], lw=.5)
    ax[2].set_ylabel(get_variable_label('tts'))
    lbl = add_subplot_label(ax[2], .89, .91, '(c)')
    
    ax[3].grid()
    ax[3].plot(indx_day, meteo_['p'])
    ax[3].set_ylabel(get_variable_label('p'))
    lbl = add_subplot_label(ax[3], .89, .91, '(d)')
    
    ax[4].grid()
    ax[4].plot(indx_day, meteo_['SMC'],
                label=get_variable_symbol('SMC'))
    ax[4].plot(indx_day, 100*  meteo_['wr'], c='indigo',
                label=get_variable_symbol('wr'))
    ax[4].set_ylim([0, 100]) 
    ax[4].set_ylabel(get_variable_symbol('SMC') + ' or ' +
                        get_variable_label('wr'))
    ax[4].legend(loc="lower left", ncol=2)
    lbl = add_subplot_label(ax[4], .89, .91, '(e)')
    
    ax[5].grid()
    ax[5].plot(indx_day, meteo_['u'])
    ax[5].set_ylabel(get_variable_label('u'))
    lbl = add_subplot_label(ax[5], .89, .91, '(f)')

    ax[6].grid()
    ax[6].plot(indx_day, meteo_['ea'])
    ax[6].set_xlabel('Time (days)')
    ax[6].set_xlim([indx_day.min(), indx_day.max()])
    ax[6].set_ylabel(get_variable_label('ea'))
    ax[8].set_xlabel('Time (days)')
    lbl = add_subplot_label(ax[6], .89, .91, '(g)')
    
    ax[7].grid()
    ax[7].plot(indx_day, meteo_['VPD'])
    ax[7].set_xlabel('Time (days)')
    ax[7].set_xlim([indx_day.min(), indx_day.max()])
    ax[7].set_ylabel(get_variable_label('VPD'))
    lbl = add_subplot_label(ax[7], .89, .91, '(h)')
    
    ax[8].grid()
    ax[8].plot(indx_day, meteo_['PET'])
    ax[8].set_xlim([indx_day.min(), indx_day.max()])
    ax[8].set_ylabel(get_variable_label('PET'))
    ax[8].set_xlabel('Time (days)')
    lbl = add_subplot_label(ax[8], .89, .91, '(i)')
    
    plt.tight_layout()
    
    if fname_ is not None:
        plt.savefig(fname_, dpi=250, bbox_inches='tight')
    
    if plt_show:
        plt.show(block=False)
    else:
        plt.close()


# Plot the maps
def do_show_bosse_map(im_, title_lb='BOSSE simulation', xlb='x [pixel]',
                      ylb='y [pixel]', add_colorbar=True, cmap='viridis',
                      fname=None, plt_show=False, return_fig_ax=False):
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(im_, cmap=cmap)
    ax.set_title(title_lb)
    ax.set_xlabel(xlb)
    ax.set_ylabel(ylb)
    if add_colorbar:
        fig.colorbar(cax)
    
    if fname != None:
        fig.savefig(fname, dpi=300)

    if plt_show:
        plt.show(block=False)
    else:
        plt.close()
    
    if return_fig_ax:
        return(fig, ax)
        

# Plot the Plant Funcitonal Types map
def do_plot_pft_map(im_, veg_, title_lb='BOSSE simulation', xlb='x [pixel]',
                      ylb='y [pixel]', add_colorbar=False, cmap=None,
                      fname=None, plt_show=False):
    
    # Define a PFT-dedicated color map
    cmap_pft=mcolors.ListedColormap(veg_['pft_col'])
    
    # Generate the PFT map
    fig, ax = do_show_bosse_map(im_, title_lb=title_lb, xlb=xlb, ylb=ylb,
                      add_colorbar=add_colorbar, cmap=cmap_pft, fname=None,
                      plt_show=plt_show, return_fig_ax=True)
    
    # Generate the colorbar
    bounds = np.arange(len(veg_['pft_in']) + 1).tolist()
    norm = mcolors.BoundaryNorm(bounds, len(veg_['pft_in']),
                                extend='neither')
    
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_pft), ax=ax)
    cbar.set_ticks([float(i_) + .5 for i_ in bounds[:-1]])
    cbar.set_ticklabels(veg_['pft_in'], fontsize=8)
    
    if fname != None:
        fig.savefig(fname, dpi=300)

    if plt_show:
        plt.show(block=False)
    else:
        plt.close()


# Plot species'spectra
def do_plot_species_spectra(sp_map, sp_id, wvl, X_, ylbl, cmp_=None, fname=None,
                            plt_show=False):
    # Plot the hyperspectral reflectance factors per species
    if  cmp_ == None:
        cmp_ = mcmaps['tab20']

    plt.figure()
    plt.grid()
    i_, id_ = 0, 0
    for i_, id_ in enumerate(sp_id):
        I_ = sp_map == id_
        plt.plot(wvl, X_[I_].T, c=cmp_(i_))
    plt.xlabel('$\\lambda$ [nm]')
    plt.ylabel(ylbl)
    plt.xlim(wvl[0], wvl[-1])
    plt.title('Hyperspectral reflectance colored per species')
    plt.tight_layout()
    
    if fname != None:
        plt.savefig(fname, dpi=300)

    if plt_show:
        plt.show(block=False)
    else:
        plt.close()