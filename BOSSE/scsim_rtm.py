# %% 0) Imports
import numpy as np
import time as time
from shapely.geometry import box

from BOSSE.scsim_emulation import MPL_pred

# %% 1) RTM and spetral
def lidf_transform(lidfa, lidfb):
    la = (lidfa + lidfb)
    lb = (lidfa - lidfb)
    return (la, lb)


def lidf_back_transform(la, lb):
    lidfa = (la + lb) / 2
    lidfb = (la - lb) / 2
    return (lidfa, lidfb)


def spectral_pred(M_, X_, check_input=False, out_shape=None, sp_res=100):
    if check_input:
        for i_, var in enumerate(M_['feat_sel']):
            print('%s: %.4f; [%.4f, %.4f]' % (var, np.mean(X_[:, i_]),
                                              M_['LB'][i_], M_['UB'][i_]))
    if out_shape is None:
        Y_ = MPL_pred(M_, X_)
        if sp_res != 100:
            tmp_shp = Y_.shape
            out_shape = [int(tmp_shp[0]**.5), int(tmp_shp[0]**.5), tmp_shp[1]]
            Y_ = spatial_upscale(Y_.reshape(out_shape), sp_res)
            out_shp = Y_.shape
            Y_ = Y_.reshape((out_shp[0] * out_shp[1], out_shp[2]))
    else:        
        Y_ = MPL_pred(M_, X_).reshape(out_shape)
        if sp_res != 100:
            Y_ = spatial_upscale(Y_, sp_res)
            
    return (Y_)  


def band_pos(wl, b_):
    I_ = np.where(wl == b_)[0]
    if len(I_) == 0:
        dif_ = np.abs(wl - b_)
        I_ = np.where(dif_ == dif_.min())[0]
    return (I_)


def get_ndvi_nirv(R_, wl, bred=680., bnir=800.):
    Ired = band_pos(wl, bred)
    Inir = band_pos(wl, bnir)
    Rshp = R_.shape
    lwl = len(wl)
    if Rshp[2] == lwl:
        ndvi_ = ((R_[:, :, Inir] - R_[:, :, Ired]) /
                 (R_[:, :, Inir] + R_[:, :, Ired]))
        nirv_ = (R_[:, :, Inir] * (R_[:, :, Inir] - R_[:, :, Ired]) /
                 (R_[:, :, Inir] + R_[:, :, Ired] - 0.08))

    elif Rshp[1] == lwl:
        ndvi_ = (R_[:, Inir] - R_[:, Ired]) / (R_[:, Inir] + R_[:, Ired])
        nirv_ = (R_[:, Inir] * (R_[:, Inir] - R_[:, Ired]) /
                 (R_[:, Inir] + R_[:, Ired] - 0.08))

    elif Rshp[0] == lwl:
        ndvi_ = (R_[Inir] - R_[Ired]) / (R_[Inir] + R_[Ired] - 0.08)
        nirv_ = (R_[Inir] * (R_[Inir] - R_[Ired]) /
                 (R_[Inir] + R_[Ired] - 0.08))
    else:
        ndvi_ = None
        nirv_ = None

    return (ndvi_, nirv_)


def optical_trait_ret(M_, X_, out_shape=None, force_baresoil=False):
    Y_ = MPL_pred(M_, X_)
        
    if force_baresoil:
        I_lai = np.where(np.array(M_['pred_vars']) == 'LAI')[0][0]
        Y_ [Y_[:, I_lai] <= 0., :] = 0.        
    
    if out_shape is not None:
        Y_ = Y_.reshape(out_shape)

    return (Y_)


# %% 2) Spatial
def mirror_image(x_):
    # Replicate an n-dim image in the two first dimensions using the mirrror
    # approach of scipy.ndimage
    shp_ = x_.shape
    xx_ = np.concatenate((np.fliplr(x_[:, 1:]), x_, np.fliplr(x_[:, :-1])),
                         axis=1)
    yxxy_ = np.concatenate((np.flipud(xx_[1:]), xx_, np.flipud(xx_[:-1])),
                          axis=0)
    
    # Define original image upper left corner
    origin_ = [shp_[0]-1, shp_[1]-1]

    # Define image coordinates
    pos = np.empty((yxxy_.shape[0], yxxy_.shape[1], 2))
    tmp_x = np.arange(yxxy_.shape[0])
    tmp_y = np.arange(yxxy_.shape[1])
    (pos[:, :, 0], pos[:, :, 1]) = np.meshgrid(tmp_x, tmp_y)

    return(yxxy_, origin_, pos)


def multivariate_gaussian(pos, mu, Sigma, selected_area):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    Z_ = np.exp(-fac / 2) / N
    
    # Set to 0 pixels outside the effective area
    Z_[selected_area == False] = 0.
    
    return(Z_ / Z_.sum()) 


def get_Gaussian_kernel(mu_, factor_, pos):
    # Define sigma in a way that if spatial resolution = 100, the PSF can sample
    # from a single species (pixels), using a treshold of 4 standard deviations
    sigma_ = (1/4) / (factor_/100)
    Sigma = np.array([[sigma_, 0.], [0., sigma_]])
    
    # Select the effective PSF area, which is the Gaussian at 4 standard deviations
    I_ = np.sqrt(np.sum((pos - mu_)**2, axis=2)) < 4 * sigma_
    
    # Get the Gaussian PSF
    GK_ = multivariate_gaussian(pos, mu_, Sigma, I_)

    return(GK_, I_)


def get_window_areas(hstep, dist_abs, pos, mu_):
    area_ = np.zeros(dist_abs[:, 0].shape).reshape(-1)
    # For pixels completely included, consider area is 1x1 = 1
    Ix1 = ((dist_abs[:, 0] +.5 <= hstep) *
           (dist_abs[:, 1] +.5 <= hstep))
    if Ix1.any():
        area_[Ix1] = 1.
    # For the rest, intersect with the averaging window
    Ix2 = np.where(Ix1 == False)[0]
    if Ix2.shape[0] > 0:
        rect_mean = box(mu_[0] - hstep, mu_[1] - hstep,
                        mu_[0] + hstep, mu_[1] + hstep)
        for i_ in Ix2:
            rect_grid = box(pos[i_, 0] - .5, pos[i_, 1] - .5,
                            pos[i_, 0] + .5, pos[i_, 1] + .5)

            inters_ = rect_grid.intersection(rect_mean)
            area_[i_] = inters_.area
        
    return(area_)


def get_Mean_kernel(out_sz, sz_, mu_, pos, hstep):
    # Define sigma in a way that if spatial resolution = 100, the PSF can sample
    # from a single species (pixels), using a treshold of 4 standard deviations
     
    # Compute distances to the center pixel
    dist_abs = np.abs(mu_ - pos)
    
    # Find the pixels within the window
    I_ = np.where(np.all(dist_abs < hstep +.5, axis=2))
    areas_ = get_window_areas(hstep, dist_abs[I_], pos[I_], mu_)

    MK_ = (areas_ /areas_.sum())

    return(MK_, I_)


def prepare_upscaling(X_, factor_):
    sz_ = X_.shape
    out_sz = int(sz_[0] * factor_ / 100)
    
    # Expand image to apply kernels outside the bounds. Use "mirror" approach
    (X_mirror, centre_mirror, pos) = mirror_image(X_)

    # Set the points where sampling must be applied
    hstep = (1/2) / (factor_/100)        
    center = np.arange(-.5 + hstep, sz_[0] + .5 - hstep, 2 * hstep)

    return(sz_, out_sz, X_mirror, centre_mirror, pos, hstep, center)
    
    
def spatial_upscale(X_, factor_, type_='GaussPFT'):
    # Prepare the upscaling
    (sz_, out_sz, X_mirror, centre_mirror, pos, hstep, center) = (
        prepare_upscaling(X_, factor_))
        
    # Preallocate
    ups_ = np.zeros((out_sz, out_sz, sz_[2]))
    
    if type_ == 'GaussPFT':
        # Generate downgraded image applying Gaussian PSF
        i_, ci_, j_, cj_ = 0, center[0], 0, center[0]
        for i_, ci_ in enumerate(center):
            for j_, cj_ in enumerate(center):
                mu_ = centre_mirror + np.array([cj_, ci_])
                GK_, I_ = get_Gaussian_kernel(mu_, factor_, pos)
                ups_[i_, j_, :] = GK_[I_] @ X_mirror[I_, :]
    elif type_ == 'Mean':
        # Generate downgraded image applying a mean filter
        i_, ci_, j_, cj_ = 0, center[0], 0, center[0]
        for i_, ci_ in enumerate(center):
            for j_, cj_ in enumerate(center):
                mu_ = centre_mirror + np.array([cj_, ci_])
                MK_, I_ = get_Mean_kernel(out_sz, sz_, mu_, pos, hstep)
                ups_[i_, j_, :] = MK_ @ X_mirror[I_[0], I_[1], :]
    else:
        raise ValueError('Upscaling type (%s) not recognized. Should be ' +
                         'GaussPFT or Mean' % type_)
            
    return(ups_)

       
def check_spatial_upscale(scsz_, factor_):
    out_sz = int(scsz_ * factor_ / 100)
    is_valid = out_sz >= 3
    return(out_sz, is_valid)


def get_possible_spatial_res(scsz_, factor_=[100, 90, 60, 30, 10, 5, 1]):
    if isinstance(factor_, list) is True:
        factor_ = np.array(factor_)

    I_ = np.zeros(factor_.shape, dtype=bool)
    for i_, sp_res in enumerate(factor_):
        _, I_[i_] = check_spatial_upscale(scsz_, sp_res)
    
    return(list(factor_[I_]))


def plot_mean_kernel_example(X_, sz_):
    import matplotlib.pyplot as plt
    col_ = ['r', 'g', 'b', 'y', 'k', 'm']
    factor_ = 100
    for k_, factor_ in enumerate([100, 90, 60, 30, 10]):
        out_sz = int(sz_[0] * factor_ / 100)
        hstep = (1/2) / (factor_/100)        
        center = np.arange(-.5 + hstep, sz_[0] + .5 - hstep, 2 * hstep)

        print('SR: %d (%.1f)' % (factor_, (sz_[0]/out_sz)))
        (X_mirror, centre_mirror, pos) = mirror_image(X_)
        
        plt.figure()
        plt.grid()
        plt.xlim([15, 90-15])
        plt.ylim([15, 90-15])
        i_, ci_, j_, cj_ = 0, center[0], 0, center[0]
        for i_, ci_ in enumerate(center):
            for j_, cj_ in enumerate(center):
                mu_ = centre_mirror + np.array([cj_, ci_])
                plt.plot(mu_[0], mu_[1], '.', c=col_[k_])
        
        i_, ci_, j_, cj_ = 0, center[0], 0, center[0]
        mu_ = centre_mirror + np.array([cj_, ci_])
        plt.plot(pos[:, :, 0].reshape(-1), pos[:, :, 1].reshape(-1), '.')
        
        plt.plot(mu_[0], mu_[1], 'o')
        plt.plot([mu_[0] - hstep, mu_[0] - hstep, mu_[0] + hstep,
                mu_[0] + hstep, mu_[0] - hstep],
                [mu_[0] - hstep, mu_[0] + hstep, mu_[0] + hstep,
                mu_[0] - hstep, mu_[0] - hstep], '-r')
        x00_ = centre_mirror[0]-.5
        plt.plot([x00_, x00_, x00_ +  30.5, x00_ +  30.5,  x00_],
                    [x00_, x00_ +  30.5, x00_ +  30.5, x00_, x00_], '--k')