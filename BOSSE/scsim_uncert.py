# %% Imports
import copy
import numpy as np


# %% Trait-correlated random noise
def get_trait_correl_uncert(rng, cov_M, target_std, n_samples, out_size):
    # Convert covariance to correlation matrix
    std = np.sqrt(np.diag(cov_M))
    corr = cov_M / np.outer(std, std)

    # Create a new covariance matrix with the desired std
    scaled_cov = corr * target_std**2

    # Generate relative noise
    u_ = rng.multivariate_normal(mean=np.zeros(cov_M.shape[0]),
                                 cov=scaled_cov,
                                 size=n_samples).reshape(out_size)
    
    return(u_)


# %% Spatially correlated random noise  
def color_noise(psd_eigenvalues, noise_in):
    # Due to circulant embedding, sizes are different. Fill with zeros and crop later
    if np.any(noise_in.shape[:2] != psd_eigenvalues.shape[:2]):
        do_crop = True
        nx, ny = noise_in.shape[:2]

        noise = (np.random.normal(size=psd_eigenvalues.shape) / np.std(
            noise_in, axis=(0, 1), keepdims=True))
        noise[:nx, :ny, :] = noise_in

    else:
        do_crop = False        
        noise = copy.deepcopy(noise_in)

    spat_autocorr_noise = np.zeros_like(noise)

    for i_ in range(noise.shape[2]):
        # Apply Fourier
        W_fft = np.fft.fft2(noise[:, :, i_])

        # Color the noise
        N_fft = (W_fft * np.sqrt(psd_eigenvalues[:, :, i_]))

        spat_autocorr_noise[:, :, i_] = np.real(np.fft.ifft2(N_fft))
    
    if do_crop:
        spat_autocorr_noise = spat_autocorr_noise[:nx, :ny]
        noise = noise[:nx, :ny]

    # Remove any tiny numerical DC offset.
    if spat_autocorr_noise.ndim == 3:
        spat_autocorr_noise -= np.mean(spat_autocorr_noise,
                                       axis=(0, 1), keepdims=True)
    elif spat_autocorr_noise.ndim == 2:
        spat_autocorr_noise -= np.mean(spat_autocorr_noise)
    else:
        raise ValueError('Spatially autocorrelated error requires at least 2D')

    # Renormalize the noise standard deviation to match that of the original
    # noise
    spat_autocorr_noise_renorm = (
        spat_autocorr_noise *
        (np.std(noise, axis=(0, 1), keepdims=True) /
         np.std(spat_autocorr_noise, axis=(0, 1), keepdims=True)))

    # psd_eigenvalues = copy.deepcopy(noise_psd_PT)
    # noise_in = copy.deepcopy(u_trait)
    # plt.clf()
    # plt.plot(spat_autocorr_noise[:, :, 1].reshape(-1),
    #          noise_in[:, :, 1].reshape(-1), '.')
    # plt.plot(spat_autocorr_noise_renorm[:, :, 1].reshape(-1),
    #          noise_in[:, :, 1].reshape(-1), '.')

    return(spat_autocorr_noise_renorm)


def standardize_average_image(image):
    # Standardize each band before combining them
    std_ = np.std(image, axis=(0, 1), keepdims=True)
    std_[np.isclose(std_, 0.)] = 1.

    # Standardize and average
    image_norm_std = (
        (image - np.mean(image, axis=(0, 1), keepdims=True))
        / std_).mean(axis=2)

    return(image_norm_std)


def autocovariance_signed_lags(image):
    image = np.asarray(image, dtype=np.float64)

    ny, nx = image.shape

    # Remove the global mean.
    x = image - np.mean(image)

    covariance = np.zeros((2 * ny - 1, 2 * nx - 1), dtype=np.float64)

    for dy in range(-(ny - 1), ny):
        if dy >= 0:
            y0a = 0
            y0b = dy
            height = ny - dy
        else:
            y0a = -dy
            y0b = 0
            height = ny + dy

        for dx in range(-(nx - 1), nx):
            if dx >= 0:
                x0a = 0
                x0b = dx
                width = nx - dx
            else:
                x0a = -dx
                x0b = 0
                width = nx + dx

            a = x[y0a:y0a + height, x0a:x0a + width]

            b = x[y0b:y0b + height, x0b:x0b + width]

            value = np.sum(a * b)

            value /= (height * width)

            covariance[ dy + ny - 1, dx + nx - 1] = value

    # Numerical symmetrization. For a real stationary field:
    # C(dx,dy) = C(-dx,-dy)
    covariance = 0.5 * (covariance
                        + np.flip(np.flip(covariance, axis=0), axis=1))

    return(covariance)


def normalize_covariance(covariance_in, nx, ny):
    center_y = ny - 1
    center_x = nx - 1
    variance = covariance_in[center_y, center_x]
    if variance <= 0:
        raise ValueError("Reference image has zero variance after mean removal.")
    covariance_out = covariance_in / variance

    return(covariance_out, variance, center_y, center_x)


def circulant_embed_covariance(signed_covariance, nx, ny):
    # Check the inputs
    expected_shape = (2 * ny - 1, 2 * nx - 1)

    if signed_covariance.shape != expected_shape:
        raise ValueError( f"Expected covariance shape {expected_shape}, "
                         f"got {signed_covariance.shape}")

    center_y = ny - 1
    center_x = nx - 1

    embedded = np.zeros((2 * ny - 2, 2 * nx - 2), dtype=np.float64)

    # Positive / zero lags.
    positive = signed_covariance[center_y:, center_x:]

    embedded[:ny, :nx] = positive[:ny, :nx]

    # x reflection
    embedded[:ny, nx:] = positive[:ny, 1:nx - 1][:, ::-1]

    # y reflection
    embedded[ny:, :nx] = positive[1:ny - 1, :nx][::-1, :]

    # x/y reflection
    embedded[ny:, nx:] = positive[1:ny - 1, 1:nx - 1][::-1, ::-1]

    return(embedded)


def covariance_to_psd(embedded_covariance):
    eigenvalues = np.real(np.fft.fft2(embedded_covariance))

    eigenvalues = np.maximum(eigenvalues, 0.0)

    return(eigenvalues)


def compute_sigle_band_psd(image, nx, ny):
    # Estimate signed-lag covariance
    covariance = autocovariance_signed_lags(image)

    # Normalize covariance
    (covariance, _, _, _) = normalize_covariance(covariance, nx, ny)

    # plt.figure(1)
    # plt.imshow(image_norm_std)
    # plt.figure(2)
    # plt.imshow(covariance)

    # Apply 2-D circulant embedding
    embedded_covariance = circulant_embed_covariance(covariance, nx, ny)
    # plt.figure(3)
    # plt.imshow(embedded_covariance)

    # 4. Wiener-Khinchin: The autocorrelation and PSD are Fourier pairs.
    # For the circulant embedding, the FFT of the covariance kernel
    # provides the eigenvalues of the covariance matrix.
    eigenvalues = covariance_to_psd(embedded_covariance)
    # plt.figure(4)
    # plt.imshow(eigenvalues)
    
    # Numerical zero modes are harmless, but if the covariance model 
    # generates no power at all, something is wrong.
    if np.max(eigenvalues) <= 0:
        raise ValueError("The embedded covariance has no positive spectral power.")

    return(eigenvalues)


def get_spatial_autocorrelation_for_noise(image, per_band=False):
    # Get image size
    nx, ny, nbands = image.shape

    if per_band:
        # Preallocate
        psd_eigenvalues = np.zeros_like(image)

        b_ = 0
        for b_ in range(nbands):

            # Compute signed-lag covariance, apply circulant embedding, 
            # and compute power spectral distribution eigenvalues later
            # used to color white noise
            psd_eigenvalues[:, :, b_] = compute_sigle_band_psd(
                image[:, :, b_], nx, ny)

    else:
        # Standardize and average the image
        image_norm_std = standardize_average_image(image)

        # Compute signed-lag covariance, apply circulant embedding, and
        # compute  power spectral distribution eigenvalues later used to
        # color white noise
        psd_eigenvalues = compute_sigle_band_psd(image_norm_std, nx, ny)

        # Create a cube matching the original image size
        psd_eigenvalues = np.repeat(
            np.expand_dims(psd_eigenvalues, axis=2), nbands, axis=2)

    return(psd_eigenvalues)


def simulate_uncertainty(utype, ref, trait, range_pt_sc, range_rf_sc, seednum,
                         constant_bias=0.05, constant_decorrelated=0.05,
                         cov_M_ref=None, cov_M_trait=None, noise_psd_ref=None,
                         noise_psd_trait=None):
    rng = np.random.default_rng(seed=seednum)
    # Define shapes
    ref_sz = ref.shape
    trait_sz = trait.shape

    if utype == 'no_uncertainty':
        u_ref = 0.
        u_trait = 0.
    elif utype == 'constant_bias':
        # The same bias applies to each spectral band or plant trait
        # plant trait For reflectnace, truncate it to prevent too small
        # values that provid huge relative uncertainties
        u_ref = rng.uniform(-constant_bias * 1.96,
                            constant_bias * 1.96) * range_rf_sc
        u_trait = rng.uniform(-constant_bias * 1.96,
                              constant_bias * 1.96) * range_pt_sc
    elif utype == 'constant_decorrelated':
        # Random relative noise of the variable range applies both to
        # reflectance and plant traits
        u_ref = (rng.normal(0, constant_decorrelated, size=ref_sz) *
                 range_rf_sc)
        u_trait = (rng.normal(0, constant_decorrelated, size=trait_sz) *
                   range_pt_sc)
    elif utype == 'constant_t-correlated':
        u_ref = (get_trait_correl_uncert(rng, cov_M_ref, constant_decorrelated,
                                   ref_sz[0] * ref_sz[1], ref_sz) *
                                   range_rf_sc)
        u_trait = (get_trait_correl_uncert(rng, cov_M_trait, constant_decorrelated,
                                     trait_sz[0] * trait_sz[1], trait_sz) *
                                     range_pt_sc)
    elif utype == 'constant_xy-correlated':
        # Random relative noise of the variable range applies both to
        # reflectance and plant traits
        u_ref = (rng.normal(0, constant_decorrelated, size=ref_sz) *
                 range_rf_sc)
        u_trait = (rng.normal(0, constant_decorrelated, size=trait_sz) *
                   range_pt_sc)
        # Apply spatial autocorrelation
        u_ref = color_noise(noise_psd_ref, u_ref)
        u_trait = color_noise(noise_psd_trait, u_trait)        
    elif utype == 'scaled_bias':
        # The same bias applies to each variable (trait or band) scaled
        # with its average value
        mean_rf_band = np.reshape(
            np.mean(np.mean(ref, axis=1), axis=0), (1, 1, ref_sz[2]))
        mean_trait = np.reshape(
            np.abs(np.mean(np.mean(trait, axis=1), axis=0)),
            (1, 1, trait_sz[2]))
        u_ref = np.repeat(np.repeat(
            (rng.uniform(-constant_bias * 1.96,
                         constant_bias * 1.96) * mean_rf_band),
            ref_sz[0], axis=0), ref_sz[1], axis=1)
        u_trait = np.repeat(np.repeat(
            (rng.uniform(-constant_bias * 1.96,
                         constant_bias * 1.96) * mean_trait),
            trait_sz[0], axis=0), trait_sz[1], axis=1)
    elif utype == 'scaled_decorrelated':
        # Random noise scaled with each data point value
        u_ref = rng.normal(0., constant_decorrelated, size=ref_sz) * np.abs(ref)
        u_trait = (rng.normal(0., constant_decorrelated, size=trait_sz) *
                   np.abs(trait))
    elif utype == 'scaled_t-correlated':
        u_ref = (get_trait_correl_uncert(rng, cov_M_ref, constant_decorrelated,
                                   ref_sz[0] * ref_sz[1], ref_sz) *
                                   np.abs(ref))
        u_trait = (get_trait_correl_uncert(rng, cov_M_trait, constant_decorrelated,
                                     trait_sz[0] * trait_sz[1], trait_sz) *
                                     np.abs(trait))
    elif utype == 'scaled_xy-correlated':
        # Random noise scaled with each data point value
        u_ref = rng.normal(0., constant_decorrelated, size=ref_sz) * np.abs(ref)
        u_trait = (rng.normal(0., constant_decorrelated, size=trait_sz) *
                   np.abs(trait))
        # Apply spatial autocorrelation
        u_ref = color_noise(noise_psd_ref, u_ref)
        u_trait = color_noise(noise_psd_trait, u_trait)

    # Remove non-finite values in case these existed
    if utype != 'no_uncertainty':
        I_ = np.where(np.isfinite(u_ref) == False)
        if I_[0].any():
            u_ref[I_[0], I_[1], I_[2]] = np.nanmean(
                np.nanmean(u_ref[:, :, I_[2]], axis=0, keepdims=True),
                        axis=1, keepdims=True)
        I_ = np.where(np.isfinite(u_trait) == False)
        if I_[0].any():
            u_trait[I_[0], I_[1], I_[2]] = np.nanmean(
                np.nanmean(u_trait[:, :, I_[2]], axis=0, keepdims=True),
                        axis=1, keepdims=True)
    
    return(u_ref, u_trait)

