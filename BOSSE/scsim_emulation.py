# %% 0) Imports
import numpy as np
import copy as copy
import time as time
import re

from scipy.special import boxcox, inv_boxcox


# %% 1) Functions
def MPL_pred(M_, X_):
    if ('PCAx' in M_.keys()) and (M_['PCAx'] is not None):
        X_ = M_['PCAx'].transform(X_ - M_['Xm'])

    if (M_['PCA'] is None) and (M_['MMSy'] is None):
        pred_ = M_['MLP'].predict(M_['SC'].transform(X_)) + M_['Ym']

    elif (M_['PCA'] is not None) and (M_['MMSy'] is None):
        pred_ = M_['PCA'].inverse_transform(M_['MLP'].predict(
                M_['SC'].transform(X_))) + M_['Ym']
    elif (M_['PCA'] is None) and (M_['MMSy'] is not None):
        Y2tr = (M_['MLP'].predict(M_['SC'].transform(X_)) + M_['Ym'])
        if Y2tr.ndim == 1:
            Y2tr = Y2tr.reshape(-1, 1)            
        pred_ = M_['MMSy'].inverse_transform(Y2tr)
    elif (M_['PCA'] is not None) and (M_['MMSy'] is not None):
        pred_ = M_['MMSy'].inverse_transform(
            M_['PCA'].inverse_transform(M_['MLP'].predict(
                M_['SC'].transform(X_))) + M_['Ym'])
    return(pred_)


def correct_var_names(leg_label):
    for i_, l_ in enumerate(leg_label):
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
        else:
            leg_label[i_] = '$' + l0_ + '$' + uds

    return(leg_label)


def do_transformation_(x_, trans_, lambda_):
    if trans_ == 'log':
        y_ = np.log1p(x_)
    elif trans_ == 'exp':
        y_ = np.exp(x_)
    elif trans_ == 'rec':
        y_ = (1. / (x_ + 1.))
    elif trans_ == 'boxcox':
        y_ = boxcox(x_ + 1., lambda_)
    elif trans_ == 'power':
        y_ = x_ ** lambda_
    else:
        y_ = copy.deepcopy(x_)
    return(y_)


def EMdotransf_(X, trans_, lambda_):
    """
    Applies inverse transformations
    """
    Y = np.zeros(X.shape)
    for i_, tr_ in enumerate(trans_):
        Y[:, i_] = do_transformation_(X[:, i_], tr_, lambda_[i_])
    return(Y)


def undo_transformation_(x_, trans_, lambda_):
    if trans_ == 'log':
        y_ = np.exp(x_)
    elif trans_ == 'exp':
        y_ = np.log1p(x_)
    elif trans_ == 'rec':
        y_ = (1. / x_) - 1.
    elif trans_ == 'boxcox':
        y_ = inv_boxcox(x_, lambda_) - 1.
    elif trans_ == 'power':
        y_ = x_ ** (1/lambda_)
    else:
        y_ = copy.deepcopy(x_)
    return(y_)


def EMtransf_(X, trans_, lambda_, drop_nonf=True):
    """
    Applies inverse transformations
    """
    if drop_nonf:
        X = X[np.all(np.isfinite(X), axis=1), :]
        
    Y = np.zeros(X.shape)
    for i_, tr_ in enumerate(trans_):
        Y[:, i_] = undo_transformation_(X[:, i_], tr_, lambda_[i_])
    
    if drop_nonf:
        Y = Y[np.all(np.isfinite(Y), axis=1), :]
    return(Y)


def sharp_truncation(X, lb, ub):
    # This avoids erroneous transformations of the generated variables
    Y = np.minimum(np.maximum(X, lb), ub)
    
    return(Y)


def truncatedGMM(n_samples, GMM, trans_, lambda_, LB, UB, LBglob, UBglob,
                 oversmp_=2, rseed_=None, max_loops=10):
    """
    Generates a random sample from a multidimensional Gaussian model and
    applies truncation using the rejection method
    """
    n_samples_aug = max(2000, int(n_samples * oversmp_))
    size_ = 0
    size2_ = 1
    Xout = np.zeros([n_samples, len(trans_)])

    if rseed_ is None:
        GMM.random_state = n_samples
    else:
        GMM.random_state = rseed_
        
    # Transform the global bounds
    LBglob_t = EMdotransf_(LBglob, trans_, lambda_)
    UBglob_t = EMdotransf_(UBglob, trans_, lambda_)

    counter_ = 0
    while (size_ < n_samples) and (counter_ < max_loops):
        # print('Round: %d (%d data)' % (counter_, size_))
        # Radomly draw, transform, and remove non-finite values
        X = EMtransf_(sharp_truncation(
            GMM.sample(n_samples=n_samples_aug)[0],
                                       LBglob_t, UBglob_t), trans_, lambda_,
                      drop_nonf=True)

        if np.any(X):
            # Remove data within the limits
            Isel = np.ones(X.shape[0], dtype=bool)
            for i_ in range(X.shape[1]):
                Isel[np.any((X[:, i_] < LB[i_], X[:, i_] > UB[i_]),
                            axis=0)] = False
                # print(sum(Isel))
            ind_sel = np.where(Isel)[0]

            # Assign truncated data to the output matrix
            if len(ind_sel) > 0:
                size2_ = min(size_ + len(ind_sel), n_samples)
                Xout[size_:size2_, :] = X[ind_sel[:size2_-size_], :]

                size_ += len(ind_sel[:size2_-size_])
        counter_ += 1

    # If not achieved in some cycles, force ranodmly the values within the
    # bounds and select the most likely values
    if (size_ < n_samples):
        # Transform bounds
        LBt = EMdotransf_(LB.reshape(1, -1), trans_, lambda_).ravel()
        UBt = EMdotransf_(UB.reshape(1, -1), trans_, lambda_).ravel()
        X = sharp_truncation(GMM.sample(n_samples=2 * n_samples_aug)[0],
                             LBglob_t, UBglob_t)
        # loglike0 = GMM.score_samples(X)
        
        # Set random values where outside the bounds
        for i_ in range(X.shape[1]):
            I_ = np.any((X[:, i_] < LBt[i_], X[:, i_] > UBt[i_]), axis=0)
            X[I_, i_] = (LBt[i_] + np.random.random(sum(I_)) *
                         (UBt[i_] - LBt[i_]))

        # Compute log-likelihood
        loglike = GMM.score_samples(X)
        
        X = EMtransf_(X, trans_, lambda_, drop_nonf=False)
        I_ = np.all(np.isfinite(X), axis=1)
        X = X[I_, :]
        Isort = np.argsort(loglike[I_])
        
        num_missing = n_samples - size_
        Xout[-num_missing:, :] = X[Isort[-num_missing:], :]
    
        if np.any(np.isfinite(Xout) == False):
            raise ValueError('Non-finite GMM plan trait values')

    return(Xout)