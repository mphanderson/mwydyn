import concurrent.futures
import configparser
import copy
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from packaging import version
from platform import python_version

import astropy
import lmfit
import matplotlib
import astropy.constants as const
import astropy.units as u
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from lmfit import Model
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm



plt.rcParams['font.size'] = 9.0

# ==============================================================================

def parse_config(config):
    """
    Reads input configuration file name (e.g. 'config.ini'), extracts
    parameter options and returns lists of parameter values for each section.

    Configuration file sections:
    - [paths]: Main directory and subdirectory management
    - [input]: Input data filename and subcube selection
    - [processing]: Parallel processing options
    - [fitting]: Fitting procedure options
    - [output]: Set various output options
    """
    # Initialise ConfigParser
    cfg = configparser.ConfigParser(inline_comment_prefixes='#')
    # Read configuration file
    cfg.read(config)
    # Get items from each section

    # [paths]:
    path = cfg.get('paths', 'path')
    if path == 'default':
        path = os.getcwd() + '/'  # Path to repositor
    inp_dir = path + cfg.get('paths', 'input_dir')
    pro_dir = path + cfg.get('paths', 'prod_dir')
    mod_dir = path + cfg.get('paths', 'model_dir')

    # [input]:
    inp_fn = cfg.get('input', 'input_fn')
    s_cube = cfg.getboolean('input', 'subcube')
    if s_cube:  # Format bounds into list if subcube: True
        s_bounds = [int(sb) for sb in cfg.get(
            'input', 'subcube_bounds').strip('[]').split(",")]
    else:
        s_bounds = None

    # [processing]:
    para = cfg.getboolean('processing', 'parallel')
    nprc = cfg.get('processing', 'nproc')
    if (nprc == 'auto') & (mp.cpu_count() > 2):
        nprc = mp.cpu_count() - 1
    elif (mp.cpu_count() > 2):
        nprc = int(nprc)
    else:
        para = False
        nprc = 1

    # [fitting]:
    rmschan = cfg.getint('fitting', 'rmschan')
    snrlim = cfg.getfloat('fitting', 'snrlim')
    n_max = cfg.getint('fitting', 'n_max')
    delbic = cfg.getfloat('fitting', 'delbic')
    fwhm_guess = cfg.getfloat('fitting', 'fwhm_guess')
    tau_guess = cfg.getfloat('fitting', 'tau_guess')
    fwhm_limits = tuple([float(t) for t in cfg.get(
        'fitting', 'fwhm_limits').strip('[]').split(',')])
    tau_limits = tuple([float(t) for t in cfg.get(
        'fitting', 'tau_limits').strip('[]').split(',')])
    v_guess_tolerance = cfg.getfloat('fitting', 'v_guess_tolerance')
    line_model = cfg.get('fitting', 'line_model')
    min_dv = cfg.getfloat('fitting', 'min_dv')
    constrain_fits = cfg.getboolean('fitting', 'constrain_fits')
    method = cfg.get('fitting', 'method')
    verbose = cfg.getboolean('fitting', 'verbose')
    cleaniter = cfg.getint('fitting', 'cleaniter')
    refrad = cfg.getint('fitting', 'refrad')
    use_integ_res = cfg.getboolean('fitting', 'use_integ_res')

    # [output]:
    save_products = cfg.getboolean('output', 'save_products')
    do_summary_figures = cfg.getboolean('output', 'do_summary_figures')
    do_plots = cfg.getboolean('output', 'do_plots')
    save_figures = cfg.getboolean('output', 'save_figures')
    save_table = cfg.getboolean('output', 'save_table')

    # Create lists of options for each section of the config file
    cfg_paths = [path, inp_dir, pro_dir, mod_dir]
    cfg_input = [inp_fn, s_cube, s_bounds]
    cfg_processing = [para, nprc]
    cfg_fitting = [rmschan, snrlim, n_max, delbic,
                   fwhm_guess, tau_guess, fwhm_limits, tau_limits,
                   v_guess_tolerance, line_model, min_dv, constrain_fits,
                   method, verbose, cleaniter, refrad,
                   use_integ_res]
    cfg_output = [save_products, do_summary_figures, do_plots, save_figures,
                  save_table]
    """
    These chould be dictionaries instead, but at least this is compact.
    """

    return cfg_paths, cfg_input, cfg_processing, cfg_fitting, cfg_output

# ==============================================================================

# Initialise code timing and set some plotting settings
start_time = time.time()
plt.ion()
plt.rcParams['image.origin'] = 'lower'

# Get input data and config file from command line arguments
if len(sys.argv) == 1:
    print(' No config file or input data specified. \n Quitting.')
    sys.exit()
else:
    arg_list = sys.argv[1:]  # Get command line arguments
    # arg_list = ['/opt/homebrew/bin/ipython', 'config.ini', 'filename.fits']

# Determine which arguments are present
dat_name = [dn for dn in arg_list if '.fits' in dn]
cfg_name = [cn for cn in arg_list if '.ini' in cn]

# Parse config file
if len(cfg_name) == 1:
    # print('\nUsing configuration file: {}'.format(cfg_name[0]))
    cfg_lists = parse_config(config=cfg_name[0])
else:
    print('\n No configuration file specified. ' +
          '\n Using default: config.ini')
    cfg_lists = parse_config(config='config.ini')

# Get section options from parsed config fi le and set appropriate variables
cfg_paths, cfg_input, cfg_processing, cfg_fitting, cfg_output = cfg_lists

# Path options
path = cfg_paths[0]
inpt_dir = cfg_paths[1]
prod_dir = cfg_paths[2]
if not os.path.exists(prod_dir):
    os.makedirs(prod_dir)  # Make product directory if it doesn't exist
model_dir = cfg_paths[3]

# Input data options
if (cfg_input[0] == 'user') & (len(dat_name) == 0):  # Check iput filename
    print('\n No input data specified in command line or configuration file.' +
          '\n Quitting.')
    sys.exit()
elif len(dat_name) == 1:
    print('\n Input data specified in command line, overriding configuration ' +
          'file entry.')  # Using input data: {}'.format(dat_name[0]))
    inpt_fn = dat_name[0]
else:
    print('\n Input data not specified in command line, using configuration ' +
          'file entry.')  # Using input data: {}'.format(cfg_input[0]))
    inpt_fn = cfg_input[0]
if cfg_input[1]:  # Check if only a subcube should be fit
    ymin, ymax, xmin, xmax = cfg_input[2]

# Parallel processing options
parallel = cfg_processing[0]
nproc = cfg_processing[1]

# Fitting options
rmschan = cfg_fitting[0]
snrlim = cfg_fitting[1]
N_max = cfg_fitting[2]
delbic = cfg_fitting[3]
fwhm_guess= cfg_fitting[4]
tau_guess = cfg_fitting[5]
fwhm_limits = cfg_fitting[6]
tau_limits = cfg_fitting[7]
v_guess_tolerance = cfg_fitting[8]
line_model = cfg_fitting[9]
min_dv = cfg_fitting[10]
constrain_fits = cfg_fitting[11]
method = cfg_fitting[12]
verbose = cfg_fitting[13]
cleaniter = cfg_fitting[14]
refrad = cfg_fitting[15]
use_integ_residuals = cfg_fitting[16]

# Output options
save_products = cfg_output[0]
do_summary_figures = cfg_output[1]
do_plots = cfg_output[2]
save_figures = cfg_output[3]
save_table = cfg_output[4]

# Unless verbose mode is enabled, turn off annoying astropy warnings:
if not verbose:
    warnings.simplefilter('ignore', category=AstropyWarning)

# ==============================================================================

def RMS(x, **kwargs):
    """
    Calculate the root mean square (RMS) of an input array x.
    """
    rms = np.sqrt(np.nanmean(np.square(x), **kwargs))

    return rms


def RMS_map(data, method='ends', auto_Plim=0.005, ends_chans=rmschan,
            frac_chans=0.125):
    """
    Returns a masked spectrum containing only noise channels

    Arguments:
        spectrum - an astropy Quantity object containing a spectrum
        method - 'auto', 'ends', or 'frac' ['ends']:
            * auto - automatically identifies noise-only regions by assuming
                that series of consecutive channels that have either all
                positive or negative are unlikely to be attributable to random
                noise. Assumes that spectra are well baselined. (EXPERIMENTAL)
            * ends - assumes that the first and last channels in a spectrum are
                emission free.
            * frac - takes the RMS from a fraction of the start and end of the
               spectrum, as defined by rmsfrac.
        Plim - in conescutive mode, this gives the minimum likelihood that an
               series of consecutive positive or negative pixels can be
               expected from random noise. Smaller values leave more spectrum
               unmasked [0.005].
        endchan - number of channels at the beginning and end of the spectrum
            assumed to be emission free [rmschan].
    Returns:
        The input spectrum with channels assumed to contain signal as NaNs
    Notes:
        The 'consecutive' method is similar to that described in Riener et al.
        (2019) Sect 3.1.1.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis, np.newaxis]

    if type(data) == u.Quantity:
        dataunit = data.unit
        dataval = data.value
    else:
        dataunit = 1
        dataval = data

    if method == 'ends':
        return RMS(np.concatenate([dataval[:ends_chans, :, :],
                                   dataval[-ends_chans:, :, :]]),
                   axis=0) * dataunit
    elif method == 'frac':
        nchans = int(frac_chans * data.shape[0])
        return RMS(np.concatenate([dataval[:nchans, :, :],
                                   dataval[-nchans:, :, :]]),
                   axis=0) * dataunit
    elif method == 'auto':
        auto_map = np.zeros_like(data[0]) * np.nan

        for i in tqdm(range(data.shape[2])):
            for j in range(data.shape[1]):
                spec = dataval[:, j, i]
                masked_spec = spec.copy()
                # Identify series of consecutive positive and negative values
                positive = np.where(spec > 0)[0]
                negative = np.where(spec < 0)[0]
                posseries = np.split(positive,
                                     np.where(np.diff(positive) != 1)[0] + 1)
                negseries = np.split(negative,
                                     np.where(np.diff(negative) != 1)[0] + 1)
                poslist = [list(p) for p in posseries]
                neglist = [list(p) for p in negseries]
                lists = poslist + neglist

                # Estimate probability of their random occurrence
                prob = 0.5**np.array([len(lis) for lis in lists])

                # Mask out windows whose occurence is inconsistent with noise
                for window in np.where(prob < Plim)[0]:
                    masked_spec[lists[window]] = np.nan
                auto_map[j, i] = RMS(masked_spec)
        return auto_map * dataunit
    else:
        print('RMS method not recognised')


def mask_spectrum(spectrum, method='ends', Plim=0.005, endchan=rmschan):
    """
    Returns a masked spectrum containing only noise channels

    Arguments:
        spectrum - an astropy Quantity object containing a spectrum
        method - 'consecutive' or 'ends' ['consecutive']:
            * consecutive - assumes that series of consecutive channels that
                have either all positive or negative are unlikely to be
                attributable to random noise, and masks those.
            * ends - assumes that the first and last channels in a spectrum are
                emission free.
        Plim - in conescutive mode, this gives the minimum likelihood that an
               series of consecutive positive or negative pixels can be
               expected from random noise. Smaller values leave more spectrum
               unmasked [0.005].
        endchan - number of channels at the beginning and end of the spectrum
            assumed to be emission free [rmschan].
    Returns:
        The input spectrum with channels assumed to contain signal as nans
    Notes:
        The 'consecutive' method is similar to that described in Riener et al.
        (2019) Sect 3.1.1.
    """
    masked_spectrum = spectrum.copy()
    if method == 'consecutive':
        # Identify series of consecutive positive and negative values
        positive = np.where(spectrum.value > 0)[0]
        negative = np.where(spectrum.value < 0)[0]
        posseries = np.split(positive, np.where(np.diff(positive) != 1)[0] + 1)
        negseries = np.split(negative, np.where(np.diff(negative) != 1)[0] + 1)
        lists = np.concatenate([posseries, negseries])

        # Estimate probability of their random occurrence
        prob = 0.5**np.array([len(lis) for lis in lists])

        # Mask out windows whose occurence is inconsistent with noise
        for window in np.where(prob < Plim)[0]:
            masked_spectrum[lists[window]] = np.nan
    elif method == 'ends':
        masked_spectrum[rmschan:-rmschan] = np.nan
    return masked_spectrum


def tau(vels, p2, p3, p4):
    """
    Purpose:
        Eq. 2.2 from CLASS hyperfine fitting guide. See:
            https://www.iram.fr/IRAMFR/GILDAS/doc/html/class-html/node11.html
    Arguments:
        vels = velocities of channels for produced spectrum
        p2 = centroid velocity
        p3 = velocity FWHM
        p4 = Tau main
    Returns:
        Model tau "spectrum".
    """
    nu_i, r_i = model['nu_0i'], model['r_i']  # Hyperfine components
    nu_rest = model['nu_rest']  # Rest frequency
    # Velocity shifts of hyperfine components
    v_i = nu_i.to(u.km / u.s, equivalencies=u.doppler_radio(nu_rest)).value

    # Calculate tau for each component
    t_i = [r_i[i] * np.exp(-4 * np.log(2) * ((vels - v_i[i] - p2) / p3)**2) for
           i in range(len(v_i))]
    # Calculate tau(nu)
    tau_nu = p4 * np.nansum(t_i, axis=0)

    return tau_nu


def Tant(vels, p1, p2, p3, p4):
    """
    Purpose:
        Equation 2.3 from CLASS hyperfine fitting guide - 1 component. See:
            https://www.iram.fr/IRAMFR/GILDAS/doc/html/class-html/node11.html
    Arguments:
        vels = velocities of channels for produced spectrum
        p1 = Tant * Tau
        p2 = centroid velocity
        p3 = velocity FWHM
        p4 = Tau main
    Returns:
        Model intensity/antenna temperature spectrum.
    """
    T_ant = (p1 / p4) * (1 - np.exp(-tau(vels, p2, p3, p4)))

    return T_ant


def set_pars(p, p_guess, p_bound, prefix=''):
    """
    Sets initial guess (p_guess) paramters and their bounds (p_bound) for an
    input LMFIT parameter object p. A parameter name prefix can be specified if
    fitting model is composed of a series of sub-models. p_guess is a list of
    values, p_bound is a list of tuples with format: (lower, upper).
    """
    # Extract p_guess values
    p1_g, p2_g, p3_g, p4_g = p_guess
    # Extract p_bound tuples
    p1_b, p2_b, p3_b, p4_b = p_bound
    # Set guess parameters
    p[prefix + 'p1'].value = p1_g
    p[prefix + 'p2'].value = p2_g
    p[prefix + 'p3'].value = p3_g
    p[prefix + 'p4'].value = p4_g
    # Set bounds of fit parameters
    p[prefix + 'p1'].min, p[prefix + 'p1'].max = p1_b
    p[prefix + 'p2'].min, p[prefix + 'p2'].max = p2_b
    p[prefix + 'p3'].min, p[prefix + 'p3'].max = p3_b
    p[prefix + 'p4'].min, p[prefix + 'p4'].max = p4_b

    return p


def fitcomp(spec, vaxis, ncomp=N_max, guesses=None, bounds=None, min_dv=min_dv,
            rmschan=rmschan, method=method, verbose=False):
    """
    The main fitting code.

    Purpose:
        Produce a best-fit model of n components in a spectrum
    Arguments:
        spec - the spectrum to fit
        vaxis - the accompanying velocity axis.
        ncomp - number of components to fit
        guesses - List of the 4 guess parameters: T*tau, vlsr, fwhm, tau
        bounds - List of tuples of the lower and upper bounds of each parameter
        min_dv - Minimum separation between velocity componenets. If set to
                 'linewidth' then separation is set to linewidth/2, though this
                 is not yet tested/fully functional.
        rmschan - Number of channels from ends of spectrum used to calculate
                  rms [25].
        method - ['leastsq'] the lmfit 'method'
    Returns:
        The lmfit mod.fit() object
    """
    if verbose:
        print(f'\n ----- Fitting spectrum at {spec.yx} -----')
    if not isinstance(ncomp, int):
        raise ValueError('ncomp must be an integer')
    if ncomp == 1:
        if verbose:
            print(f'\n Fitting {ncomp} component')
    elif ncomp > 1:
        if verbose:
            print(f'\n Fitting {ncomp} components')
    if ncomp > N_max:
        raise ValueError('The number of componenets (ncomp) must be between'
                         + ' 1 and {}'.format(N_max))

    rms = RMS(mask_spectrum(spec)).value
    w = 1 / rms

    if guesses is None:
        # T*tau, vlsr, fwhm, tau
        guesses = [np.nanmax(spec).value,
                   vaxis[np.argmax(spec)].value,
                   fwhm_guess,
                   tau_guess]
    if bounds is None:
        '''
        Notes on bounds [Ttau, vlsr, fwhm, tau]:
        -Ttau should be constrained by [N*rms, peak] * tau_main bounds.
        -vlsr should be probably constrained by the dispersion of the cloud.
        -fwhm should be probably constrained by the dispersion of the cloud.
        -tau (main) could be the same way as GILDAS, or be more generous.
        '''
        fwhm_min, fwhm_max = fwhm_limits
        tau_min, tau_max = tau_limits
        v_gt = v_guess_tolerance
        vlsr_min, vlsr_max = (np.max((guesses[1] - v_gt, vaxis.min().value)),
                              np.min((guesses[1] + v_gt, vaxis.max().value)))

        bounds = [(tau_min * 3 * rms, tau_max * 1.5 * guesses[0]),
                  (vlsr_min, vlsr_max),
                  (fwhm_min, fwhm_max),
                  (tau_min, tau_max)]

    fitlist = []

    for i in range(ncomp):
        comp = i + 1
        pfix = '_' + str(comp) + '_'   # Model parameter prefix
        if i == 0:
            mod = Model(Tant, prefix=pfix)
            pars = mod.make_params()
        else:
            # if verbose: print('\n * Fitting {} components'.format(ncomp))
            # Add extra line model (component) to total model (series)
            mod += Model(Tant, prefix=pfix)
            # Extend pars to include new model parameters
            pn = mod.param_names[-4:]  # Get 4 newest parameter names
            pars.add_many((pn[0],), (pn[1],), (pn[2],), (pn[3],))

        pars = set_pars(p=pars,
                        p_guess=guesses,
                        p_bound=bounds,
                        prefix=pfix)


        if constrain_fits:
            # Add minimum component peak strength (strongest hyperfine, r=7/27)
            pars.add('min_peak', value=2 * snrlim * rms, min=snrlim * rms,
                     vary=True)
            pars[pfix + 'p1'].expr = ('min_peak * ' + pfix +
                                      'p4 / (1 - exp(-' + pfix +
                                      'p4 * (7/27)))')
        '''
        WARNING: This is expression currently hard-coded for N2H+(1-0)

        This is currently set as snrilim, with the inital guess at 2*snrlim,
        but could be lowered.

        We could consider having this only activate when line_model='n2h+_1-0',
        ideally though there should be a way for the user to edit this condition
        depending on the molecule (might be a lot of work)
        '''

        if (i > 0):
            '''
            If the initial fit wasn't an obvious success, try guessing the
            velocity from the velocity range of detected components. Previously
            from the velocity of the residual maximmum.
            '''
            previous_vlsr = fitlist[0].best_values['_1_p2']

            # Establish range of detected emission
            '''Set the minimum SNR for this method of guess the second vlsr_max
            to give an insignificant probability of mis-interpretating a random
            noise feature as emission. There is a 1 in 2000 chance of
            encountering a spurious SNR = 3.5 noise feature, so this limit
            should work so long as the user has significantly fewer than 2000
            channels per spectrum. At a snr_vguess of 8.2, the isolated
            component in the CLASS 7-component N2H+ model will just be
            detected, and so values lower than this risk this method not
            working properly. The spatial refitting procedure should, however,
            fix most issues arising from this.'''

            snr_vguess = max([snrlim / 2, 3.5])

            detected_v_ax = v_ax[np.where(spec.value > (snr_vguess * rms))]
            vrange = [detected_v_ax.min().value, detected_v_ax.max().value]

            # Given 1st comp fit, calculate expected range of hyperfine comps
            previous_hf_v = previous_vlsr + model_dv.value

            # Determine offset between vrange and expected range from 1st comp
            diff = [vrange[0] - previous_hf_v.min(),
                    vrange[1] - previous_hf_v.max()]
            vsep_guess = diff[np.argmax(np.abs(diff))]

            guesses[1] = previous_vlsr + vsep_guess

            vlsr_min, vlsr_max = (np.max((guesses[1] - v_gt,
                                          vaxis.min().value)),
                                  np.min((guesses[1] + v_gt,
                                          vaxis.max().value)))

            # Update the bounds for the new velocity guess
            bounds = [(tau_min * 3 * rms, tau_max * 1.5 * guesses[0]),
                      (vlsr_min, vlsr_max),
                      (fwhm_min, fwhm_max),
                      (tau_min, tau_max)]

            if min_dv == 'linewidth':
                min_dv_expr = ('(((_1_p3/2) + _1_p2) or ' +
                               '(-(_1_p3/2) + _1_p2))')
                if i > 1:
                    min_dv_expr = (min_dv_expr + ' and ' +
                                   '(((_2_p3/2) + _1_p2) or ' +
                                   ' (-(_2_p3/2) + _2_p2))')
            else:
                if vsep_guess < 0:
                    pars.add('min_dv', value=vsep_guess - min_dv,
                             max=-min_dv, vary=True)
                elif vsep_guess > 0:
                    pars.add('min_dv', value=vsep_guess + min_dv,
                             min=min_dv, vary=True)
                min_dv_expr = '(min_dv + _1_p2)'
                if i > 1:
                    if vsep_guess < 0:
                        pars.add('min_dv', value=vsep_guess - min_dv,
                                 max=-min_dv, vary=True)
                    elif vsep_guess > 0:
                        pars.add('min_dv', value=vsep_guess + min_dv,
                                 min=min_dv, vary=True)
                    min_dv_expr = (min_dv_expr + ' and ' +
                                   '(min_dv + _2_p2)')
            pars[pfix + 'p2'].expr = min_dv_expr

            pars = set_pars(p=pars,
                            p_guess=guesses,
                            p_bound=bounds,
                            prefix=pfix)

        fit = mod.fit(spec.value, pars, vels=vaxis.value, weights=w,
                      method=method, scale_covar=False,
                      nan_policy='propagate')

        residuals = fit.data - fit.best_fit

        # Add additional data to fit object
        fit.rms = rms
        fit.ncomp = comp
        fit.residuals = residuals
        fit.rms_residuals = RMS(residuals)
        fitlist.append(fit)

    return fitlist


def bestcomps(fitlist, verbose=False, delbic0=delbic, redchimax=25):
    """
    Determine which number of components is optimal

    Arguments:
        fits - list of model fits. The result of fitcomp.
        verbose - more print statements
        sigrac - [0.2] fractional improvement in redchi/bic to be
                 significant
        delbic0 - [10] minimum decrease in BIC considered significant
               Wikipedia suggests a value of 10, but might overfit for us.
        redchimax - [25] Limit at which one component will be returned with a
                    bad quality flag.
    Returns:
        nbest - the optimal number of components
    """
    ncomps = len(fitlist)
    converged = [fit.success for fit in fitlist]

    if verbose:
        print(f'\n  Determing the best of {ncomps} components')

    rchs = [fit.redchi for fit in fitlist]
    bics = [fit.bic for fit in fitlist]
    del_bics = [bics[i - 1] - bics[i] for i in range(ncomps)[1:]]

    nbest = 0

    for i in range(ncomps):
        # Check if fit converged
        if fitlist[i].success:
            if verbose:
                print(f'\nComponent {i + 1} fit converged')
            quality = 1
        else:
            if verbose:
                print(f'\nComponent {i + 1} fit not converged')
            quality = 0
        if i == 0:
            nbest = 1

        # Check for improvement in BIC
        if i > 0:
            if verbose:
                print(f'-> Comparing fits with {i} and {i + 1} components...')
            if del_bics[i - 1] > delbic0:
                if verbose:
                    print('   + Significant improvement in BIC')
                nbest = i + 1
            elif verbose:
                print('   - No significant improvement in BIC')
    if (nbest == 0):
        nbest = 1 + np.nanargmin(bics)
        quality = 0
        if verbose:
            print('No fantastic fit found, returning BIC minimum.')
    if verbose:
        if nbest == 1:
            print('1 component is the best fit.\n')
        else:
            print(f'{nbest} components is the best fit.\n')

    return nbest, quality


def plot_fits(fitlist, precision=2, title=None, components=True):
    """
    Plots the data, fit, and residuals on an axis object
    Arguments:
        fitlist - list of lmfit objects to plot
        precision - number of decimal places on the printed fit parameters
        title - title for the figure
    """

    fig = plt.figure(figsize=(8, 2.5 * len(fitlist)))
    plt.subplots_adjust(hspace=0)
    if title is not None:
        fig.suptitle(title)

    for f, fit in enumerate(fitlist):
        ncomps = f + 1
        if ncomps == 1:
            compstring = 'component'
        else:
            compstring = 'components'
        if ncomps == bestcomps(fitlist)[0]:
            fc = 'k'
            alpha = 1
        else:
            fc = 'grey'
            alpha = 0.5
        if fit.success:
            fitstring = 'converged'
        else:
            fistring = 'not converged'

        ax = fig.add_subplot(len(fitlist), 1, ncomps)
        ax.plot(fit.userkws['vels'], fit.residuals,
                c='k', alpha=alpha, label='residuals', lw=1)
        ax.step(fit.userkws['vels'], fit.data,
                c='k', alpha=alpha, label='data')
        if components:
            # print(fit.best_values)
            for i in range(ncomps):
                if i == 0:
                    clabel = 'components'
                else:
                    clabel = None
                Ttau = fit.best_values[f'_{i + 1}_p1']
                vcen = fit.best_values[f'_{i + 1}_p2']
                fwhm = fit.best_values[f'_{i + 1}_p3']
                taum = fit.best_values[f'_{i + 1}_p4']
                ax.plot(v_ax, Tant(v_ax.value, Ttau, vcen, fwhm, taum),
                        c='salmon', lw=1, alpha=alpha, label=clabel)
        ax.plot(fit.userkws['vels'], fit.best_fit,
                c='r', alpha=alpha, label='model')
        if f == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
        ax.set_xlabel('Velocity [km s$^{-1}$]')
        ax.axhline(5 * fit.rms, color='k', ls='--', lw=0.8, alpha=alpha)
        ax.text(0.05, 0.9,
                '{} {}: {}'.format(ncomps, compstring, fitstring)
                + '\nTant * tau = {}'.format(
                    np.round([fit.best_values['_{}_p1'.format(i)] for i in
                              1 + np.arange(ncomps)], precision))
                + '\nvlsr = {}'.format(
                    np.round([fit.best_values['_{}_p2'.format(i)] for i in
                              1 + np.arange(ncomps)], precision))
                + '\nFWHM = {}'.format(
                    np.round([fit.best_values['_{}_p3'.format(i)] for i in
                              1 + np.arange(ncomps)], precision))
                + '\ntau = {}'.format(
                    np.round([fit.best_values['_{}_p4'.format(i)] for i in
                              1 + np.arange(ncomps)], precision))
                + '\nrms_res = {:.4f}'.format(fit.rms_residuals)
                + '\nrchisq = {:.2f}'.format(fit.redchi)
                + '\nBIC = {:.1f}'.format(fit.bic),
                  va='top', ha='left', transform=ax.transAxes, c=fc)
    xmin = np.min([ax.get_position().xmin for ax in fig.axes])
    xmax = np.max([ax.get_position().xmax for ax in fig.axes])
    ymin = np.min([ax.get_position().ymin for ax in fig.axes])
    ymax = np.max([ax.get_position().ymax for ax in fig.axes])

    fig.text(xmin * 0.4, (ymin + ymax) * 0.5,
             'Brightness [{}]'.format(u.Unit(head['BUNIT']).to_string()),
             rotation='vertical', va='center')


def plot_fits_yx(iy, ix):
    """"
    A thin wrapper for plot_fits which takes the y and x indices as opposed to
    the flattend j index.
    """
    jind = np.where(np.sum((yxindices.T == [iy, ix]), axis=1) == 2)[0][0]
    plot_fits(fitcomp(spec[:, jind], v_ax), title=(iy, ix))


def plot_fit(y, x, components=True):
    """
    Plots the best fit for a pixel based on what is saved in the data products
    Arguments:
        iy, ix - pixel coordinates of the spectrum
    """

    fig = plt.figure(figsize=(8, 2.5))
    fig.suptitle(f'(x, y) = ({x}, {y})')

    ncomps = products[-1]['ncomp'][y, x]
    compstring = 'component'
    if ncomps > 1:
        compstring += 's'

    precision = 2  # Number of decimal places to quote fit parameters to
    indices = np.where(~np.isnan(products[-1]['p2_vcen'][:, y, x]))[0]
    Ttau_values = products[-1]['p1_Ttau'][:, y, x][indices]
    vcen_values = products[-1]['p2_vcen'][:, y, x][indices]
    fwhm_values = products[-1]['p3_fwhm'][:, y, x][indices]
    taum_values = products[-1]['p4_taum'][:, y, x][indices]
    success = products[-1]['quality'][y, x]
    thismodel = products[-1]['model'][:, y, x]
    if success == 1:
        fitstring = 'converged'
    else:
        fistring = 'not converged'

    ax = fig.add_subplot(111)
    ax.plot(v_ax, data[:, y, x] - thismodel, label='residuals', c='k', lw=1)
    ax.step(v_ax, data[:, y, x], label='data', c='k')
    if components:
        for i in range(ncomps):
            if i == 0:
                clabel = 'components'
            else:
                clabel = None
            ax.plot(v_ax, Tant(v_ax.value, Ttau_values[i], vcen_values[i],
                               fwhm_values[i], taum_values[i]),
                    c='salmon', lw=1, label=clabel)
    ax.plot(v_ax, thismodel, label='model', c='r')
    ax.legend()
    ax.set_xlabel('Velocity [km s$^{-1}$]')
    ax.set_ylabel('Brightness [{}]'.format(u.Unit(head['BUNIT']).to_string()))
    ax.axhline(snrlim * rms_map[y, x].value, color='k', ls='--', lw=0.5)
    ax.text(0.05, 0.9,
            '{} {}: {}'.format(ncomps, compstring, fitstring)
            + '\nTant * tau = {}'.format(
                np.round([Ttau_values[i] for i in range(ncomps)], precision))
            + '\nvlsr = {}'.format(
                np.round([vcen_values[i] for i in range(ncomps)], precision))
            + '\nFWHM = {}'.format(
                np.round([fwhm_values[i] for i in range(ncomps)], precision))
            + '\ntau = {}'.format(
                np.round([taum_values[i] for i in range(ncomps)], precision))
            + '\nrms_res = {:.4f}'.format(RMS((data[:, y, x] - thismodel)))
            + '\nrchisq = {:.2f}'.format(products[-1]['rchi'][y, x])
            + '\nBIC = {:.1f}'.format(products[-1]['bic'][y, x]),
            va='top', ha='left', transform=ax.transAxes, c='k')
    xmin = np.min([ax.get_position().xmin for ax in fig.axes])
    xmax = np.max([ax.get_position().xmax for ax in fig.axes])
    ymin = np.min([ax.get_position().ymin for ax in fig.axes])
    ymax = np.max([ax.get_position().ymax for ax in fig.axes])


def clickspec(event, refit=False):
    """
    Produces the set of spectra when clicking on any pixel in the summary
    figure
    """
    global ix, iy
    ix, iy = int(np.round(event.xdata, 0)), int(np.round(event.ydata, 0))
    jind = np.where(np.sum((yxindices.T == [iy, ix]), axis=1) == 2)[0][0]
    print('Producing spectrum for (x, y) = ({}, {})'.format(ix, iy))
    ax = event.inaxes
    for ax in [event.canvas.figure.axes[i] for i in
               [0, 2, 4, 6, 8, 10, 12, 14]]:
        if len(ax.collections) > 0:
            ax.collections[0].remove()
        ax.scatter(ix, iy, marker='+', c='r', s=81)

    plt.draw()
    if refit:
        plot_fits(fitcomp(spec[:, jind], v_ax), title=(iy, ix))
    else:
        plot_fit(iy, ix)


def summary_figure(numbers=False, crop_back=True, back_pad=10):
    """
    Produces the summary figure
    Arguments:
        number - if True, plot numbers in the pixels of the integer maps
        crop_back - if True, crop the axes to remove large areas of no data
                    Assumes background values will be 0s or NaNs
        back_pad - minimum number of pixels padding between data and axes
    """
    intmodel = np.nansum(products[-1]['model'], axis=0)
    vmin, vmax = np.nanpercentile(intdata.value, [0, 99])

    if crop_back:
        nzeros = len(np.where(intdata.value == 0)[0])
        n_nans = len(np.where(np.isnan(intdata.value))[0])
        if nzeros > n_nans:
            datapixels = np.where(intdata.value != 0)
        elif n_nans > nzeros:
            datapixels = np.where(~np.isnan(intdata.value))
        else:
            # print('Background values neither 0 or nan. Do not crop.')
            crop_back = False

    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.03, right=0.96, top=0.93, bottom=0.02,
                        hspace=0.02, wspace=0.50)
    fig.suptitle(f'{inpt_fn}\n snrlim={snrlim} delbic={delbic}')

    ax1 = fig.add_subplot(241)
    ax1.set_title('Integrated intensity')
    im = ax1.imshow(intdata.value, vmin=vmin, vmax=vmax)
    if crop_back:
        ymax, xmax = np.max(datapixels, axis=1)
        ymin, xmin = np.min(datapixels, axis=1)
        ax1.set_xlim(xmin - back_pad, xmax + back_pad)
        ax1.set_ylim(ymin - back_pad, ymax + back_pad)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)
    cbar.set_label('[' + (u.Unit(head['BUNIT']) * v_ax.unit).to_string() + ']')

    ax = fig.add_subplot(242, sharex=ax1, sharey=ax1)
    ax.set_title('Integrated model')
    im = ax.imshow((intmodel * chanwidth).value, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)
    cbar.set_label('[' + (u.Unit(head['BUNIT']) * v_ax.unit).to_string() + ']')

    ax = fig.add_subplot(243, sharex=ax1, sharey=ax1)
    ax.set_title('Residuals')
    im = ax.imshow((np.nansum(data, axis=0) -
                    np.nansum(products[-1]['model'], axis=0))
                   * np.abs(v_ax[1] - v_ax[0]).value)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)
    cbar.set_label('[' + (u.Unit(head['BUNIT']) * v_ax.unit).to_string() + ']')

    ax = fig.add_subplot(244, sharex=ax1, sharey=ax1)
    ax.set_title('No. components')
    im = ax.imshow(products[-1]['ncomp'], vmin=0, vmax=N_max)
    if numbers:
        for (j, i), label in np.ndenumerate(products[-1]['comp']):
            ax.text(i, j, label, ha='center', va='center')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0, format="%.0f",
                        ticks=range(N_max + 1))

    ax = fig.add_subplot(245, sharex=ax1, sharey=ax1)
    ax.set_title('rms')
    im = ax.imshow(rms_map.value)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)
    cbar.set_label('[' + (u.Unit(head['BUNIT'])).to_string() + ']')

    ax = fig.add_subplot(246, sharex=ax1, sharey=ax1)
    ax.set_title('Reduced chi-squared')
    im = ax.imshow(products[-1]['rchi'])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)

    ax = fig.add_subplot(247, sharex=ax1, sharey=ax1)
    ax.set_title('BIC')
    im = ax.imshow(products[-1]['bic'])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)

    ax = fig.add_subplot(248, sharex=ax1, sharey=ax1)
    ax.set_title('Quality')
    im = ax.imshow(products[-1]['quality'], vmin=-1, vmax=1)
    if numbers:
        for (j, i), label in np.ndenumerate(prod_qual):
            ax.text(i, j, label, ha='center', va='center', c='w')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0, ticks=[-1, 0, 1])

    for ax in [fig.axes[i] for i in [0, 2, 4, 6, 8, 10]]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add map-clicking feature
    click = fig.canvas.mpl_connect('button_press_event', clickspec)

    print('\n * Click on a pixel to inspect the spectrum and fit')

    return fig


# ======================== Define the model ====================================

def get_model(model_name):
    """
    Imports model from JSON file, reformats, adds units, normalises.
    """
    # Load model dictionary (Probably should have to check for valid model)
    with open(model_dir + model_name + '.json', 'r') as file:
        model = json.load(file)

    # Pull items from dictionary, reformat
    f_unit = u.Unit(model['freq_unit'])
    f_0i = model['nu_0i'] * f_unit
    f_rest = model['nu_rest'] * f_unit
    r_i = model['r_i'] / np.sum(model['r_i'])

    # Create new model dictionary, populate with reformated items
    model_dict = {}
    model_dict['nu_0i'] = f_0i
    model_dict['nu_rest'] = f_rest
    model_dict['r_i'] = r_i

    return model_dict


model = get_model(line_model)
# Store the component separations (in km/s) in an array for reference
model_dv = ((model['nu_rest'] - model['nu_0i'])
            / model['nu_rest']) * const.c.to('km/s')


# ========================== Begin main ========================================

print('\n ======================= Running mwydyn ===========================')
print(f' File: {inpt_fn}')
print(f' Config file: {cfg_name[0]}')
print(f' Maximum no. components: {N_max}')
print(f' Minimum S/N: {snrlim}')
print(f' Critical BIC value: {delbic}')
print(f' Initial FWHM guess: {fwhm_guess}')
print(f' Initial tau guess: {tau_guess}')
print(f' Fitting method: {method}')
print(f' Fit constraints: {constrain_fits}')
if parallel:
    print(f' Parallel processing on {nproc} processors')
    if do_plots:
        do_plots = False
        print(' Warning: can\'t produce plots in parallel mode.'
              + ' Setting do_plots = False.')
else:
    print(' Serial processing')
    if do_plots:
        print(' Producing one figure for each spectrum')
print(f' Cleaning iterations: {cleaniter}')
print(f' Cleaning radius: {refrad} pixels')
print('')
if N_max > 3:
    print('\n ! Warning: fitting more than 3 components. ' +
          'Strange things may happen...')

# Import data
data, head = fits.getdata(inpt_dir + inpt_fn, ext=0, header=True)

# We should convert the spectral data into TA* or Tmb
if u.Unit(head['BUNIT']) == (u.Jy / u.beam):
    # Calculate beam area
    omega_B = (np.pi / (4 * np.log(2))) * \
        (head['BMAJ'] * u.deg * head['BMIN'] * u.deg)
    # Get frequency of data
    if 'RESTFRQ' in head:
        rest_nu = head['RESTFRQ'] * u.Hz  # CASA also has ALTRVAL?
    elif 'RESTFREQ' in head:
        rest_nu = head['RESTFREQ'] * u.Hz  # For NOEMA
    data = data << u.Unit(head['BUNIT'])
    # Convert to brightness temperature K
    data = data.to(u.K,
                   equivalencies=u.brightness_temperature(frequency=rest_nu,
                                                          beam_area=omega_B))
    # Update header, strip unit from data
    head['BUNIT'] = data.unit.to_string()
    data = data.value

# Select subcube if range specified in control panel
if 'xmax' in locals():
    data = data[:, ymin:ymax, xmin:xmax]
    if len(np.where((data != 0) & np.isfinite(data))[0]) == 0:
        print('\n   Warning: no valid spectra in subcube')

# Initialise velocity axis with km/s units
if 'CUNIT3' not in head:
    head['CUNIT3'] = 'm/s'
v_ax = ((head['CRVAL3'] + head['CDELT3']
         * (np.arange(head['NAXIS3']) + 1 - head['CRPIX3'])
         ) * u.Unit(head['CUNIT3'])).to('km/s')

chanwidth = np.abs(v_ax[1] - v_ax[0])

# Integrated intensity image for various plots
intdata = np.nansum(data, axis=0) * chanwidth

# Create an RMS map by estimating noise in line-free channels
rms_map = RMS_map(data) * u.Unit(head['BUNIT'])

# Set detection threshold (for dimmest or brightest component?)
threshold = snrlim * rms_map
# Choose which spectra to fit based on detection threshold
'''NOTE: Need to look more carefully at noise cutoff.'''
mask = (data > threshold.value).sum(axis=0) > 0  # Should be >>0?
'''...loop over selected spectra and run fit...'''

# Current spectrum to be fitted
spec = data[:, mask] << u.Unit(head['BUNIT'])
yxindices = np.asarray(np.where(mask))

# Initialise data product cubes...
sizez, sizey, sizex = data.shape
prod_ncomp = np.zeros_like(data[0], dtype='int')

prod_Ttau = np.zeros([N_max, sizey, sizex]) * np.nan
prod_vcen = np.zeros_like(prod_Ttau) * np.nan
prod_fwhm = np.zeros_like(prod_Ttau) * np.nan
prod_taum = np.zeros_like(prod_Ttau) * np.nan

prod_model = np.zeros_like(data)
prod_rchi = np.zeros_like(data[0]) * np.nan
prod_bic = np.zeros_like(data[0]) * np.nan
prod_qual = np.zeros_like(data[0], dtype='int')
prod_fits = np.zeros_like(data[0])
prod_spid = np.zeros_like(data[0]) * np.nan


def analyse_spectrum(j):
    """
    For a given spectrum index, j, this function will run the fitting code,
    establish the best fit, and write out the best-fitting parameter values and
    velocity-axis indices for populating the data products. By not accessing
    intialized cubes directly, this function is parallelisable.

    Note: If using parallel mode, plot_fits will not work.
    """
    # Set current spectrum to be fit
    s = spec[:, j]
    yind, xind = yxindices.T[j]
    s.yx = (yind, xind)

    testfits = fitcomp(s, v_ax, verbose=verbose)
    nbest, quality = bestcomps(testfits, verbose=verbose)

    if do_plots:
        plot_fits(testfits, title=str(s.yx))

    bf = testfits[nbest - 1]

    results = [(yind, xind), nbest, quality, bf.redchi, bf.bic]

    if quality >= 0:
        # Loop over number of fit components
        for k in range(bf.ncomp):
            # Component prefix
            c_pfix = '_' + str(k + 1) + '_'
            # Find element of v_ax corresponding to best v_cen
            c_ind = np.digitize(bf.best_values[c_pfix + 'p2'], v_ax.value[:-1])
            results.append(c_ind)
            results.append([bf.best_values[c_pfix + p] for p in
                            ['p1', 'p2', 'p3', 'p4']])
    if parallel:
        return results
    else:
        return results, testfits


# ===================== Loop over all spectra ==================================

pixels = range(spec.shape[1])

fit_desc = ' Fitting each spectrum '

# print('\n Performing fits to all spectra:')
if parallel:
    # Vaguely following the tutorial from:
    # https://www.youtube.com/watch?v=fKl2JW_qrso
    if version.parse(python_version()) < version.parse("3.8"):
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as \
                executor:
            results = list(tqdm(executor.map(analyse_spectrum, pixels),
                                total=len(pixels), position=0,
                                desc=fit_desc))
    else:
        if sys.platform == 'darwin':
            # Probably a terrible hack to get it working on macOS
            mpc = mp.get_context('fork')
        else:
            mpc = None
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc,
                                                    mp_context=mpc) as \
                executor:
            results = list(tqdm(executor.map(analyse_spectrum, pixels),
                                total=len(pixels), position=0,
                                desc=fit_desc))
else:
    results = []
    allfits = []
    for j in tqdm(pixels, position=0, desc=fit_desc):
        output = analyse_spectrum(j)
        results.append(output[0])
        allfits.append(output[1])

# Write results to product arrays
if verbose:
    print('\n Parsing results')

for j in pixels:
    result = results[j]
    yidx, xidx = result[0]
    nbest = result[1]

    # 2D data products
    prod_ncomp[yidx, xidx] = nbest
    prod_qual[yidx, xidx] = result[2]
    prod_rchi[yidx, xidx] = result[3]
    prod_bic[yidx, xidx] = result[4]
    prod_spid[yidx, xidx] = j

    for k in range(nbest):
        # Set element of spectra in products to best fit value
        c_ind = result[2 * k + 5]

        # Set element of Kth component to best fit value
        prod_Ttau[k, yidx, xidx] = result[2 * k + 6][0]
        prod_vcen[k, yidx, xidx] = result[2 * k + 6][1]
        prod_fwhm[k, yidx, xidx] = result[2 * k + 6][2]
        prod_taum[k, yidx, xidx] = result[2 * k + 6][3]

        # Store number of components as 2D array
        prod_model[:, yidx, xidx] += Tant(v_ax.value,
                                          result[2 * k + 6][0],
                                          result[2 * k + 6][1],
                                          result[2 * k + 6][2],
                                          result[2 * k + 6][3])


# List to store the products after each stage
products = [{'results': results.copy(),
             'ncomp': prod_ncomp.copy(),
             'quality': prod_qual.copy(),
             'rchi': prod_rchi.copy(),
             'bic': prod_bic.copy(),
             'model': prod_model.copy(),
             'residuals': data - prod_model.copy(),
             'specID': prod_spid.copy(),
             'p1_Ttau': prod_Ttau.copy(),
             'p2_vcen': prod_vcen.copy(),
             'p3_fwhm': prod_fwhm.copy(),
             'p4_taum': prod_taum.copy()}]


# ============================ Refit components ================================

def spatial_refit(y, x, q):
    """
    Checks surrounding pixels for better fits, and uses those fit parameters
    as initial guesses for a refit.

    Arguments;
    y, x - pixel coordinates
    q - the number of the clean iteration. q = 0 refers to the original fits.
    """
    takenewfit = 0
    thisbic = products[q]['bic'][y, x]
    # Pad the bic array with nans to make subarrays easier
    padbic = np.pad(products[q]['bic'], pad_width=refrad, mode='constant',
                    constant_values=np.nan)
    localbic = padbic[y:y + refrad + 2, x: x + refrad + 2]
    localbicmin = np.nanmin(localbic)
    if thisbic == localbicmin:
        # Do not refine if this is the best-fitting pixel already
        return [None]
    elif np.nansum(localbic) == 0:
        return [None]
    else:
        # Process only if this pixel isn't the local bic minimum
        # Coordinates of best bic in local bic map
        bestbic = np.unravel_index(np.nanargmin(localbic), localbic.shape)

        # Convert coords to 'full' bic map
        besty, bestx = np.array(bestbic) - refrad  # Subtract for pad
        besty += y
        bestx += x

        local_vcen = products[q]['p2_vcen'][:, besty, bestx]

        # Identify indices with the fit parametrs
        indices = np.where(~np.isnan(local_vcen))[0]

        if len(indices) != products[q]['ncomp'][besty, bestx]:
            print(f'Warning: something screwing up: y={y}, x={x}, q={q}')

        # Extract the fitting results
        newguess_comp = products[q]['ncomp'][besty, bestx]
        newguess_Ttau = products[q]['p1_Ttau'][indices, besty, bestx]
        newguess_vcen = products[q]['p2_vcen'][indices, besty, bestx]
        newguess_fwhm = products[q]['p3_fwhm'][indices, besty, bestx]
        newguess_taum = products[q]['p4_taum'][indices, besty, bestx]

        newguesses = np.vstack((newguess_Ttau, newguess_vcen,
                                newguess_fwhm, newguess_taum)).T.flatten()

        rspec = data[:, y, x]
        rms = rms_map[y, x].value
        w = 1 / rms

        # Generate the new guess parameters
        for i in range(newguess_comp):
            comp = i + 1
            pfix = '_' + str(comp) + '_'   # Model parameter prefix
            if i == 0:
                mod = Model(Tant, prefix=pfix)
                pars = mod.make_params()
            else:
                mod += Model(Tant, prefix=pfix)
                # Extend pars to include new model parameters
                pn = mod.param_names[-4:]  # Get 4 newest parameter names
                pars.add_many((pn[0],), (pn[1],), (pn[2],), (pn[3],))

            fwhm_min, fwhm_max = fwhm_limits
            tau_min, tau_max = tau_limits
            v_gt = v_guess_tolerance

            vlsr_min, vlsr_max = (np.max((newguesses[int(1 + 4 * i)] - v_gt,
                                         v_ax.min().value)),
                                  np.min((newguesses[int(1 + 4 * i)] + v_gt,
                                         v_ax.max().value)))
            bounds = [(tau_min * 3 * rms,
                       tau_max * 1.5 * newguesses[int(0 + 4 * i)]),
                      (vlsr_min, vlsr_max),
                      (fwhm_min, fwhm_max),
                      (tau_min, tau_max)]

            pars = set_pars(p=pars,
                            p_guess=newguesses[int(0 + 4 * i):
                                               int(0 + 4 * i) + 4],
                            p_bound=bounds,
                            prefix=pfix)

            if constrain_fits:
                # Add minimum component peak strength
                pars.add('min_peak', value=2 * snrlim * rms, min=snrlim * rms,
                         vary=True)
                pars[pfix + 'p1'].expr = ('min_peak * ' + pfix +
                                          'p4 / (1 - exp(-' + pfix +
                                          'p4 * (7/27)))')

            # Fit a new model based on revised guesses
            newfit = mod.fit(rspec, pars, vels=v_ax.value, weights=w,
                             method=method, scale_covar=False,
                             nan_policy='propagate')
            newfit.ncomp = comp
            newfit.residuals = newfit.data - newfit.best_fit
            newfit.rms = rms
            newfit.rms_residuals = RMS(newfit.residuals)

            # Take the new fit if its BIC is significantly better
            condition1 = thisbic - newfit.bic > delbic
            if use_integ_residuals:
                # Also require that the new integrated residuals are smaller?
                old_res_sum = np.nansum(products[q]['residuals'][:, y, x],
                                        axis=0)
                new_res_sum = np.nansum(newfit.residuals)
                condition2 = np.abs(new_res_sum) < np.abs(old_res_sum)
                if condition1:
                    print(old_res_sum, new_res_sum)

            else:
                condition2 = True

            if condition1 & condition2:
                thisbic = newfit.bic
                takenewfit = 1
                results = [(y, x), newfit.ncomp, int(newfit.success),
                           newfit.redchi, newfit.bic]

                # Append best fit parameters as list per component
                for k in range(newfit.ncomp):
                    c_pfix = '_' + str(k + 1) + '_'
                    # Find index of best v_cen
                    c_ind = np.digitize(newfit.best_values[c_pfix + 'p2'],
                                        v_ax.value)
                    results.append(c_ind)
                    results.append([newfit.best_values[c_pfix + p] for p in
                                    ['p1', 'p2', 'p3', 'p4']])
        if takenewfit == 0:
            results = [None]
        return results


def spatial_refit_parallel(j):
    yy, xx = yxindices.T[j]
    result = spatial_refit(yy, xx, q)
    return result


if cleaniter > 0:
    # print('\n Refitting components where a local pixel has a better BIC')

    # Cube to store a flag for whether pixels were refitted, with dimensions
    # CleanIter * ypix * xpix
    ny, nx = np.shape(data[0])
    refitted_pixels = np.zeros([cleaniter, ny, nx])


    pbar = tqdm(total=cleaniter, position=0, desc=' Spatial refinement    ')


    for q in range(cleaniter):
        refit_desc = f' Spatial refinement {q + 1}/{cleaniter}'

        newresults = []

        if parallel:
            if python_version() < '3.8':
                with concurrent.futures.ProcessPoolExecutor(max_workers=nproc)\
                        as executor:
                    newresults = list(executor.map(
                        spatial_refit_parallel, pixels))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=nproc,
                                                            mp_context=mpc)\
                        as executor:
                    newresults = list(executor.map(spatial_refit_parallel,
                                                   pixels))
        else:
            for x in range(data.shape[2]):
                for y in range(data.shape[1]):
                    newresult = spatial_refit(y, x, q)
                    newresults.append(newresult)

        # Save the new results to the relevant arrays in the products dict
        new_products = copy.deepcopy(products[q])
        new_products['results'] = newresults

        for newresult in newresults:
            if newresult != [None]:
                y, x = newresult[0]
                if newresult[0] == [0, 0]:
                    print(f'{y},{x} refitted on iteration {q + 1}')
                # refitted_pixels[y, x] += 1
                refitted_pixels[q, y, x] = 1
                ncomp = newresult[1]

                new_products['ncomp'][y, x] = ncomp
                new_products['quality'][y, x] = newresult[2]
                new_products['rchi'][y, x] = newresult[3]
                new_products['bic'][y, x] = newresult[4]

                # Clear model parameter arrays before adding new values
                new_products['model'][:, y, x] = 0
                new_products['p1_Ttau'][:, y, x] = np.nan
                new_products['p2_vcen'][:, y, x] = np.nan
                new_products['p3_fwhm'][:, y, x] = np.nan
                new_products['p4_taum'][:, y, x] = np.nan

                for k in range(ncomp):
                    # Set element of spectra in products to best fit value
                    z = newresult[2 * k + 5]
                    new_products['p1_Ttau'][k, y, x] = newresult[2 * k + 6][0]
                    new_products['p2_vcen'][k, y, x] = newresult[2 * k + 6][1]
                    new_products['p3_fwhm'][k, y, x] = newresult[2 * k + 6][2]
                    new_products['p4_taum'][k, y, x] = newresult[2 * k + 6][3]
                    new_products['model'][:, y, x] += \
                        Tant(v_ax.value,
                             newresult[2 * k + 6][0],
                             newresult[2 * k + 6][1],
                             newresult[2 * k + 6][2],
                             newresult[2 * k + 6][3])
                new_products['residuals'][:, y, x] = \
                    data[:, y, x] - new_products['model'][:, y, x]
        products.append(new_products)
        pbar.update(1)

        lastiter = q + 1

        # If no further changes were made, we can stop early
        if sum([item == [None] for item in newresults]) == len(newresults):
            print(f'\n No new results on refinement iteration {lastiter}: stopping.')
            break

    # # Close progress bar
    # pbar.close()

    # Consolidate refit maps
    total_refits = np.sum(refitted_pixels, axis=0)
    last_refit = np.argmax(refitted_pixels, axis=0) + 1
    last_refit[total_refits == 0] = 0

    if np.nanmax(last_refit) == cleaniter:
        print('')
        print(' WARNING:')
        print(' Refits still being performed on the last iteration of spatial')
        print(' refinement. You may want to increase the value of CleanIter.')

else:
    lastiter = 0


def spatial_refit_figure(crop_back=True, back_pad=10):
    """
    This figure gives a quick overview of the spatial refitting iterations.
    """

    ncols = 2 + lastiter

    imin, imax = np.nanpercentile(intdata.value, [0, 99])
    rmin, rmax = np.nanpercentile(np.nansum(products[0]['residuals'] *
                                            chanwidth.value,
                                  axis=0), [1, 99])
    bmin, bmax = np.nanpercentile(products[0]['bic'], [1, 99])

    if crop_back:
        nzeros = len(np.where(intdata.value == 0)[0])
        n_nans = len(np.where(np.isnan(intdata.value))[0])
        if nzeros > n_nans:
            datapixels = np.where(intdata.value != 0)
        elif n_nans > nzeros:
            datapixels = np.where(~np.isnan(intdata.value))
        else:
            # print('Background values neither 0 or nan. Do not crop.')
            crop_back = False

    fig = plt.figure(figsize=((2 + lastiter) * 2, 7))
    fig.suptitle(f'{inpt_fn}\n snrlim={snrlim} delbic={delbic}')
    plt.subplots_adjust(left=0.01, bottom=0.065, right=0.966, top=0.88,
                        wspace=0.141, hspace=0.43)

    ax1 = fig.add_subplot(4, ncols, 1)
    im = ax1.imshow(intdata.value, vmin=imin, vmax=imax)
    cbar = fig.colorbar(im)
    ax1.set_title('Integrated data')
    if crop_back:
        ymax, xmax = np.max(datapixels, axis=1)
        ymin, xmin = np.min(datapixels, axis=1)
        ax1.set_xlim(xmin - back_pad, xmax + back_pad)
        ax1.set_ylim(ymin - back_pad, ymax + back_pad)

    ax = fig.add_subplot(4, ncols, ncols + 1, sharex=ax1, sharey=ax1)
    im = ax.imshow(total_refits, vmin=0)
    ax.set_title('No. refits')
    cbar = fig.colorbar(im)


    # Show map where the value is equal to the iteration where the model
    # was last spatially refined
    ax = fig.add_subplot(4, ncols, 2 * ncols + 1, sharex=ax1, sharey=ax1)
    im = ax.imshow(last_refit, vmin=0)
    ax.set_title('Last refitted')
    cbar = fig.colorbar(im)

    # for p in range(1 + cleaniter):
    for p in range(1 + lastiter):
        ax = fig.add_subplot(4, ncols, p + 2,
                             sharex=ax1, sharey=ax1)
        im = ax.imshow(np.nansum(products[p]['model'], axis=0) *
                       chanwidth.value, vmin=imin, vmax=imax)
        cbar = fig.colorbar(im)
        if p == 0:
            ax.set_title(f'Initial integrated model')
        else:
            ax.set_title(f'Integrated model {p}')

        ax = fig.add_subplot(4, ncols, p + ncols + 2,
                             sharex=ax1, sharey=ax1)
        im = ax.imshow(np.nansum(products[p]['residuals'], axis=0) *
                       chanwidth.value, vmin=-rmax, vmax=rmax, cmap='RdBu_r')
        if p == 0:
            ax.set_title(f'Initial residual')
        else:
            ax.set_title(f'Residual {p}')
        cbar = fig.colorbar(im)

        ax = fig.add_subplot(4, ncols, p + 2 * ncols + 2,
                             sharex=ax1, sharey=ax1)
        im = ax.imshow(products[p]['bic'], vmin=bmin, vmax=bmax, cmap='magma')
        if p == 0:
            ax.set_title(f'Initial BIC')
        else:
            ax.set_title(f'BIC {p}')
        cbar = fig.colorbar(im)

        # Show the difference between this iteration and the last
        if p > 0:
            ax = fig.add_subplot(4, ncols, p + 3 * ncols + 2,
                                 sharex=ax1, sharey=ax1)
            im = ax.imshow((np.nansum(products[p]['residuals'], axis=0) -
                            np.nansum(products[p -1 ]['residuals'], axis=0)) *
                            chanwidth.value, vmin=-rmax, vmax=rmax, cmap='RdBu_r')
            ax.set_title(f'Residual for iter {p}-{p - 1}')
            cbar = fig.colorbar(im)

    return fig


# ============= Wrap-up stage: save output and summary figures ============== #

def header_comments(header):
    """
    Adds versions of packages used as comments to an input FITS header.
    NOTE: If we want to add dependencies, they need to be imported.
    """
    # Add comments to header
    header['COMMENT'] = '--------======== mwydyn.py ========--------'
    header['COMMENT'] = 'Release version: TEMP'  # Repository version tag
    header['COMMENT'] = 'Python version: {}'.format(sys.version.split(' ')[0])
    header['COMMENT'] = 'LMFIT version: {}'.format(lmfit.__version__)
    header['COMMENT'] = 'numpy version: {}'.format(np.__version__)
    header['COMMENT'] = 'astropy version: {}'.format(astropy.__version__)
    header['COMMENT'] = 'matplotlib version: {}'.format(matplotlib.__version__)
    header['COMMENT'] = 'mwydyn config file: {}'.format(cfg_name[0])
    header['COMMENT'] = 'Maximum no. components: {}'.format(N_max)
    header['COMMENT'] = 'Minimum S/N: {}'.format(snrlim)
    header['COMMENT'] = 'Critical BIC value: {}'.format(delbic)
    header['COMMENT'] = 'fwhm guess: {}'.format(fwhm_guess)
    header['COMMENT'] = 'tau guess: {}'.format(tau_guess)
    header['COMMENT'] = 'Fitting method: {}'.format(method)
    header['COMMENT'] = 'Fit constraints: {}'.format(constrain_fits)
    if parallel:
        header['COMMENT'] = 'Parallel processing used'
        header['COMMENT'] = 'No. processors: {}'.format(nproc)
    else:
        header['COMMENT'] = 'Serial processing used'
    header['COMMENT'] = 'Cleaning iterations: {}'.format(cleaniter)
    header['COMMENT'] = 'Cleaning radius: {} pixels'.format(refrad)
    header['COMMENT'] = '--------======== mwydyn.py ========--------'

    return header


if save_products:
    if verbose:
        print('\n Saving data products')
    output = list(products[-1].keys())[1:]
    stub = inpt_fn.split('.fits')[0]

    # Create new headers for data products
    if not verbose:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning, append=True)
            hdr2D = WCS(head).celestial.to_header()
            hdr3D = WCS(head).to_header()
    else:
        hdr2D = WCS(head).celestial.to_header()
        hdr3D = WCS(head).to_header()

    # Add package version comments to headers
    hdr2D = header_comments(hdr2D)
    hdr3D = header_comments(hdr3D)
    hdr_Ncomp = hdr3D.copy()
    hdr_Ncomp['CDELT3'] = 1
    hdr_Ncomp['CRPIX3'] = 1
    hdr_Ncomp['CRVAL3'] = 1

    # Write out products
    fits.writeto(f'{prod_dir}{stub}_data.fits', data=data,
                 header=head, overwrite=True)
    fits.writeto(f'{prod_dir}{stub}_rms.fits', data=rms_map.value,
                 header=hdr2D, overwrite=True)
    for prod in output:
        if prod in ['ncomp', 'quality', 'rchi', 'bic', 'rms', 'specID']:
            hdr = hdr2D
        elif prod in ['p1_Ttau', 'p2_vcen', 'p3_fwhm', 'p4_taum']:

            hdr = hdr_Ncomp
        else:
            hdr = hdr3D
        fits.writeto(f'{prod_dir}{stub}_{prod}.fits', data=products[-1][prod],
                     header=hdr, overwrite=True)
    if cleaniter > 0:
        fits.writeto(f'{prod_dir}{stub}_refits.fits', data=refitted_pixels,
                     header=hdr2D, overwrite=True)

    print(f'\n Data products saved to: {prod_dir}')

# Product the summary figures
if do_summary_figures:
    if cleaniter > 0:
        spatial_refit_fig = spatial_refit_figure()
        if save_figures:
            plt.savefig(f'{prod_dir}Figure_{stub}_refits.pdf')
    summary_fig = summary_figure()
    if save_figures:
        plt.savefig(f'{prod_dir}Figure_{stub}_summary.pdf')

if save_table:
    Ttau = products[-1]['p1_Ttau']
    vlsr = products[-1]['p2_vcen']
    fwhm = products[-1]['p3_fwhm']
    taum = products[-1]['p4_taum']
    ncomp = products[-1]['ncomp']
    quality = products[-1]['quality']
    rchi = products[-1]['rchi']
    bic = products[-1]['bic']
    rms = rms_map
    specID = products[-1]['specID']

    # Get WCS from header
    wcs = WCS(head)

    # Get x,y indices of cubes (v index probably not useful)
    vi, yi, xi = np.indices(Ttau.shape)

    # Get mask of where there are datapoints in the cubes
    mask = ~np.isnan(Ttau)

    # Create 1D arrays of each of the quantities for the table
    Ttau_1D = Ttau[mask]
    vlsr_1D = vlsr[mask]
    fwhm_1D = fwhm[mask]
    taum_1D = taum[mask]
    yi_1D = yi[mask]
    xi_1D = xi[mask]
    ncomp_1D = np.tile(ncomp, (Ttau.shape[0], 1, 1))[mask]
    quality_1D = np.tile(quality, (Ttau.shape[0], 1, 1))[mask]
    rchi_1D = np.tile(rchi, (Ttau.shape[0], 1, 1))[mask]
    bic_1D = np.tile(bic, (Ttau.shape[0], 1, 1))[mask]
    rms_1D = np.tile(rms, (Ttau.shape[0], 1, 1))[mask]
    specID_1D = (np.tile(specID, (Ttau.shape[0], 1, 1))[mask]).astype(int)

    # Create WCS coordinates from x,y indices
    coords = SkyCoord.from_pixel(xp=xi_1D, yp=yi_1D, wcs=wcs)
    RA = coords.ra
    Dec = coords.dec
    glon = coords.galactic.l
    glat = coords.galactic.b

    # ----- Calculate T_ant from fit parameters using N2H+ model
    # Save the integrated intensity
    int_T_ant_1D = np.zeros_like(Ttau_1D, dtype='float64') * u.K * u.km / u.s
    # And the peak intensity
    peak_T_ant_1D = np.zeros_like(Ttau_1D, dtype='float64') * u.K

    for i in range(len(Ttau_1D)):
        T_ant_spec = Tant(vels=v_ax.value,
                        p1=Ttau_1D[i], p2=vlsr_1D[i],
                        p3=fwhm_1D[i], p4=taum_1D[i]) * u.K
        int_T_ant_1D[i] = np.sum(T_ant_spec) * chanwidth
        peak_T_ant_1D[i] = np.max(T_ant_spec)

    # Create code for spectrum type (for later analysis)
    speccode_1D = ncomp_1D.astype(float)
    # speccode: ncomp.c_num, c_num=1 is the strongest, c_num=ncomp is the weakest
    s_idx = np.unique(specID_1D)  #  List of unique spectrum IDs

    for s in range(len(s_idx)):
        spec_mask = specID_1D == s  # Create mask
        sc_set = speccode_1D[spec_mask]
        if len(sc_set) == 1:
            # Generate c_num
            c_num = 0.1
        elif len(sc_set) > 1:
            # Get integrated intensities of components
            intT_set = int_T_ant_1D[spec_mask]
            # Generate c_num (only works for n_comp < 10)
            c_num = (np.argsort(intT_set)[::-1] / 10) + 0.1

        # Apply to spec code
        speccode_1D[spec_mask] = sc_set + c_num

    # Create table (NOTE: Units and descriptions need to change!)
    tab = Table(data=[xi_1D, yi_1D,
                      RA, Dec,
                      glon, glat,
                      specID_1D * u.dimensionless_unscaled,
                      Ttau_1D * u.K,
                      vlsr_1D * u.km / u.s,
                      fwhm_1D * u.km / u.s,
                      taum_1D * u.dimensionless_unscaled,
                      int_T_ant_1D,
                      peak_T_ant_1D,
                      ncomp_1D * u.dimensionless_unscaled,
                      speccode_1D * u.dimensionless_unscaled,
                      quality_1D * u.dimensionless_unscaled,
                      rchi_1D * u.dimensionless_unscaled,
                      bic_1D * u.dimensionless_unscaled,
                      rms_1D],
                names=['x_ind', 'y_ind',
                       'RA', 'Dec',
                       'glon', 'glat',
                       'specID',
                       'Ttau', 'vlsr',
                       'fwhm', 'tau',
                       'int_T_ant',
                       'peak_T_ant',
                       'ncomp', 'ncomp.compID',
                       'quality', 'rchi', 'bic', 'rms'],
                # Units not agreeing with astropy now for some reason.
                # Getting a mystery float division by zero error, so have
                # moved the unit definitions into the data argument, only
                # updating where the variables are not quantities already.
                # units=[u.pixel, u.pixel,
                #        u.deg, u.deg,
                #        u.deg, u.deg,
                #        u.dimensionless_unscaled,
                #        u.K, u.km / u.s,
                #        u.km / u.s, u.dimensionless_unscaled,
                #        u.K * u.km / u.s,
                #        u.K,
                #        u.dimensionless_unscaled,
                #        u.dimensionless_unscaled,
                #        u.dimensionless_unscaled,
                #        u.dimensionless_unscaled,
                #        u.dimensionless_unscaled,
                #        u.K],
                descriptions=['x pixel index', 'y pixel index',
                              'RA (FK5)', 'Dec (FK5)',
                              'Galactic longitude', 'Galactic latitude',
                              'Spectrum ID',
                              'p1: T_ant * tau', 'p2: VLSR',
                              'p3: FWHM', 'p4: tau_main',
                              'Integrated intensity of model spectrum',
                              'Peak intensity of model spectrum',
                              'Number of components in spectrum',
                              'Component ID, format: ncom.compID, with ' +
                              'compID=1 being the component with largest ' +
                              'integrated intensity',
                              'Quality flag for fit',
                              'Reduced chi-squared value',
                              'Bayesian information criterion',
                              'rms noise of spectrum'])

    stub_fn = inpt_fn.split('.fits')[0]

    # Why is ECSV not working?
    # tab.write(f'{prod_dir}/{stub_fn}_table.ecsv', format='ecsv',
    #           overwrite=True)
    tab.write(f'{prod_dir}/{stub_fn}_table.fits', format='fits',
              overwrite=True)

t_ex = time.time() - start_time
print("\n Finished in {:.2f} minutes".format(t_ex / 60))
print(" ==================================================================")
