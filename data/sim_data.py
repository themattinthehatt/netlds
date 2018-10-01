"""Simulate data using the Model class in models.py"""

import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, './../')
from netlds.models import *
from netlds.generative import *
from netlds.inference import *


def build_model(
        num_time_pts, dim_obs, dim_latent, np_seed=0, tf_seed=0,
        obs_noise='gaussian'):
    """
    Build netlds model to simulate data

    Args:
        num_time_pts (int): number of time points per trial
        dim_obs (int): number of observation dimensions
        dim_latent (int): number of latent space dimensions
        np_seed (int, optional): numpy rng seed
        tf_seed (int, optional): tensorflow rng seed
        obs_noise (str, optional): distribution of observation noise
            'gaussian' | 'poisson'

    Returns:
        Model object

    """

    dtype = np.float32

    # for reproducibility
    np.random.seed(np_seed)
    tf.set_random_seed(tf_seed)

    # define model parameters
    A = np.array([[1.0, 0.9], [-0.9, 0.01]], dtype=dtype)
    z0_mean = np.array([[0.4, 0.3]], dtype=dtype)
    Q = 0.03 * np.random.randn(2, 2).astype(dtype)
    Q = np.matmul(Q, Q.T)
    Q_sqrt = np.linalg.cholesky(Q)
    C = np.random.randn(dim_latent, dim_obs).astype(dtype)
    d = np.abs(np.random.randn(1, dim_obs).astype(dtype))

    gen_params = {
        'A': A, 'z0_mean': z0_mean, 'Q_sqrt': Q_sqrt, 'Q0_sqrt': Q_sqrt,
        'C': C, 'd': d}

    # specify inference network for approximate posterior
    inf_network = SmoothingLDSCoupled
    inf_network_params = {
        'dim_input': dim_obs,
        'dim_latent': dim_latent,
        'num_time_pts': num_time_pts}

    # specify probabilistic model
    gen_model = LDSCoupled
    if obs_noise is 'gaussian':
        R_sqrt = np.sqrt(0.05 * np.random.uniform(
            size=(1, dim_obs)).astype(dtype))
        gen_params['R_sqrt'] = R_sqrt

    gen_model_params = {
        'dim_obs': dim_obs,
        'dim_latent': dim_latent,
        'num_time_pts': num_time_pts,
        'noise_dist': obs_noise,
        'gen_params': gen_params}

    # initialize model
    model = LDSCoupledModel(
        inf_network=inf_network, inf_network_params=inf_network_params,
        gen_model=gen_model, gen_model_params=gen_model_params)

    return model
