from netlds.models import *
from netlds.generative import *
from netlds.inference import *
from data.sim_data import build_model
import os

try:

    # set simulation parameters
    num_time_pts = 20
    dim_obs = 50
    dim_latent = 2
    obs_noise = 'poisson'
    results_dir = '/home/mattw/results/tmp/'  # for storing simulation params

    # build simulation
    model = build_model(
        num_time_pts, dim_obs, dim_latent, num_layers=2, np_seed=1,
        obs_noise=obs_noise)
    # checkpoint model so we can restore parameters and sample
    checkpoint_file = results_dir + 'true_model.ckpt'
    model.checkpoint_model(checkpoint_file=checkpoint_file, save_filepath=True)
    y, z = model.sample(num_samples=128, seed=123)

    # specify inference network for approximate posterior
    inf_network = SmoothingLDS
    inf_network_params = {
        'dim_input': dim_obs,
        'dim_latent': dim_latent,
        'num_mc_samples': 4,
        'num_time_pts': num_time_pts}

    # specify probabilistic model
    gen_model = FLDS
    noise_dist = obs_noise
    gen_model_params = {
        'dim_obs': dim_obs,
        'dim_latent': dim_latent,
        'num_time_pts': num_time_pts,
        'noise_dist': noise_dist,
        'nn_params': [{'units': 15}, {'units': 15}, {}],  # 3 layer nn to output
        'gen_params': None}

    # initialize model
    models = LDSModel(
        inf_network=inf_network, inf_network_params=inf_network_params,
        gen_model=gen_model, gen_model_params=gen_model_params,
        np_seed=1, tf_seed=1)

    # set optimization parameters
    adam = {'learning_rate': 1e-3}
    opt_params = {
        'learning_alg': 'adam',
        'adam': adam,
        'epochs_training': 10,   # max iterations
        'epochs_display': None,  # output to notebook
        'epochs_ckpt': None,     # checkpoint model parameters (inf=last epoch)
        'epochs_summary': None,  # output to tensorboard
        'batch_size': 4,
        'use_gpu': True,
        'run_diagnostics': False}  # memory/compute time in tensorboard

    data_dict = {'observations': y, 'inf_input': y}
    model.train(data=data_dict, opt_params=opt_params)

    os.remove(checkpoint_file)

    print('test successful')

except:
    print('test error')
