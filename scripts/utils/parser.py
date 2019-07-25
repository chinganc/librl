import os, time, git, gym
import tensorflow as tf
import numpy as np
from rl.experimenter import MDP
from rl.core.utils import logz


def configure_log(config, unique_log_dir=False):
    """ Configure output directory for logging. """

    # parse config to get log_dir
    top_log_dir = config['top_log_dir']
    log_dir = config['exp_name']
    seed = config['seed']

    # create dirs
    os.makedirs(top_log_dir, exist_ok=True)
    if unique_log_dir:
        log_dir += '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(top_log_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, '{}'.format(seed))
    os.makedirs(log_dir, exist_ok=True)

    # Log commit number.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config['git_commit_sha'] = sha

    # save config
    logz.configure_output_dir(log_dir)
    logz.save_params(config)


def setup_mdp(c, seed):
    """ Set seed and then create an MDP. """
    envid = c['envid']
    env = gym.make(envid)
    env.seed(seed)
    # fix randomness
    if tf.__version__[0]=='2':
        tf.random.set_seed(seed)
    else:
        tf.set_random_seed(seed)  # graph-level seed
    np.random.seed(seed)
    mdp = MDP(env, gamma=c['gamma'], horizon=c['horizon'])
    return mdp


