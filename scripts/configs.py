# This file contains certain default configs that can used to compose new
# configs by overwriting certain fields in CONFIG.

config_cp = {
    'exp_name': 'cp',
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,
        'gamma': 1.0
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 100},
        'rollout_kwargs': {'min_n_samples': 2000},
    },
}


config_hopper = {
    'exp_name': 'hopper',
    'mdp': {
        'envid': 'DartHopper-v1',
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 0},
        'rollout_kwargs': {'min_n_samples': 16000},
    },
}


config_snake = {
    'exp_name': 'snake',
    'mdp': {
        'envid': 'DartSnake7Link-v1',
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'rollout_kwargs': {'min_n_samples': 16000},
    },
}


config_walker3d = {
    'exp_name': 'walker3d',
    'mdp': {
        'envid': 'DartWalker3d-v1',
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'rollout_kwargs': {'min_n_samples': 16000},
    },
}

