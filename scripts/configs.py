# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy

# This file contains certain default configs that can used to compose new
# configs by overwriting certain fields in CONFIG.

def def_traj_config(c):
    c = copy.deepcopy(c)
    c['experimenter']['ro_kwargs']['max_n_rollouts'] = \
        c['experimenter']['ro_kwargs']['min_n_samples']/c['mdp']['horizon']
    c['experimenter']['ro_kwargs']['min_n_samples'] = None
    return c

# Dart

config_cp = {
    'exp_name': 'cp',
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,
        'gamma': 1.0
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 100},
        'ro_kwargs': {'min_n_samples': 2000},
    },
}

config_cp_traj = def_traj_config(config_cp)

config_dip = {
    'exp_name': 'dip',
    'mdp': {
        'envid': 'DartDoubleInvertedPendulumEnv-v1',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 2000},
    },
}

config_dip_traj = def_traj_config(config_dip)

config_hopper = {
    'exp_name': 'hopper',
    'mdp': {
        'envid': 'DartHopper-v1',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_hopper_traj = def_traj_config(config_hopper)

config_reacher = {
    'exp_name': 'reacher',
    'mdp': {
        'envid': 'DartReacher-v1',
        'horizon': 500,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 500},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_reacher3d_traj = def_traj_config(config_reacher)


config_reacher3d = {
    'exp_name': 'reacher3d',
    'mdp': {
        'envid': 'DartReacher3d-v1',
        'horizon': 500,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 500},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_reacher3d_traj = def_traj_config(config_reacher3d)

config_dog = {
    'exp_name': 'dog',
    'mdp': {
        'envid': 'DartDog-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_dog_traj = def_traj_config(config_dog)

config_humanwalker = {
    'exp_name': 'humanwalker',
    'mdp': {
        'envid': 'DartHumanWalker-v1',
        'horizon': 300,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_humanwalker_traj = def_traj_config(config_humanwalker)

config_walker2d = {
    'exp_name': 'walker2d',
    'mdp': {
        'envid': 'DartWalker2d-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 500},
        'ro_kwargs': {'min_n_samples': 16000},
        'rw_scale': 0.01,
    },
}

config_walker2d_traj = def_traj_config(config_walker2d)

config_walker3d = {
    'exp_name': 'walker3d',
    'mdp': {
        'envid': 'DartWalker3d-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_walker3d_traj = def_traj_config(config_walker3d)

config_snake = {
    'exp_name': 'snake',
    'mdp': {
        'envid': 'DartSnake7Link-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_sanke_traj = def_traj_config(config_snake)


#PyBullet
config_bhumanoid = {
    'exp_name': 'bhumanoid',
    'mdp': {
        'envid': 'HumanoidPyBulletEnv-v0',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 50000,
                      'max_n_rollouts': None},
    },
}

config_bhumanoid_traj = def_traj_config(config_bhumanoid)


#Mujoco
config_mujoco_humanoid = {
    'exp_name': 'mujoco_humanoid',
    'mdp': {
        'envid': 'Humanoid-v2',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 50000,
                      'max_n_rollouts': None},
    },
}

config_mujoco_humanoid_traj = def_traj_config(config_mujoco_humanoid)



config_mujoco_halfcheetah = {
    'exp_name': 'mujoco_mujoco_halfcheetah',
    'mdp': {
        'envid': 'HalfCheetah-v2',
        'horizon': 500,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000*4},
    },
}

config_mujoco_halfcheetah_traj = def_traj_config(config_mujoco_halfcheetah)


config_mujoco_reacher = {
    'exp_name': 'mujoco_reacher',
    'mdp': {
        'envid': 'Reacher-v2',
        'horizon': 500,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_mujoco_reacher_traj = def_traj_config(config_mujoco_reacher)


config_mujoco_ant = {
    'exp_name': 'mujoco_ant',
    'mdp': {
        'envid': 'Ant-v2',
        'horizon': 500,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_mujoco_ant_traj = def_traj_config(config_mujoco_ant)

config_mujoco_walker2d = {
    'exp_name': 'mujoco_walker2d',
    'mdp': {
        'envid': 'Walker2d-v2',
        'horizon': 500,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_mujoco_walker2d_traj = def_traj_config(config_mujoco_walker2d)

