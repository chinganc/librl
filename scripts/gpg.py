import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import GeneralizedPolicyGradient as Algorithm
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP

def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create mdp and fix randomness
    mdp = ps.setup_mdp(c['mdp'], c['seed'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    if mdp.use_time_info:
        ob_shape = (np.prod(ob_shape)+1,)

    # define expert
    expert = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=-1,
                                   units=(64,))
    expert.restore('./experts', name='cp1000_mlp_policy_64_seed_9')
    expert.name = 'expert'

    # define the learner
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=-1,
                                   units=(128,128))

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='expert value function',
                                   units=(256,256))

    # Create algorithm
    if not c['use_experts']:
        experts = None
    else:
        experts = [expert]

    alg = Algorithm(policy, vfn,
                    experts=experts,
                    horizon=mdp.horizon, gamma=mdp.gamma,
                    **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['rollout_kwargs'])
    exp.run(**c['experimenter']['run_kwargs'])


CONFIG = {
    'top_log_dir': 'log_gpg',
    'exp_name': 'cp',
    'seed': 1234,
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'eval_freq': 1,
            'save_freq': 5,
        },
        'rollout_kwargs': {
            'min_n_samples': 2000,
            'max_n_rollouts': None,
        },
    },
    'algorithm': {
        'lr': 1e-3,
        'delta':None,
        'lambd':0.1,
        'use_policy_as_expert': True,
        'max_n_batches':2,
        'max_n_batches_experts':100,
        'n_pretrain_itrs': 2,
        'n_warm_up_itrs':None,
    },
    'use_experts':True,
}

if __name__ == '__main__':
    main(CONFIG)
