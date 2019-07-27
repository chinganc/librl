import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import GeneralizedPolicyGradient
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

    # Define the learner
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=-1,
                                   units=(128,128))

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='expert value function',
                                   units=(256,256))

    # Define expert
    expert = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=-1,
                                   units=())  # size doesn't matter
    expert.restore('./experts', name='cp1000_mlp_policy_64_seed_9')
    expert.name = 'expert'

    # Create algorithm
    experts = [expert] if c['use_experts'] else None
    alg = GeneralizedPolicyGradient(policy, vfn,
                                    experts=experts,
                                    horizon=mdp.horizon, gamma=mdp.gamma,
                                    **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['rollout_kwargs'])
    exp.run(**c['experimenter']['run_kwargs'])


CONFIG = {
    'top_log_dir': 'log_gpg',
    'exp_name': 'cp',
    'seed': 0,
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
            'min_n_samples': None, #2000,
            'max_n_rollouts': 2,
        },
    },
    'algorithm': {
        'optimizer':'adam',
        'lr':0.001,
        'max_kl':0.05,
        'delta':None,
        'lambd':0.5,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':5,
        # new kwargs
        'eps':0.5,
        'use_policy_as_expert': True,
        'max_n_batches_experts':100,
    },
    'use_experts':True,
}

if __name__ == '__main__':
    main(CONFIG)
