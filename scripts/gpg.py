import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import GeneralizedPolicyGradient
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP


import os
def create_experts(path, name):
    def load_expert(path, name):
        expert = RobustKerasMLPGassian((1,), (1,), init_lstd=0, name='dummy')
        expert.restore(path, name=name)
        expert.name = 'expert'
        return expert

    for f in os.scandir(path):  # for a single expert
        if f.name.endswith(name):
            experts = [load_expert(path, name)]
            break
    else: # for a set of experts
        experts = []
        for d in os.scandir(path):
            experts.append(load_expert(os.path.join(path,d.name,'saved_policies'), name))
            experts[-1].name = 'expert_'+str(len(experts))
    return experts

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
                                   init_lstd=c['init_lstd'],
                                   units=c['policy_units'])

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function',
                              units=c['value_units'])

    # Define expert
    if c['use_experts']:
        experts = create_experts(c['expert_path'],c['expert_name'])
        if c['n_experts'] is not None and len(expert)>c['n_experts']:
            expert = experts[c['n_experts']]
    else:
        experts=None

    # Create algorithm
    ro_by_n_samples = c['experimenter']['rollout_kwargs'] is not None
    alg = GeneralizedPolicyGradient(policy, vfn,
                                    experts=experts,
                                    horizon=mdp.horizon, gamma=mdp.gamma,
                                    ro_by_n_samples=ro_by_n_samples,
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
            'save_freq': None,
        },
        'rollout_kwargs': {
            'min_n_samples': None, #2000,
            'max_n_rollouts': 4, #None,
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
        'uniform':True,
        'use_policy_as_expert': True,
        'max_n_batches_experts':100,
    },
    'policy_units': (128,128),
    'value_units': (256,256),
    'init_lstd': -1,
    #
    'use_experts':True,
    'expert_path':'./experts/cp_experts',
    'expert_name':'policy_best', # 'cp1000_mlp_policy_64_seed_9',
    'n_experts': None,
}

if __name__ == '__main__':
    main(CONFIG)
