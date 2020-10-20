# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import Mamba
from rl.core.function_approximators.policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP

from utils.plot import read_attr

import os


def load_policy(path, name):
    policy = RobustKerasMLPGassian((1,), (1,), init_lstd=0, name='dummy')
    policy.restore(path, name=name)
    return policy

def load_vfn(path, name):
    vfn = SuperRobustKerasMLP((1,), (1,), name='dummy')
    vfn.restore(path, name=name)
    return vfn


def create_experts(envid, name, path=None, order=True, mix_experts=False, mix_swtich=None):
    def load_expert(path, name):
        expert = load_policy(path, name)
        expert.name = 'expert_policy'
        return expert

    if path is None:
        path = os.path.join('experts',envid)

    dirs = os.listdir(path)
    dirs.sort()


    # Parse expert names
    if type(name) is str:
        # Load all the policies with the same name from all the seeds
        expert_dir_name = [(d,name) for d in dirs]
    elif type(name) is list:
        # The seed and policy pair is provided explicitly
        expert_dir_name = [ string.split('#') for string in name]
    else:
        ValueError('Unknown expert names')

    expert_and_vals = []
    for d, policy_name in expert_dir_name:
        d_path = os.path.join(path,d)
        # load expert
        expert = load_expert(os.path.join(d_path, 'saved_policies'), policy_name)
        # load its MeanSumOfRewards
        score = read_attr(os.path.join(d_path, 'log.txt'), 'MeanSumOfRewards')
        if policy_name=='policy_best':
            val = np.max(score)
        else:
            itr = int(policy_name.split('_')[-1])
            val= score[itr]
        expert_and_vals.append([expert, val, d])

    # sort from the best to the worst
    expert_and_vals.sort(reverse=True, key=lambda i:i[1])
    experts = [expert for expert, val, d in expert_and_vals]

    if not order:  # use a random order
        ind = np.random.permutation(len(experts))
        experts = [ experts[i] for i in ind]

    for i, expert in enumerate(experts):
        expert.name = 'expert_'+str(i)

    # mix experts into swiching experts
    if mix_experts:
        assert mix_swtich is not None
        new_experts = []
        for i in range(len(experts)):
            new_experts.append(SwtichingExpert(experts[i], experts[-1-i], mix_swtich))
        experts = new_experts
    return experts


from rl.core.function_approximators.policies import Policy
class SwtichingExpert(Policy):
    def __init__(self, policy_1, policy_2, mix_swtich):
        self._policy_1 = policy_1
        self._policy_2 = policy_2
        self._mix_swtich = mix_swtich
        self.x_shape = self._policy_1.x_shape
        self.y_shape = self._policy_1.y_shape

    def predict(self, xs, **kwargs):
        assert len(xs)==1
        return np.array([self._call(xs[0])])

    def logp(self, xs, ys): # HACK
        return np.zeros(len(xs))

    def _call(self, x_and_t):
        # assume the last entry is time
        t = x_and_t[-1]
        if t <= self._mix_swtich:
            return self._policy_1(x_and_t)
        else:
            return self._policy_2(x_and_t)

    @property
    def variable(self):
        return None



def create_learner(envid, seed, policy0, vfn0):
    # Try to load the initial policy and value function
    policy_name = 'learner_policy_'+str(seed)
    vfn_name = 'learner_vfn_'+str(seed)
    policy=vfn=None

    path = os.path.join('init_learner',envid)
    if os.path.exists(path):
        for f in os.scandir(path):
            if f.name.endswith(policy_name):
                policy = load_policy(path, policy_name)
            if f.name.endswith(vfn_name):
                vfn = load_vfn(path, vfn_name)
    if policy is None:
        print("Cannot find existing initial policy. Create a new one.")
        policy = policy0
        policy.save(path, name=policy_name)
    if vfn is None:
        print("Cannot find existing initial vfn. Create a new one.")
        vfn = vfn0
        vfn.save(path, name=vfn_name)

    policy.name = 'learner_policy'
    vfn.name = 'learner_vfn'

    return policy, vfn


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
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='learner_policy',
                                    init_lstd=c['init_lstd'],
                                    units=c['policy_units'])
    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='learner_vfn',
                                units=c['value_units'])
    if c['reuse_initialization']:
        policy, vfn = create_learner(c['mdp']['envid'], c['seed'], policy, vfn)

    # Define experts
    if c['use_experts']:
        experts = create_experts(c['mdp']['envid'], **c['expert_info'])
        if c['n_experts'] is not None and len(experts)>c['n_experts']:
            experts = experts[:c['n_experts']]
            # experts = experts[1:c['n_experts']+1]  # HACK exclude the first one
        if len(experts)<1:
            experts = None
    else:
        experts=None

    # Create algorithm
    ro_by_n_samples = c['experimenter']['ro_kwargs']['min_n_samples'] is not None
    alg = Mamba(policy, vfn,
                experts=experts,
                horizon=mdp.horizon, gamma=mdp.gamma,
                mix_unroll_kwargs={'ro_by_n_samples':ro_by_n_samples},
                **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['ro_kwargs'])
    exp.run(seed=c['seed'], **c['experimenter']['run_kwargs'],)


CONFIG = {
    'top_log_dir': 'log_mamba',
    'exp_name': 'cp',
    'seed': 0,
    'mdp': {
        'envid':  'HalfCheetah-v2',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes': 4,
        'min_ro_per_process': 2,  # needs to be at least 2 so the experts will be rollout
        'max_run_calls':25,
        'rw_scale':0.01,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'eval_freq': 1,
            'save_freq': None,
        },
        'ro_kwargs': {
            'min_n_samples': None,
            'max_n_rollouts': 8,
        },
    },
    'algorithm': {
        'optimizer':'rnatgrad0',
        'lr':0.1,
        'max_kl':0.05,
        'delta':None,
        'lambd':0.9,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':2,
        # new kwargs
        'eps':1.0,
        'strategy':'max',
        'policy_as_expert': False,
        'max_n_batches_experts':2,
        'use_bc': True,
        'n_bc_steps':1000,
        'n_value_steps':200,
    },
    'policy_units': (128,128),
    'value_units': (256,256),
    'reuse_initialization':True,
    'init_lstd': -1,
    'use_experts': True,
    'expert_info':{
        'name': 'policy_200', #['100#policy_10', '500#policy_best'],
        'path': None, #'experts/DartCartPole-v1/'+str(100)+'/saved_policies',
        'order': True,  # True to use the ordering based on performance; False to use a random ordering
        'mix_experts': True,
        'mix_swtich': 0.5,
    },
    'n_experts': 2, # None,
}

if __name__ == '__main__':
    main(CONFIG)
