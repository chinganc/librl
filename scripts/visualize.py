# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import gym
from rl.experimenter.mdps import MDP
from rl.algorithms.algorithm import PolicyAgent
from rl.core.function_approximators.policies import RobustKerasMLPGassian

def load_policy(path, name):
    policy = RobustKerasMLPGassian((1,), (1,), init_lstd=0, name='dummy')
    policy.restore(path, name=name)
    return policy

def main(envid,
         policy_path,
         policy_name,
         horizon=1000,
         seed=None):

    # Create mdp
    env = gym.make(envid)
    env.seed(seed)
    mdp = MDP(env, horizon=horizon)

    # Load policy
    policy = load_policy(policy_path, policy_name)
    agent = PolicyAgent(policy)

    # Run one rollout
    ros, agents = mdp.run(agent, max_n_rollouts=1, with_animation=True)
    ro = ros[0]
    print('sum of reward', np.sum(ro['rws']))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dir', help='The dir of experiments', type=str)
    parser.add_argument('-n','--name', help='The name of policy', type=str, default='learner_policy_best')
    parser.add_argument('-e','--envid', help='The name of environment', type=str, default='Humanoid-v2')
    parser.add_argument('-t','--horizon', help='The problem horizon', type=int, default=1000)
    parser.add_argument('-s','--seed', help='Seed for the environment', type=int, default=0)

    args = parser.parse_args()

    main(envid=args.envid,
         policy_path=args.dir,
         policy_name=args.name,
         horizon=args.horizon)
