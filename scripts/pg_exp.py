import argparse
import tensorflow as tf

from scripts import configs as C
from rl import experimenter as Exp
from rl.configs import parser as ps
from rl.core.utils import tf_utils as U

from rl.experimenter import MDP
from rl.algorithms.pg import PolicyGradient
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian, tfRobustMLPGaussian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP
def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create env and fix randomness
    env, envid, seed = ps.general_setup(c['general'])


    horizon = None
    gamma=1.0
    env = MDP(env, gamma=gamma, horizon=horizon)

    # Create objects for defining the algorithm
    ob_shape = env.ob_shape
    ac_shape = env.ac_shape
    if horizon is not None: 
        ob_shape = (np.prod(ob_shape)+1,)

    #policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
    #                               init_lstd=0.1,
    #                               units=(256, 256))
    policy = tfRobustMLPGaussian(ob_shape, ac_shape,
                                   init_lstd=0.1,
                                   units=(256, 256))

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function', 
                              units=(256,256))
    alg = PolicyGradient(policy, vfn)

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, env, c['experimenter']['rollout_kwargs'])
    exp.run(**c['experimenter']['run_alg_kwargs'])


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('configs_name', type=str)
    #args = parser.parse_args()
    #configs = getattr(C, args.configs_name)

    configs = {
        'general': {
            'top_log_dir': 'log',
            'envid': 'DartCartPole-v1',
            'seed': 0,
            'exp_name': 'cp',
            'horizon': None,  # the max length of rollouts in training
        },
        'experimenter': {
            'run_alg_kwargs': {
                'n_itrs': 100,
                'pretrain': False,
                'final_eval': False,
            },
            'rollout_kwargs': {
                'min_n_samples': 2000,
                'max_n_rollouts': None,
            },
        }
    }

    main(configs)
