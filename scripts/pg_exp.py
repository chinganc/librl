import argparse
import tensorflow as tf

from scripts import configs as C
from rl import experimenter as Exp
from rl.configs import parser as ps
from rl.core.utils import tf_utils as U

from rl.algorithms.pg import PolicyGradient
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP
def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create env and fix randomness
    env, envid, seed = ps.general_setup(c['general'])

    # Create objects for defining the algorithm
    ob_shape = env.observation_space.shape
    ac_shape = env.action_space.shape
    policy = RobustKerasMLPGassian(ob_shape, ac_shape,
                                   init_lstd=0.1,
                                   units=(256, 256))
    vfn = SuperRobustKerasMLP(ob_shape, (1,))
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
                'max_rollout_len': None,  # the max length of rollouts in training
            },
        }
    }

    main(configs)
