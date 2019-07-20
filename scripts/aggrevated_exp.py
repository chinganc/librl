import argparse
import tensorflow as tf
import numpy as np

from scripts import configs as C
from rl import experimenter as Exp
from rl.configs import parser as ps
from rl.core.utils import tf_utils as U
from rl.experimenter import MDP
from rl.algorithms.aggrevated import AggreVaTeD
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian, tfRobustMLPGaussian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP
def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create env and fix randomness
    env, envid, seed = ps.general_setup(c['general'])
    horizon = c['general']['horizon']
    gamma=1.0
    env = MDP(env, gamma=gamma, horizon=horizon)

    # Create learnable objects
    ob_shape = env.ob_shape
    ac_shape = env.ac_shape
    if np.isclose(gamma,1.0):
        ob_shape = (np.prod(ob_shape)+1,)

    # define expert
    expert = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=0.1,
                                   units=(256, 256))
    expert.restore('./experts')
    expert.name = 'expert'
   
    # define the learner
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=0.1,
                                   units=(256, 256))

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function',
                                   units=(256,256))

   
    # Create algorithm
    alg = AggreVaTeD(policy, expert, vfn, horizon)

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, env, c['experimenter']['rollout_kwargs'])
    exp.run(**c['experimenter']['run_alg_kwargs'])

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('configs_name', type=str)
    #args = parser.parse_args()
    #configs = getattr(C, args.configs_name)

    configs = {
        'general': {
            'top_log_dir': 'log',
            'envid': 'DartCartPole-v1',
            'seed': 230,
            'exp_name': 'cp',
            'horizon': 1000,  # the max length of rollouts in training
        },
        'experimenter': {
            'run_alg_kwargs': {
                'n_itrs': 50,
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
