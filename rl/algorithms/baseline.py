import copy
import numpy as np
from functools import partial
from rl.algorithms import PolicyGradient
from rl.algorithms import utils as au
from rl import online_learners as ol
from rl.core.datasets import Dataset
from rl.core.utils.misc_utils import timed, zipsame
from rl.core.utils import logz
from rl.core.utils.mvavg import PolMvAvg
from rl.core.function_approximators import online_compatible
from rl.core.function_approximators.supervised_learners import SupervisedLearner


class Baseline:

    def __init__(self, experts):
        self.experts = experts

    def pi(self, ob, t, done):
        k = np.random.randint(0, len(expert))
        return experts[k](ob)

    def pi_ro(self, ob, t, done):
        return self.pi(ob, t, done)

    def logp(self, obs, acs):
        return np.zeros((len(obs),1))

    def pretrain(self, gen_ro):
        pass

    def update(self, ro):
        pass
