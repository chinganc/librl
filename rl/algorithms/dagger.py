from rl.algorithms.algorithm import Algorithm, PolicyAgent, merge_ros
from rl.core.function_approximators.supervised_learners.tf2_supervised_learners import create_robust_keras_supervised_learner
from rl.core.datasets import Dataset

class DAgger(Algorithm):
    def __init__(self, policy,
                 mode='bc', # 'bc' or 'dagger'
                 expert = None,  # a Policy (optional)
                 max_n_samples=None,  # number of trajectories to keep
                 max_n_batches=20,  # number of batches to keep
                 init_ro = None,  # a Dataset of batch data of expert trajectories
                 n_pretrain_itrs=1):

        self.policy = policy
        # Wrap the policy as a supervised learner
        self.superpolicy = create_robust_keras_supervised_learner(self.policy)  # linked by reference
        assert mode in ['bc', 'dagger']
        self.mode = mode
        self.expert = expert
        if self.expert is None:
            self.mode = 'bc'
            print('No expert is provided. Switch to behavior cloning mode.')

        # for data aggregation
        self.dataset = Dataset(max_n_batches=max_n_batches,
                               max_n_samples=max_n_samples)  # number of trajectory
        if init_ro is not None:
            assert isinstance(init_ro, Dataset)
            self.dataset.extend(init_ro)
        self._n_pretrain_itrs = n_pretrain_itrs

    def _update(self, **kwargs):
        xs = self.dataset['obs_short']
        if self.mode=='bc':
            ys = self.dataset['acs']
        if self.mode=='dagger':
            ys = self.expert(xs)
        return self.superpolicy.update(xs=xs, ys=ys, **kwargs)

    def pretrain(self, gen_ro=None, **kwargs):
        if (gen_ro is not None) and (self.expert is not None):
            for _ in range(self._n_pretrain_itrs):
                ros, _ = gen_ro(PolicyAgent(self.expert))
                self.dataset.append(merge_ros(ros))
        return self._update(**kwargs)

    def update(self, ros, agents):
        self.dataset.append(merge_ros(ros))
        return self._update()

    def get_policy(self):
        return self.policy

    def agent(self, mode):
        if self.mode=='bc':
            return PolicyAgent(self.expert)
        elif self.mode=='dagger':
            return PolicyAgent(self.policy)
        else:
            raise NotImplementedError
