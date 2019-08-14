import functools, copy
import time, os
import numpy as np
from rl.algorithms import Algorithm
from rl.core.utils.misc_utils import safe_assign, timed
from rl.core.utils import logz
from rl.core.utils.mp_utils import Worker, JobRunner

class Experimenter:

    def __init__(self, alg, mdp, ro_kwargs, n_processes=5):
        """
            ro_kwargs is a dict with keys, 'min_n_samples', 'max_n_rollouts'
        """
        self.alg = safe_assign(alg, Algorithm)
        self.mdp = mdp
        self.ro_kwargs = ro_kwargs
        self._n_processes = n_processes
        if n_processes>1:
            self._jobrunner = JobRunner([Worker() for _ in range(n_processes)])

        self._n_samples = 0  # number of data points seen
        self._n_rollouts = 0
        self.best_policy = copy.deepcopy(self.alg.policy)
        self.best_performance = -float('Inf')

    def gen_ro(self, agent, prefix='', to_log=False, eval_mode=False):
        """ Run the agent in the mdp and return rollout statistics as a Dataset
            and the agent that collects it.    
        """
        ro, agent = self.mdp.run(agent, **self.ro_kwargs)

        if not eval_mode:
            self._n_rollouts += len(ro)
            self._n_samples += ro.n_samples
        if to_log:
            # current ro
            gamma = self.mdp.gamma
            sum_of_rewards = [ ((gamma**np.arange(len(r.rws)))*r.rws).sum() for r in ro]
            performance = np.mean(sum_of_rewards)
            rollout_lens = [len(rollout) for rollout in ro]
            n_samples = sum(rollout_lens)
            logz.log_tabular(prefix + "NumSamples", n_samples)
            logz.log_tabular(prefix + "NumberOfRollouts", len(ro))
            logz.log_tabular(prefix + "MeanSumOfRewards", performance)
            logz.log_tabular(prefix + "StdSumOfRewards", np.std(sum_of_rewards))
            logz.log_tabular(prefix + "MaxSumOfRewards", np.max(sum_of_rewards))
            logz.log_tabular(prefix + "MinSumOfRewards", np.min(sum_of_rewards))
            logz.log_tabular(prefix + "MeanRolloutLens", np.mean(rollout_lens))
            logz.log_tabular(prefix + "StdRolloutLens", np.std(rollout_lens))
            # total
            logz.log_tabular(prefix + 'TotalNumberOfSamples', self._n_samples)
            logz.log_tabular(prefix + 'TotalNumberOfRollouts', self._n_rollouts)
            if performance >= self.best_performance:
                self.best_policy = copy.deepcopy(self.alg.policy)
                self.best_performance = performance
            logz.log_tabular(prefix + 'BestSumOfRewards', self.best_performance)

        return ro, agent

    def run(self, n_itrs, pretrain=True,
            save_freq=None, eval_freq=None, final_eval=False, final_save=True):

        eval_policy = eval_freq is not None
        save_policy = save_freq is not None

        start_time = time.time()
        if pretrain:
            self.alg.pretrain(functools.partial(self.gen_ro, to_log=False))

        # Main loop
        for itr in range(n_itrs):
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)

            if eval_policy:
                if itr % eval_freq == 0:
                    with timed('Evaluate policy performance'):
                        self.gen_ro(self.alg.agent('target'), to_log=True, eval_mode=True)

            with timed('Generate env rollouts'):
                ro, agent = self.gen_ro(self.alg.agent('behavior'), to_log=not eval_policy)
            self.alg.update(ro, agent)

            if save_policy:
                if itr % save_freq == 0:
                    self._save_policy(self.alg.policy, itr)
            # dump log
            logz.dump_tabular()

        # save final policy
        if final_save:
            self._save_policy(self.alg.policy, n_itrs)
            self._save_policy(self.best_policy, 'best')

        if final_eval:
            self.gen_ro(self.agent('target'), to_log=True, eval_mode=True)
            logz.dump_tabular()

    def _save_policy(self, policy, suffix):
        path = os.path.join(logz.LOG.output_dir,'saved_policies')
        name = policy.name+'_'+str(suffix)
        policy.save(path, name=name)


