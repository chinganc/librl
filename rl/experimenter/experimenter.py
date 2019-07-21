import functools
import time, os
import numpy as np
from rl.algorithms import Algorithm
from rl.core.utils.misc_utils import safe_assign, timed
from rl.core.utils import logz


class Experimenter:

    def __init__(self, alg, env, ro_kwargs):
        """
        ro_kwargs is a dict with keys, 'min_n_samples', 'max_n_rollouts'
        """
        self.alg = safe_assign(alg, Algorithm)
        self._gen_ro = functools.partial(env.rollout, **ro_kwargs)
        self._n_samples = 0  # number of data points seen
        self._n_rollouts = 0

    def gen_ro(self, pi, logp=None, prefix='', to_log=False, eval_mode=False):
        ro = self._gen_ro(pi, logp)
        if not eval_mode:
            self._n_rollouts += len(ro)
            self._n_samples += ro.n_samples
        if to_log:
            # current ro
            sum_of_rewards = [rollout.rws.sum() for rollout in ro]
            rollout_lens = [len(rollout) for rollout in ro]
            n_samples = sum(rollout_lens)
            logz.log_tabular(prefix + "NumSamples", n_samples)
            logz.log_tabular(prefix + "NumberOfRollouts", len(ro))
            logz.log_tabular(prefix + "MeanSumOfRewards", np.mean(sum_of_rewards))
            logz.log_tabular(prefix + "StdSumOfRewards", np.std(sum_of_rewards))
            logz.log_tabular(prefix + "MaxSumOfRewards", np.max(sum_of_rewards))
            logz.log_tabular(prefix + "MinSumOfRewards", np.min(sum_of_rewards))
            logz.log_tabular(prefix + "MeanRolloutLens", np.mean(rollout_lens))
            logz.log_tabular(prefix + "StdRolloutLens", np.std(rollout_lens))
            # total
            logz.log_tabular(prefix + 'TotalNumberOfSamples', self._n_samples)
            logz.log_tabular(prefix + 'TotalNumberOfRollouts', self._n_rollouts)
        return ro

    def run(self, n_itrs, pretrain=True,
            save_freq=None, eval_freq=None, final_eval=False):

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
                    with timed('Evaluate env rollouts'):
                        self.gen_ro(self.alg.pi, to_log=True, eval_mode=True)

            with timed('Generate env rollouts'):
                ro = self.gen_ro(self.alg.pi_ro, logp=self.alg.logp, to_log=not eval_policy)
            self.alg.update(ro)

            if save_policy:
                if itr % save_freq == 0:
                    self._save_policy(itr)
            # dump log
            logz.dump_tabular()

        # save final policy
        self._save_policy(n_itrs)
        if final_eval:
            self.gen_ro(self.alg.pi, to_log=True, eval_mode=True)
            logz.dump_tabular()


    def _save_policy(self, itr):
        path =os.path.join(logz.LOG.output_dir,'saved_policies')
        name = self.alg.policy.name+'_'+str(itr)
        self.alg.policy.save(path, name=name)


