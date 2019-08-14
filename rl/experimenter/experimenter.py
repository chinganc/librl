import functools, copy
import time, os
import numpy as np
from rl.algorithms import Algorithm
from rl.core.utils.misc_utils import safe_assign, timed
from rl.core.utils import logz
from rl.core.utils.mp_utils import Worker, JobRunner

class Experimenter:

    def __init__(self, alg, mdp, ro_kwargs, n_processes=1, min_ro_per_process=1):
        """
            ro_kwargs is a dict with keys, 'min_n_samples', 'max_n_rollouts'
        """
        self.alg = safe_assign(alg, Algorithm)
        self.mdp = mdp
        self.ro_kwargs = ro_kwargs

        self._n_processes = n_processes
        self._min_ro_per_process = min_ro_per_process

        self._n_samples = 0  # number of data points seen
        self._n_rollouts = 0
        self.best_policy = copy.deepcopy(self.alg.policy)
        self.best_performance = -float('Inf')

    def gen_ro(self, agent, prefix='', to_log=False, eval_mode=False):
        """ Run the agent in the mdp and return rollout statistics as a Dataset
            and the agent that collects it.
        """
        if self._n_processes>1: # parallel data collection
            # determine ro_kwargs and number of jobs
            if self.ro_kwargs['max_n_rollouts'] is None:
                M_r = None
                N = self._n_processes
            else:
                M_r = max(1, self._min_ro_per_process)
                N = int(np.ceil(self.ro_kwargs['max_n_rollouts']/M_r))

            if self.ro_kwargs['min_n_samples'] is None:
                m_s = None
            else:
                m_s = int(np.ceil(self.ro_kwargs['min_n_samples']/N))
            ro_kwargs = {'min_n_samples': m_s, 'max_n_rollouts': M_r}

            # start data collection.
            job = ((agent,), ro_kwargs)
            [self._job_runner.put(job) for _ in range(N)]
            res = self._job_runner.aggregate(N)
            ros, agents = [r[0] for r in res], [r[1] for r in res]
        else:
            ro, agent = self.mdp.run(agent, **self.ro_kwargs)
            ros, agents = [ro], [agent]

        # Log
        ro = functools.reduce(lambda x,y: x+y, ros)
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

        return ros, agents

    def run(self, n_itrs, pretrain=True,
            save_freq=None, eval_freq=None, final_eval=False, final_save=True):

        eval_policy = eval_freq is not None
        save_policy = save_freq is not None

        # Start the processes.
        if self._n_processes>1:
            workers = [Worker(method=self.mdp.run) for _ in range(self._n_processes)]
            self._job_runner = JobRunner(workers)


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
                ros, agents = self.gen_ro(self.alg.agent('behavior'), to_log=not eval_policy)
            self.alg.update(ros, agents)

            if save_policy:
                if itr % save_freq == 0:
                    self._save_policy(self.alg.policy, itr)
            # dump log
            logz.dump_tabular()

        # Save the final policy.
        if final_save:
            self._save_policy(self.alg.policy, n_itrs)
            self._save_policy(self.best_policy, 'best')

        if final_eval:
            self.gen_ro(self.agent('target'), to_log=True, eval_mode=True)
            logz.dump_tabular()


        # Close the processes.
        if self._n_processes>1:
            self._job_runner.stop()

    def _save_policy(self, policy, suffix):
        path = os.path.join(logz.LOG.output_dir,'saved_policies')
        name = policy.name+'_'+str(suffix)
        policy.save(path, name=name)


