import copy
import numpy as np
from functools import partial
from rl.algorithms import GeneralizedPolicyGradient
from rl.algorithms import utils as au
from rl import online_learners as ol
from rl.core.datasets import Dataset
from rl.core.utils.misc_utils import timed, zipsame
from rl.core.utils import logz
from rl.core.utils.mvavg import PolMvAvg
from rl.core.function_approximators import online_compatible
from rl.core.function_approximators.supervised_learners import SupervisedLearner

from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradient




def process_ro(ro, vfn, lambd):
    ro = copy.deepcopy(ro)
    for r in ro:
        v = np.squeeze(vfn(r.obs))
        r.rws[1:-1] = r.rws[1:-1]+(1-lambd)*(lambd**np.arange(len(r.obs)-2))*v[1:-1]
        r.rws[0] -= v[0]
        # r.rws*= lambd
    return ro


class ValueBasedAENew(ValueBasedAE):

    def __init__(self, *args, ref_vfn=None, ref_lambd=None, **kwargs):
        self.ref_vfn =ref_vfn
        self.ref_lambd = ref_lambd
        super().__init__(*args, **kwargs)

    def advs(self, ro, **kwargs):
        ro = process_ro(ro, self.ref_vfn, self.ref_lambd)
        return super().advs(ro, **kwargs)


class GeneralizedPolicyGradientNew(GeneralizedPolicyGradient):
    """ Use max_k V^k as the value function. It overwrites the behavior policy. """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Create oracle (correction term).
        self.ae_cor = ValueBasedAENew(self.policy, copy.deepcopy(self.vfn.vfns[0]),
                ref_vfn = self.ae.vfn, ref_lambd=self.ae.lambd,
                gamma=self.ae.gamma, delta=self.ae.delta, lambd=0.9, horizon=self.ae.horizon, use_is='one', max_n_batches=self.ae.max_n_batches,)
        self.oracle_cor = ValueBasedPolicyGradient(self.policy, self.ae_cor)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                for k, expert in enumerate(self.experts):
                    pi_exp = lambda ob, t, done: expert(ob)
                    ro = gen_ro(pi_exp, logp=expert.logp)
                    self.aes[k].update(ro)
                    self.policy.update(ro['obs_short'])
                    #if self._use_policy_as_expert and k==len(self.experts)-1:
                    #    self.oracle.update(ro, update_vfn=False, policy=self.policy)
                if self._use_policy_as_expert:
                    self.ae_cor.update(ro) # XXX
        self._reset_pi_ro()

    def update(self, ro):
        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        with timed('Update oracle'):
            # Split ro into two phases
            rollouts = ro.to_list()
            ro_mix = [rollouts[i] for i in self._ind_ro_mix]
            ro_pol = [rollouts[i] for i in self._ind_ro_pol]
            assert (len(ro_mix)+len(ro_pol))==len(rollouts)
            ro_exps = [ [] for _ in range(len(self.experts))]
            for r, t, s, k in zipsame(ro_mix, self._t_switch, self._scale, self._k_star):
                assert len(r)>=t  # t >= 1
                if not self._use_policy_as_expert or k<len(self.experts)-1:
                    r = r[t-1:] # we take one more time step
                r.scale = s
                ro_exps[k].append(r)
            if self._use_policy_as_expert:
                ro_pol += ro_exps[-1]
                del ro_exps[-1]
            ro_exps = [Dataset(ro_exp) for ro_exp in ro_exps]
            ro_pol = Dataset(ro_pol)
            # update oracle
            self.oracle.update(ro_pol, update_vfn=False, policy=self.policy)
            self.oracle_cor.update(ro_pol, update_vfn=True, policy=self.policy)
            # update value functions
            EV0, EV1 = [], []
            for k, ro_exp in enumerate(ro_exps):
                if len(ro_exp)>0:
                    _, ev0, ev1 = self.aes[k].update(ro_exp)
                    EV0.append(ev0)
                    EV1.append(ev1)
            if self._use_policy_as_expert:
                _, ev0, ev1 = self.aes[-1].update(ro_pol)

            # for adaptive sampling
            self._avg_n_steps.update(np.mean([len(r) for r in ro_pol]))

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.policy.variable)
            g += self.oracle_cor.grad(self.policy.variable) * self.ae.lambd

        with timed('Policy update'):
            if isinstance(self.learner, ol.FisherOnlineOptimizer):
                if self._optimizer=='trpo_wl':  # use also the loss function
                    self.learner.update(g, ro=ro, policy=self.policy, loss_fun=self.oracle.fun)
                else:
                    self.learner.update(g, ro=ro, policy=self.policy)
            else:
                self.learner.update(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        if self._use_policy_as_expert:
            logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
            logz.log_tabular('ExplainVarianceAfter(AE)', ev1)
        logz.log_tabular('MeanExplainVarianceBefore(AE)', np.mean(EV0))
        logz.log_tabular('MeanExplainVarianceAfter(AE)', np.mean(EV1))
        logz.log_tabular('NumberOfExpertRollouts', np.sum([len(ro) for ro in ro_exps]))
        logz.log_tabular('NumberOfLearnerRollouts', len(ro_pol))

        # reset
        self._reset_pi_ro()
        self._itr+=1
