# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import copy
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.oracle import rlOracle
from rl.core.oracles import tfLikelihoodRatioOracle
from rl.core.function_approximators.policies import tfPolicy
from rl.core.utils.tf2_utils import ts_to_array
from rl.core.utils.misc_utils import flatten, zipsame
from rl.core.datasets import Dataset

class tfValueBasedPolicyGradient(rlOracle):
    """ A wrapper of tfLikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """
    def __init__(self, policy, ae,
                 use_is='one', avg_type='sum',
                 biased=False, use_log_loss=False, normalized_is=False):
        assert isinstance(ae, ValueBasedAE)
        self._ae = ae
        # define the internal oracle
        assert isinstance(policy, tfPolicy)
        self._policy_t = copy.deepcopy(policy)  # just a template
        def ts_logp_fun():
            return  self._policy_t.ts_logp(self.ro['obs_short'], self.ro['acs'])
        self._or = tfLikelihoodRatioOracle(
                    ts_logp_fun, self._policy_t.ts_variables,
                    biased=biased, # basic mvavg
                    use_log_loss=use_log_loss, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one', 'multi', None]
        self._use_is = use_is  # use importance sampling for polcy gradient
        assert avg_type in ['avg', 'sum']
        self._avg_type = avg_type
        self._scale = None
        self._ro = None

    def fun(self, x):
        return self._or.fun(self._policy_t.unflatten(x)) * self._scale

    def grad(self, x):
        return self._or.grad(self._policy_t.unflatten(x)) * self._scale

    def update(self, ro, policy, update_nor=True):
        # Sync policies' parameters.
        self._policy_t.assign(policy) # NOTE new tf.Variables may be created in assign!!
        # Compute adv.
        self._ro = ro
        advs, vfns = self._ae.advs(self.ro, use_is=self._use_is)
        adv = np.concatenate(advs)
        self._scale = 1.0 if self._avg_type=='avg' else len(adv)/len(advs)
        # Update the loss function.
        if self._or._use_log_loss is True:
            #  - E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * adv(ob, ac) ]
            if self._use_is:  # consider importance weight
                w_or_logq = np.concatenate(self._ae.weights(ro, policy=self.policy))
            else:
                w_or_logq = np.ones_like(adv)
        else:  # False or None
            #  - E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * adv(ob, ac) ]
            assert self._use_is in ['one', 'multi']
            w_or_logq = ro['lps']
        # Update the tfLikelihoodRatioOracle.
        self._or.update(-adv, w_or_logq, update_nor=update_nor, # loss is negative reward
                        ts_var=self._policy_t.ts_variables) # NOTE sync
        # Update the value function at the end, so it's unbiased.
        return self._ae.update(ro)

    @property
    def ro(self):
        return self._ro



class tfValueBasedExpertGradient(rlOracle):
    """ A wrapper of tfLikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """
    def __init__(self, policy, ae,
                 use_is='one', normalized_is=False):
        assert isinstance(ae, ValueBasedAE)
        self._ae = ae
        # define the internal oracle
        assert isinstance(policy, tfPolicy)
        self._policy_t = copy.deepcopy(policy)  # just a template
        def ts_logp_fun_exp():
            return  self._policy_t.ts_logp(self._ro_or['obs_short'], self._ro_or['acs'])
        self._or = tfLikelihoodRatioOracle(
                    ts_logp_fun_exp, self._policy_t.ts_variables,
                    biased=False, # basic mvavg
                    use_log_loss=False, normalized_is=normalized_is)
        # another oracle for control variate's bias
        def ts_logp_fun_pol():
            return  self._policy_t.ts_logp(self._ro_cv['obs_short'], self._ro_cv['acs'])
        self._cv = tfLikelihoodRatioOracle(
                    ts_logp_fun_pol, self._policy_t.ts_variables,
                    biased=False, # basic mvavg
                    use_log_loss=False, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one'] #, 'multi']
        self._use_is = use_is  # use importance sampling for polcy gradient
        self._scale_or = 0.
        self._scale_cv = 0.
        self._ro_or = None
        self._ro_cv = None

    def fun(self, x):
        f1 = 0. if self._ro_exp is None else self._or.fun(self._policy_t.unflatten(x))*self._scale_or
        f2 = 0. if self._ro_pol is None else self._cv.fun(self._policy_t.unflatten(x))*self._scale_cv
        return f1+f2

    def grad(self, x):
        g1 = np.zeros_like(x) if self._ro_or is None \
                else self._or.grad(self._policy_t.unflatten(x))*self._scale_or
        g2 = np.zeros_like(x) if self._ro_cv is None \
                else self._cv.grad(self._policy_t.unflatten(x))*self._scale_cv
        print(np.linalg.norm(g1), np.linalg.norm(g2))
        return g1+g2

    def update(self, ro_exp=None, ro_pol=None, policy=None, update_nor=True):
        """ Need to provide either `ro_exp` or `ro_pol`, and `policy`.

            `ro_exp` is used to compute an unbiased but noisy estimate of

                E_{pi}[\nabla \pi(s,a) \hat{A}_{\pi^*}(s,a)]

            when \hat{A}_{\pi^*} given by `self._or` is unbiased.

            `ro_pol` provides a biased gradient which can be used as a control
            variate (when `ro_exp` is provided) or just to define a biased
            oracle.
        """
        assert (ro_exp is not None) or (ro_pol is not None)
        assert policy is not None

        # Sync policies' parameters.
        self._policy_t.assign(policy) # NOTE new tf.Variables may be created in assign!!
        # Update the oracles
        n_rollouts = len(ro_exp) if ro_pol is None else len(ro_pol)
        self._ro_or = None
        if ro_exp is not None:
            # compute adv
            if len(ro_exp)>0:
                advs, _ = self._ae.advs(ro_exp, use_is=self._use_is)
                advs = [a[0:1] for a in advs]
                adv = np.concatenate(advs)
                if ro_pol is not None:  # compute the control variate
                    advs_cv, _ = self._ae.advs(ro_exp, use_is=self._use_is, lambd=0.)
                    advs_cv = [a[0:1] for a in advs_cv]
                    adv -= np.concatenate(advs_cv)
                adv *= ro_exp['scale'][0]  # account for random switching
                logq = np.concatenate([r.lps[0:1] for r in ro_exp])
                # update noisy oracle
                self._scale_or = len(adv)/n_rollouts
                self._or.update(-adv, logq, update_nor=update_nor, # loss is negative reward
                                ts_var=self._policy_t.ts_variables) # NOTE sync
                self._ro_or = Dataset([r[0:1] for r in ro_exp])  # for defining logp

        self._ro_cv = None
        if ro_pol is not None:
            # update biased oracle
            advs, _ = self._ae.advs(ro_pol, use_is=self._use_is, lambd=0.)
            adv = np.concatenate(advs)
            self._scale_cv = len(adv)/n_rollouts
            logq = ro_pol['lps']

            self._cv.update(-adv, logq, update_nor=update_nor, # loss is negative reward
                            ts_var=self._policy_t.ts_variables) # NOTE sync
            self._ro_cv = ro_pol  # for defining logp

        # Update the value function at the end, so it's unbiased.
        if ro_exp is not None:
            return self._ae.update(ro_exp)
        else:  # when biased gradient is used
            return self._ae.update(ro_pol)

    @property
    def ro(self):
        return self._ro_or + self._ro_cv
