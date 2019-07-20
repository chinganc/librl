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
                 use_is='one', avg_type='avg',
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

    def fun(self, policy):
        return self._or.fun(policy.variables) * self._scale

    def grad(self, policy):
        return self._or.grad(policy.variables) * self._scale

    def update(self, ro, policy):
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
        self._or.update(-adv, w_or_logq, update_nor=True, # loss is negative reward
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
            return  self._policy_t.ts_logp(self.ro_exp['obs_short'], self.ro_exp['acs'])
        self._or = tfLikelihoodRatioOracle(
                    ts_logp_fun_exp, self._policy_t.ts_variables,
                    biased=False, # basic mvavg
                    use_log_loss=False, normalized_is=normalized_is)
        # another oracle for control variate's bias
        def ts_logp_fun_pol():
            return  self._policy_t.ts_logp(self.ro_pol['obs_short'], self.ro_pol['acs'])
        self._cv = tfLikelihoodRatioOracle(
                    ts_logp_fun_pol, self._policy_t.ts_variables,
                    biased=False, # basic mvavg
                    use_log_loss=False, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one', 'multi']
        self._use_is = use_is  # use importance sampling for polcy gradient
        self._or_scale = None
        self._cv_scale = None
        self.ro_exp = None
        self.ro_pol = None

    def fun(self, policy):
        return self._or.fun(policy.variables)*self._or_scale + self._cv.fun(policy.variables)*self._cv_scale

    def grad(self, policy):
        #import pdb; pdb.set_trace()
        print(self.ro_exp.n_samples,self.ro_pol.n_samples)
        g1= self._or.grad(policy.variables)*self._or_scale
        g2= self._cv.grad(policy.variables)*self._cv_scale
        return g1+g2

    def update(self, ro_exp=None, ro_pol=None, policy=None):
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
        if ro_exp is not None:
            # compute adv
            if len(ro_exp)>0:
                advs, _ = self._ae.advs(ro_exp, use_is=self._use_is)
                advs = [a[0:1]*s for a, s in zipsame(advs, ro_exp['scale'])]
                adv = np.concatenate(advs)
                self.ro_exp = Dataset([r[0:1] for r in ro_exp])  # NOTE needs for defning logp
                if ro_pol is not None:  # compute the control variate
                    advs_cv, _ = self._ae.advs(ro_exp, use_is=self._use_is, lambd=0.)
                    advs_cv = [a[0:1]*s for a, s in zipsame(advs_cv, ro_exp['scale'])]
                    adv -= np.concatenate(advs_cv)
                logqs = [r.lps[0:1] for r in ro_exp]
                logq = np.concatenate(logqs)
                # update noisy oracle
                self._or_scale = len(adv)/len(advs)
                self._or.update(-adv, logq, update_nor=True, # loss is negative reward
                                ts_var=self._policy_t.ts_variables) # NOTE sync
            else:
                self._or._ts_var = self._policy_t.ts_variables
                self._or_scale = 0.

        if ro_pol is not None:
            # update biased oracle
            advs, _ = self._ae.advs(ro_pol, use_is=self._use_is, lambd=0.)
            adv = np.concatenate(advs)
            self._cv_scale = len(adv)/len(advs)
            logq = ro_pol['lps']
            self._cv.update(-adv, logq, update_nor=True, # loss is negative reward
                            ts_var=self._policy_t.ts_variables) # NOTE sync
            self.ro_pol = ro_pol

        # Update the value function at the end, so it's unbiased.
        if ro_exp is not None:
            if len(ro_exp)>0:
                return self._ae.update(ro_exp)
            return None, None, None
        else:  # when biased gradient is used
            return self._ae.update(ro_pol)

    @property
    def ro(self):
        return self.ro_exp + self.ro_pol
