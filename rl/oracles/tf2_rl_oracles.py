import numpy as np
import copy
from rl.adv_estimators.advantage_estimator import ValueBasedAdvFuncApp
from rl.oracles.oracle import rlOracle
from rl.core.oracles import tfLikelihoodRatioOracle
from rl.core.function_approximators.policies import tfPolicy


class tfValueBasedPolicyGradient(rlOracle):
    """ A wrapper of tfLikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """
    def __init__(self, policy, ae,
                 use_is='one', avg_type='avg',
                 nor=None, biased=False, use_log_loss=False, normalized_is=False):
        assert isinstance(ae, ValueBasedAdvFuncApp)
        self._ae = ae
        # define the internal oracle
        assert isinstance(policy, tfPolicy)
        self._policy_t = copy.deepcopy(policy)  # just a template
        def ts_logp_fun (ts_vars):
            self._policy_t.ts_variables = ts_vars  # just assign the variables
            return self._policy_t.ts_logp(self._ro.obs, self._ro.acs)
        self._or = tfLikelihoodRatioOracle(ts_logp_fun,
                    nor=nor, biased=biased, # basic mvavg
                    use_log_loss=use_log_loss, normalized_is=normalize_is)
        # some configs for computing gradients
        assert use_is in ['one', 'multi', None, False]
        self._use_is = use_is  # use importance sampling for polcy gradient
        assert avg_type in ['avg', 'sum']
        self._avg_type = avg_type
        self._scale = None
        self._ro = None

    def fun(self, policy):
        return self._or.fun(policy.ts_variables) * self._scale

    def grad(self, policy):
        return self._or.grad(policy.ts_variables) * self._scale

    def update(self, ro):
        # Compute adv.
        self._ro = ro
        advs, vfns = self._ae.advs(self.ro, use_is=self._use_is)
        adv = np.concatenate(advs)
        self._scale = 1.0 if self._avg_type=='avg' else len(adv)/len(advs)
        # Update the loss function.
        if self._or._use_log_loss is True:
            #  - E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * adv(ob, ac) ]
            if self.use_is:  # consider importance weight
                w_or_logq = np.concatenate(self._ae.weights(ro, policy=self.policy))
            else:
                w_or_logq = np.ones_like(adv)
        else:  # False or None
            #  - E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * adv(ob, ac) ]
            assert self.use_is in ['one', 'multi']
            w_or_logq = ro.lps
        # Update the tfLikelihoodRatioOracle.
        self._or.update(-adv, w_or_logq, update_nor=True)  # loss is negative reward
        # Update the value function at the end, so it's unbiased.
        self._ae.update(ro)

    @property
    def ro(self):
        return self._ro
