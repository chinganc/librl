# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from rl.core.function_approximators.policies.policy import Policy

class  myPolicy(Policy):

    def __init__(self, x_shape, y_shape, fun, **kwargs):
        super().__init__(x_shape, y_shape)
        self._fun = fun

    def predict(self, xs, **kwargs):
        return self._fun(xs[:,:-1])

    @property
    def variable(self):
        return None

    @variable.setter
    def variable(self, val):
        pass

from hw_expert.load_policy import load_policy

def load_hw_expert():
    policy_fn = load_policy('hw_expert/Humanoid-v2.pkl')
    policy = myPolicy((377,),(17,), policy_fn)
    return policy

