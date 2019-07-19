import numpy as np
from rl.adv_estimators.performance_estimate import PerformanceEstimate as PE


gamma=1.0
delta=0.9
lambd=1
pe = PE(gamma, lambd, delta)
c = np.arange(10)
V = np.arange(10)*2

#td
adv = pe.adv(c, V, lambd=0, done=True)
assert np.isclose(adv[0],  c[0]+delta*V[1]-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])

#monte carlo
adv = pe.adv(c, V, lambd=1., done=True)
q = np.sum(delta**np.arange(10)*c)
assert np.isclose(adv[0],  q-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])


#test weights
w = np.random.random(c.shape)
w = w[:-1]
adv = pe.adv(c, V, w=w, lambd=0, done=True)
assert np.isclose(adv[0],  c[0]+delta*V[1]-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])

