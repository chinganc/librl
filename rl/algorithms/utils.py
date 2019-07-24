import numpy as np

def natural_t(horizon, gamma):
    if horizon < float('Inf'):
        p0 = gamma**np.arange(horizon)
        sump0 = np.sum(p0)
        p0 = p0/sump0
        ind = np.random.multinomial(1,p0)
        t_switch =  np.where(ind==1)[0][0]+1
        p = p0[t_switch-1]
    else:
        t_switch = np.random.geometric(p=1-gamma)[0]
        p = gamma**t_switch*(1-gamma)
    return  _normalize_sampling(t_switch, p, horizon, gamma)

def cyclic_t(rate, horizon, gamma):
    if getattr(cyclic_t, '_itr', None) is None:
        cyclic_t._itr = 0
    assert horizon < float('Inf')
    t_switch = (int(rate*cyclic_t._itr)%horizon)+1
    p = 1./horizon
    cyclic_t._itr +=1
    return _correct_scale(t_switch, p, horizon, gamma)

def exponential_t(beta, horizon, gamma):
    t_switch = int(np.ceil(np.random.exponential(beta)))
    p = 1./beta*np.exp(-t_switch/beta)
    return _normalize_sampling(t_switch, p, horizon, gamma)

def _normalize_sampling(t_switch, p, horizon, gamma):
    # correct for potential discount factor
    t_switch = min(t_switch, horizon-1)
    if horizon < float('Inf'):
        p0 = gamma**np.arange(horizon)
        sump0 = np.sum(p0)
        p0 = p0/sump0
        pp = p0[t_switch]
    else:
        sump0 = 1/(1-gamma)
        pp = gamma**t_switch*(1-gamma)
    scale = (pp/p)*sump0
    return t_switch, scale


