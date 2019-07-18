import time
import numpy as np
from rl.core.datasets import Dataset


class MDP:
    """ A wrapper for gym env. """

    def __init__(self, env, gamma=1.0, horizon=None, v_end=None):
        self.env = env
        self.gamma = gamma
        self.horizon = horizon
        self.v_end = v_end or (lambda ob, done : 0.)

    def rollout(self, pi, logp=None,
                min_n_samples=None, max_n_rollouts=None,
                with_animation=False):
        # pi takes (ob, time) as input
        if logp is None:  # viewed as deterministic
            logp = lambda obs, acs: np.zeros((len(acs),1))
        return  generate_rollout(pi, logp, self.env, v_end=self.v_end,
                              min_n_samples=min_n_samples, max_n_rollouts=max_n_rollouts,
                              max_rollout_len=self.horizon, with_animation=with_animation)

    @property
    def ob_shape(self):
        return self.env.observation_space.shape

    @property
    def ac_shape(self):
        return self.env.action_space.shape

    def rw_bounds(self):  #TODO
        raise NotImplementedErrort

class Rollout(object):
    """ A container for storing statistics along a trajectory. """

    def __init__(self, obs, acs, rws, sts, done, logp):
        # obs, acs, rws, sts, lps are lists of vals
        # logp is a callable function
        assert len(obs) == len(sts) == len(rws)
        assert len(obs) == len(acs) + 1
        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.rws = np.array(rws)
        self.sts = np.array(sts)
        self.dns = np.zeros((len(self)+1,))
        self.dns[-1] = float(done)
        self.tms = np.arange(len(obs))  # for convenience
        self.lps = logp(self.obs[:-1], self.acs)

    @property
    def obs_short(self):
        return self.obs[:-1,:]

    @property
    def sts_short(self):
        return self.sts[:-1,:]

    @property
    def rws_short(self):
        return self.rws[:-1,:]

    @property
    def done(self):
        return bool(self.dns[-1])

    def __len__(self):
        return len(self.acs)



def generate_rollout(pi, logp, env, v_end,
                     full_obs=False,
                     min_n_samples=None, max_n_rollouts=None,
                     max_rollout_len=None,
                     with_animation=False):
    """ Collect rollouts until we have enough samples.

        All rollouts are COMPLETE in that they never end prematurely: they end
        either when done is true or max_rollout_len is reached.

        Args:
            pi: a function that maps ob to ac
            logp: either None or a function that maps (obs, acs) to log probabilities
            env: a gym environment
            v_end: the terminal value when the episoide ends (a callable function of ob and done)
            max_rollout_len: the maximal length of a rollout (i.e. the problem's horizon)
            min_n_samples: the minimal number of samples to collect
            max_n_rollouts: the maximal number of rollouts
            with_animation: display animiation of the first rollout
    """

    assert (min_n_samples is not None) or (max_n_rollouts is not None)  # so we can stop
    min_n_samples = min_n_samples or float('Inf')
    max_n_rollouts = max_n_rollouts or float('Inf')
    max_rollout_len = max_rollout_len or float('Inf')
    max_rollout_len = min(env._max_episode_steps, max_rollout_len)
    n_samples = 0
    rollouts = []
    # try to retrieve the underlying state, if available
    get_state = lambda: None
    if hasattr(env, 'env'):
        if hasattr(env.env, 'state'):
            get_state = lambda: env.env.state  # openai gym env, which is a TimeLimit object
    elif hasattr(env, 'state'):
        get_state = lambda: env.state

    def step(ac, tm):
        ob, rw, dn, info = env.step(ac)  # current reward, next ob and dn
        st = get_state()
        if st is None:
            st = ob
 
        # ob = np.concatenate([ob.reshape([-1,1]), np.array((tm,))[:,None ]])
        ob = np.concatenate([ob.reshape([-1]), np.array((tm,))])

        return st, ob, rw, dn, info

    def reset(tm):
        ob = env.reset()  # observation
        st = get_state()  # env state
        ob = np.concatenate([ob.reshape([-1]), np.array((tm,))])
        return st, ob

    # start rollout
    while True:
        animate_this_rollout = len(rollouts) == 0 and with_animation
        obs, acs, rws, sts = [], [], [], []
        tm = 0  # time step
        dn = False
        st, ob = reset(tm)
        # each trajectory
        while True:
            if animate_this_rollout:
                env.render()
                time.sleep(0.05)
            ac = pi(ob, tm, dn) # apply action and get to the next state
            obs.append(ob)
            sts.append(st)
            acs.append(ac)
            st, ob, rw, dn, _ = step(ac, tm)
            rws.append(rw)
            tm += 1
            if dn or tm >= max_rollout_len:
                break # due to steps limit or entering an absorbing state
        # save the terminal state/observation/reward
        sts.append(st)
        obs.append(ob)
        rws.append(v_end(ob, dn))  # terminal reward
        # end of one rollout
        rollout = Rollout(obs=obs, acs=acs, rws=rws, sts=sts, done=dn, logp=logp)
        rollouts.append(rollout)
        n_samples += len(rollout)
        if (n_samples >= min_n_samples) or (len(rollouts) >= max_n_rollouts):
            break
    ro = Dataset(rollouts)
    return ro
