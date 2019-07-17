import time
import numpy as np
from rl.core.datasets import Dataset


class Rollout(object):
    """ A container for storing statistics along a trajectory. """

    def __init__(self, obs, acs, rws, sts, done, logp=None):
        # obs, acs, rws, sts, lps are lists of vals
        assert len(obs) == len(sts) == len(rws)
        assert len(obs) == len(acs) + 1
        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.rws = np.array(rws)
        self.sts = np.array(sts)
        self.dns = np.zeros((len(self)+1,))
        self.dns[-1] = float(done)
        if logp is None:  # viewed as deterministic
            self.lps = np.zeros(self.acs.shape)
        else:
            self.lps = logp(self.obs[:-1], self.acs)

    @property
    def obs_short(self):
        return self.obs[:-1,:]

    @property
    def sts_short(self):
        return self.sts[:-1,:]

    @property
    def rws_short(self):
        return seif.rws[:-1,:]

    @property
    def done(self):
        return bool(self.dns[-1])

    def __len__(self):
        return len(self.acs)


def get_state(env):
    # try to retrieve the underlying state, if available
    if hasattr(env, 'env'):
        return env.env.state  # openai gym env, which is a TimeLimit object
    elif hasattr(env, 'state'):
        return env.state
    else:
        return None


def generate_rollout(pi, logp, env, v_end=None,
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
    if v_end is None:
        v_end = lambda ob, done : 0.
    min_n_samples = min_n_samples or float('Inf')
    max_n_rollouts = max_n_rollouts or float('Inf')
    max_rollout_len = max_rollout_len or float('Inf')
    max_rollout_len = min(env._max_episode_steps, max_rollout_len)
    n_samples = 0
    rollouts = []
    while True:
        animate_this_rollout = len(rollouts) == 0 and with_animation
        ob = env.reset()  # observation
        st = get_state(env)  # env state
        obs, acs, rws, sts = [], [], [], []
        steps = 0  # steps so far
        while True:
            if animate_this_rollout:
                env.render()
                time.sleep(0.05)
            obs.append(ob)
            sts.append(st)
            ac = pi(ob) # apply action and get to the next state
            acs.append(ac)
            ob, rw, dn, _ = env.step(ac)  # current reward, next ob and dn
            rws.append(rw)
            st = get_state(env)
            steps += 1
            if dn or steps >= max_rollout_len:
                break # due to steps limit or entering an absorbing state
        # save the terminal state/observation/reward
        sts.append(st)
        obs.append(ob)
        rws.append(v_end(ob, dn))
        # end of one rollout
        rollout = Rollout(obs=obs, acs=acs, rws=rws, sts=sts, done=dn, logp=logp)
        rollouts.append(rollout)
        n_samples += len(rollout)
        if (n_samples >= min_n_samples) or (len(rollouts) >= max_n_rollouts):
            break
    ro = Dataset(rollouts)
    return ro
