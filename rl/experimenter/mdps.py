import time
import numpy as np
from rl.core.datasets import Dataset


class MDP:
    """ A wrapper for gym env. """
    def __init__(self, env, gamma=1.0, horizon=None, use_time_info=None, v_end=None):
        self.env = env
        self.gamma = gamma
        if horizon is None:
            horizon = float('Inf')
        self.horizon = horizon
        self.v_end = v_end or (lambda ob, done : 0.)  # terminal reward
        self.use_time_info = lambda t: t/self.horizon

    @property
    def ob_shape(self):
        return self.env.observation_space.shape

    @property
    def ac_shape(self):
        return self.env.action_space.shape

    def run(self, pi, logp=None,
                  min_n_samples=None, max_n_rollouts=None,
                  with_animation=False):
        """ `pi` takes (ob, time, done) as input"""
        if logp is None:  # viewed as deterministic
            logp = lambda obs, acs: np.zeros((len(acs),1))
        return generate_rollout(pi, logp, self.env, v_end=self.v_end,
                                use_time_info=self.use_time_info,
                                min_n_samples=min_n_samples,
                                max_n_rollouts=max_n_rollouts,
                                max_rollout_len=self.horizon,
                                with_animation=with_animation)


class Rollout(object):
    """ A container for storing statistics along a trajectory. """

    def __init__(self, obs, acs, rws, sts, done, logp):
        """
            `obs`, `acs`, `rws`, `sts`  are lists of floats
            `done` is bool
            `logp` is a callable function or an nd.array

            `obs`, `sts`, `rws` can be of length of `acs` or one element longer if they contain the
            terminal observation/state/reward.
        """
        assert len(obs)==len(sts)==len(rws)
        assert (len(obs) == len(acs)+1) or (len(obs)==len(acs))
        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.rws = np.array(rws)
        self.sts = np.array(sts)
        self.dns = np.zeros((len(self)+1,))
        self.dns[-1] = float(done)
        if isinstance(logp, np.ndarray):
            assert len(logp)==len(acs)
            self.lps = logp
        else:
            self.lps = logp(self.obs[:len(self)], self.acs)

    @property
    def obs_short(self):
        return self.obs[:len(self),:]

    @property
    def sts_short(self):
        return self.sts[:len(self),:]

    @property
    def rws_short(self):
        return self.rws[:len(self),:]

    @property
    def done(self):
        return bool(self.dns[-1])

    def __len__(self):
        return len(self.acs)

    def __getitem__(self, key):
        obs=self.obs[key]
        acs=self.acs[key]
        rws=self.rws[key]
        sts=self.sts[key]
        logp=self.lps[key]
        done = bool(self.dns[key][-1])
        return Rollout(obs=obs, acs=acs, rws=rws, sts=sts,
                       done=done,logp=logp)


def generate_rollout(pi, logp, env, v_end,
                     use_time_info=None,
                     min_n_samples=None, max_n_rollouts=None,
                     max_rollout_len=None,
                     with_animation=False):

    """ Collect rollouts until we have enough samples or rollouts.

        All rollouts are COMPLETE in that they never end prematurely, even when
        `min_n_samples` is reached. They end either when done is true or
        max_rollout_len is reached.

        Args:
            `pi`: a function that maps ob to ac
            `logp`: either None or a function that maps (obs, acs) to log
                  probabilities (called at end of each rollout)
            `env`: a gym environment
            `v_end`: the terminal value when the episoide ends (a callable function of ob and done)
            `use_time_info`: a function that maps time to desired features
            `max_rollout_len`: the maximal length of a rollout (i.e. the problem's horizon)
            `min_n_sample`s: the minimal number of samples to collect
            `max_n_rollouts`: the maximal number of rollouts
            `with_animation`: display animiation of the first rollout
    """

    assert (min_n_samples is not None) or (max_n_rollouts is not None)  # so we can stop
    min_n_samples = min_n_samples or float('Inf')
    max_n_rollouts = max_n_rollouts or float('Inf')
    max_rollout_len = max_rollout_len or float('Inf')
    max_rollout_len = min(env._max_episode_steps, max_rollout_len)
    n_samples = 0
    rollouts = []
    # try to retrieve the underlying state, if available
    get_state = None
    if hasattr(env, 'env'):
        if hasattr(env.env, 'state'):
            get_state = lambda: env.env.state  # openai gym env, which is a TimeLimit object
    elif hasattr(env, 'state'):
        get_state = lambda: env.state
    # whether to augment state/observation with time information
    if use_time_info is not None:
        post_process = lambda x,t: np.concatenate([x.flatten(), (use_time_info(t),)])
    else:
        post_process = lambda x,t :x

    def step(ac, tm):
        ob, rw, dn, info = env.step(ac)  # current reward, next ob and dn
        st = ob if get_state is None else get_state()
        # may need to augment observatino and state with time
        ob = post_process(ob, tm)
        st = post_process(st, tm)
        return st, ob, rw, dn, info

    def reset(tm):
        ob = env.reset()
        st = ob if get_state is None else get_state()
        ob = post_process(ob, tm)
        st = post_process(st, tm)
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
