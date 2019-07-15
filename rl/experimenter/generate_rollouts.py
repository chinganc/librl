import time
from rl.core.datasets import Dataset

class Rollout(object):
    """ A container for storing statistics along a trajectory. """

    def __init__(self, obs, acs, rws, sts, dns, logp=None):
        # obs, acs, rws, sts, lps are lists of vals
        assert len(obs) == len(sts)
        assert len(acs) == len(rws)
        assert len(obs) == len(acs) + 1
        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.rws = np.array(rws)
        self.sts = np.array(sts)
        self.dns = np.array(dns)
        if logp is None:  # viewed as deterministic
            self.lps = np.zeros(self.acs.shape)
        else:
            self.lps = logp(self.obs_short, self.acs)

    @property
    def obs_short(self):
        return self.obs[:-1]

    @property
    def sts_short(self):
        return self.sts[:-1]

    def __len__(self):
        return len(self.rws)


 def get_state(env):
    if hasattr(env, 'env'):
        return env.env.state  # openai gym env, which is a TimeLimit object
    else:
        return env.state


def generate_rollout(pi, logp, env,
                     max_n_samples=None, max_n_rollouts=None,
                     max_rollout_len=None,
                     with_animation=False):
    """ Collect rollouts until we have enough samples.

        All rollouts are COMPLETE in that they never end prematurely: they end
        either when done is true or max_rollout_len is reached.

        Args:
            pi: a function that maps ob to ac
            logp: either None or a function that maps (obs, acs) to log probabilities
            env: a gym environment
            max_rollout_len: the maximal length of a rollout (i.e. the problem's horizon)
            max_n_samples: the minimal number of samples to collect
            max_n_rollouts: the maximal number of rollouts
            with_animation: display animiation of the first rollout
    """

    assert (max_n_samples is not None) or (max_n_rollouts is not None)  # so we can stop

    max_n_samples = max_n_samples or float('Inf')
    max_n_rollouts = max_n_rollouts or float('Inf')
    max_rollout_len = max_rollout_len or float('Inf')
    max_rollout_len = min(env._max_episode_steps, max_rollout_len)
    max_n_samples =
    n_samples = 0
    rollouts = []
    while True:
        animate_this_rollout = len(rollouts) == 0 and with_animation
        ob = env.reset()  # observation
        st = get_state(env)  # env state
        dn = False  # special state (whether it is an absorbing state)
        obs, acs, rws, sts, dns = [], [], [], [], []
        steps = 0  # steps so far
        while True:
            if animate_this_rollout:
                env.render()
                time.sleep(0.05)
            obs.append(ob)
            sts.append(st)
            dns.append(dn)
            # apply action and get to the next state
            ac = pi(ob)
            acs.append(ac)
            ob, rw, done, _ = env.step(ac)
            rws.append(rw)
            st = get_state(env)
            steps += 1
            if steps >= max_rollout_len:  # breaks due to steps limit
                sts.append(st)
                obs.append(ob)
                dns.append(False)
                break
            elif done:
                # breaks due to absorbing state
                sts.append(st)
                obs.append(ob)
                dns.apend(True)
                break
            else:
                dn = False
        # end of one rollout
        rollout = Rollout(obs=obs, acs=acs, rws=rws, sts=sts, dns=dns, logp=logp)
        rollouts.append(rollout)
        n_samples += len(rollout)
        if (n_samples >= max_n_samples) or (len(rollouts) >= max_n_rollouts):
            break
    ro = Dataset(rollouts)
    return ro
