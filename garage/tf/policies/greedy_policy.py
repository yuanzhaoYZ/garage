import numpy as np

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.policies import Policy
from garage.tf.spaces import Box


class GreedyPolicy(Policy, Serializable):
    def __init__(self,
                 env_spec,
                 max_epsilon=1.0,
                 min_epsilon=0.02,
                 decay_period=10000):
        Serializable.quick_init(self, locals())
        super(GreedyPolicy, self).__init__(env_spec)

        self._env_spec = env_spec
        self._max_epsilon = max_epsilon
        self._epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay_period = decay_period
        self._action_space = env_spec.action_space

    def _init_qf(self, qf):
        self._qf = qf

    @property
    def vectorized(self):
        return True

    @overrides
    def get_action(self, observation):
        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period

        rand = np.random.random()
        opt_action = np.argmax(self._qf(observation))
        if rand < self._epsilon:
            opt_action = np.random.randint(0, self._action_space.n)

        return opt_action, dict()

    @overrides
    def get_actions(self, observations):

        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period

        opt_action = np.argmax(self._qf(observations), axis=1)
        for itr in range(len(opt_action)):
            if np.random.random() < self._epsilon:
                if isinstance(self._action_space, Box):
                    opt_action[itr] = np.random.randint(
                        0, self._action_space.flat_dim)
                else:
                    opt_action[itr] = np.random.randint(
                        0, self._action_space.n)

        return opt_action, dict()
