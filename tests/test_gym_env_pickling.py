import pickle
import unittest

import gym
from nose2.tools import params

from garage.envs import GarageEnv


class TestGymEnvs(unittest.TestCase):
    @params(list(gym.envs.registry.all()))
    def test_pickling_wrapper(self, spec):
        env = spec.make()
        env = GarageEnv(env)

        # Roundtrip serialization
        env = pickle.loads(pickle.dumps(env))
        assert env

        # Step and render still work after pickling
        assert env.reset()
        env.step(env.action_space.sample())
        env.render()

    @params(list(gym.envs.registry.all()))
    def test_step_and_render(self, spec):
        env = spec.make()
        env = GarageEnv(env)

        # Step and render work
        assert env.reset()
        done = False
        for _ in range(100):
            _, _, done, _ = env.step(env.action_space.sample())
            env.render()
            if done:
                break
