import unittest

import gym
import numpy as np

from garage.envs.wrappers import GrayScale
from garage.misc.overrides import overrides


class TestGrayScale(unittest.TestCase):
    @overrides
    def setUp(self):
        self.env = gym.make("Breakout-v0")
        self.env_r = GrayScale(gym.make("Breakout-v0"))

        self.obs = self.env.reset()
        self.obs_r = self.env_r.reset()

    def test_gray_scale_output(self):
        """
        RGB to grayscale conversion using scikit-image.

        Weights used for conversion:
        Y = 0.2125 R + 0.7154 G + 0.0721 B

        Reference:
        http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2grey
        """

        gray_scale_output = np.dot(self.obs[:, :, :3],
                                   [0.2125, 0.7154, 0.0721]) / 255.0
        np.testing.assert_array_almost_equal(gray_scale_output, self.obs_r)
