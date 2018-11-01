import gym

from garage.envs import normalize
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import GreedyPolicy
from garage.tf.q_functions import CNNQFunction
from garage.wrapper.repeat_action_wrapper import RepeatActionWrapper
from garage.wrapper.resize_and_grayscale_wrapper import ResizeAndGrayscaleWrapper
from garage.wrapper.stack_frames_wrapper import StackFramesWrapper

env = TfEnv(
    normalize(
        StackFramesWrapper(
            RepeatActionWrapper(
                ResizeAndGrayscaleWrapper(
                    gym.make("Breakout-v0"), w=84, h=84, plot=True),
                frame_to_repeat=4),
            n_frames_stacked=4)))

replay_buffer = SimpleReplayBuffer(
    env_spec=env.spec, size_in_transitions=int(1e4), time_horizon=100)

policy = GreedyPolicy(env_spec=env.spec)

qf = CNNQFunction(
    env_spec=env.spec, hidden_sizes=(8, 4, 3), num_filters=(16, 32, 32))

algo = DQN(
    env=env,
    policy=policy,
    qf=qf,
    replay_buffer=replay_buffer,
    min_buffer_size=1e3,
    n_train_steps=1,
    dueling=True)

algo.train()
