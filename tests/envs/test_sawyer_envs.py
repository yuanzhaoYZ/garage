import numpy as np

from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs.mujoco.sawyer import BinSortingEnv
from garage.envs.mujoco.sawyer import BlockStackingEnv
from garage.envs.mujoco.sawyer import PickAndPlaceEnv
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.envs.util import spec
from garage.misc.instrument import run_experiment
from garage.policies import GaussianMLPPolicy


def run_bin_sorting(*_):
    env = BinSortingEnv()

    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=2000,
        baseline=baseline,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


def run_block_stacking(*_):
    env = BlockStackingEnv()

    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=2000,
        baseline=baseline,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


def run_pick_and_place(*_):
    initial_goal = np.array([0.6, -0.1, 0.80])
    env = PickAndPlaceEnv(initial_goal)
    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=2000,
        baseline=baseline,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


def test_env():
    initial_goal = np.array([0.6, -0.1, 0.80])
    initial_qpos = {
        'right_j0': -0.140923828125,
        'right_j1': -1.2789248046875,
        'right_j2': -3.043166015625,
        'right_j3': -2.139623046875,
        'right_j4': -0.047607421875,
        'right_j5': -0.7052822265625,
        'right_j6': -1.4102060546875,
    }
    env = ReacherEnv(initial_goal, initial_qpos, control_method='position_control')
    for i in range(9999):
        env.render()
        action = initial_goal
        next_obs, reward, done, _ = env.step(action)
        if done:
            print(reward)
            break
    env.reset()
    env.close()


def test_pnp():
    initial_goal = np.array([0.6, -0.1, 0.80])
    initial_qpos = {
        'right_j0': -0.140923828125,
        'right_j1': -1.2789248046875,
        'right_j2': -3.043166015625,
        'right_j3': -2.139623046875,
        'right_j4': -0.047607421875,
        'right_j5': -0.7052822265625,
        'right_j6': -1.4102060546875,
    }
    env = PickAndPlaceEnv(initial_goal=initial_goal, initial_qpos=initial_qpos)

    for i in range(9999):
        env.render()
        action = env.action_space.sample()
        env.step(action)


test_pnp()


#
# test_env()
# run_experiment(
#     run_bin_sorting,
#     n_parallel=2,
#     plot=True,
# )
