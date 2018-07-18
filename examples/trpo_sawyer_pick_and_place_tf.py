import numpy as np

from garage.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.envs import normalize
from garage.envs.util import spec, tf_spec
from garage.envs.mujoco.sawyer import PickAndPlaceEnv
from garage.misc.instrument import run_experiment
from garage.tf.policies import GaussianMLPPolicy


def run(*_):
    env = normalize(PickAndPlaceEnv())

    baseline = LinearFeatureBaseline(env_spec=spec(env))
    policy = GaussianMLPPolicy(env_spec=tf_spec(env), hidden_sizes=(32, 32))
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=20000,
        max_path_length=500,
        n_itr=2000,
        discount=0.99,
        step_size=0.01,
        plot=True
    )
    algo.train()


run_experiment(
    run,
    n_parallel=2,
    plot=True,
    exp_prefix='trpo_pnp',
)


# env = PickAndPlaceEnv()
#
# action = np.array([0.75, 0.25, 0.1])
# gripper = np.array([10, -10])
# for _ in range(4000):
#     env.render()
#     env.sim.data.set_mocap_pos('mocap', action)
#     env.sim.data.set_mocap_quat('mocap', np.array([0., 1., 1., 0.]))
#     # ctrl_set_action(env.sim, gripper)
#     env.sim.step()
#
# for _ in range(495):
#     env.render()
#     action = np.array([0, 0, -0.01, 10])
#     _, reward, _, _ = env.step(action)
#     print(env._grasp(), reward)
#
# for _ in range(1500):
#     env.render()
#     action = np.array([0, 0, 0, -50])
#     _, reward, _, _ =env.step(action)
#     print(env._grasp(), reward)
#
# for _ in range(1500):
#     env.render()
#     action = np.array([0, 0, 0.01, -50])
#     _, reward, _, _ =env.step(action)
#     print(env._grasp(), reward)



