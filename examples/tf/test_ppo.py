"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 526.092
    RiseTime: itr 488
"""
import gym

from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def run_task(*_):
    """
    Wrap PPO training task in the run_task function.

    :param _:
    :return:
    """
    env = TfEnv(normalize(gym.make("Pendulum-v0")))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 126, 64),
        output_nonlinearity=None,
        std_share_network=True)

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(256, 126, 64)),
    )

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1024,
        max_path_length=100,
        n_itr=1000,
        discount=0.99,
        gae_lambda=0.98,
        policy_ent_coeff=0.0,
        optimizer_args=dict(max_grad_norm=0.5),
        plot=False,
    )
    algo.train()
    # env = TfEnv(normalize(gym.make("Pendulum-v0")))
    # policy = GaussianMLPPolicy(
    #     env_spec=env.spec,
    #     hidden_sizes=(256, 128, 64),
    #     hidden_nonlinearity=tf.nn.tanh,
    #     output_nonlinearity=None,
    #     std_share_network=True,
    # )
    # baseline = GaussianMLPBaseline(
    #     env_spec=env.spec,
    #     regressor_args=dict(hidden_sizes=(256, 128, 64)),
    # )
    # algo = TRPO(
    #     env=env,
    #     policy=policy,
    #     baseline=baseline,
    #     batch_size=1024,
    #     max_path_length=100,
    #     n_itr=1000,
    #     discount=0.99,
    #     gae_lambda=0.98,
    #     policy_ent_coeff=0.0,
    #     plot=False,
    # )
    # algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
