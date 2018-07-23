from garage.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.envs import normalize
from garage.envs.util import spec, tf_spec
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.misc.instrument import run_experiment
from garage.tf.policies import GaussianMLPPolicy

def run(*_):
    env = normalize(ReacherEnv())

    baseline = LinearFeatureBaseline(env_spec=spec(env))
    policy = GaussianMLPPolicy(env_spec=tf_spec(env), hidden_sizes=(32, 32))
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=400,
        n_itr=2000,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()


run_experiment(
    run,
    n_parallel=2,
    plot=False,
)

