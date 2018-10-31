from check_modules_util import check_modules

modules = [
    "garage.algos", "garage.baselines", "garage.core", "garage.distributions",
    "garage.envs", "garage.exploration_strategies", "garage.misc",
    "garage.optimizers", "garage.plotter", "garage.policies",
    "garage.q_functions", "garage.regressors", "garage.replay_buffer",
    "garage.sampler", "garage.spaces"
]

non_required_modules = [
    "tensorflow", "theano", "pygame", "Box2D", "dm_control", "mujoco_py"
]

# This test will fail for "tensorflow", since garage.misc.logger and
# garage.misc.tensorboard_output are imported

check_modules(modules, non_required_modules)
