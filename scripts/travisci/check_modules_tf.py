from check_modules_util import check_modules

modules = [
    "garage.tf.algos", "garage.tf.baselines", "garage.tf.core",
    "garage.tf.distributions", "garage.tf.envs",
    "garage.tf.exploration_strategies", "garage.tf.misc",
    "garage.tf.optimizers", "garage.tf.plotter", "garage.tf.policies",
    "garage.tf.q_functions", "garage.tf.regressors", "garage.tf.replay_buffer",
    "garage.tf.samplers", "garage.tf.spaces"
]

non_required_modules = [
    "theano", "pygame", "Box2D", "dm_control", "mujoco_py", "lasagne"
]

check_modules(modules, non_required_modules)
