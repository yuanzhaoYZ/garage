from check_modules_util import check_modules

modules = [
    "garage.theano.algos", "garage.theano.baselines", "garage.theano.core",
    "garage.theano.distributions", "garage.theano.envs", "garage.theano.misc",
    "garage.theano.optimizers", "garage.theano.policies",
    "garage.theano.q_functions", "garage.theano.regressors",
    "garage.theano.sampler", "garage.theano.spaces"
]

non_required_modules = [
    "tensorflow", "pygame", "Box2D", "dm_control", "mujoco_py"
]

# This test will fail for "tensorflow", since garage.misc.logger and
# garage.misc.tensorboard_output are imported

check_modules(modules, non_required_modules)
