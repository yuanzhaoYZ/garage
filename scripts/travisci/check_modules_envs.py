from check_modules_util import check_modules

modules = [
    "garage.envs.box2d", "garage.envs.box2d.models",
    "garage.envs.box2d.parser", "garage.envs.dm_control",
    "garage.envs.mujoco.gather", "garage.envs.mujoco.hill",
    "garage.envs.mujoco.maze", "garage.envs.mujoco.randomization"
]

non_required_modules = ["tensorflow", "theano", "lasagne"]

# This test will fail for "tensorflow", since garage.misc.logger and
# garage.misc.tensorboard_output are imported

check_modules(modules, non_required_modules)
