import joblib

import numpy as np


def get_coodinate_transformation(gazebo_env, mujoco_env):

    # Make copies of the environments first
    gazebo_env = joblib.loads(joblib.dumps(gazebo_env))
    mujoco_env = joblib.loads(joblib.dumps(mujoco_env))

    # Force both envs to be position control
    gazebo_env._robot._control_mode = "position"
    mujoco_env._control_method = "position_control"

    # Get two gripper position to calculate to transition
    gazebo_env.reset()    
    mujoco_env.reset()

    gpos_g1 = gazebo_env._robot_.gripper_position
    gpos_m1 = mujoco_env.gripper_position

    # Set the mujoco env to be random joint position start to get another jpos
    mujoco_env.randomize_start_jpos = True
    obs = mujoco_env.reset()
    
    jpos = obs[:7]
    joint_positions = {
        "right_j{}".format(i): jpos[i]
        for i in range(7)
    }
    gazebo_env._robot._limb.set_joint_positions(joint_positions)

    gpos_g2 = gazebo_env._robot.gripper_position 
    gpos_m2 = obs[7:]

    dist_g = np.linalg.norm(gpos_g1 - gpos_g2, axis=-1)
    dist_m = np.linalg.norm(gpos_m1 - gpos_m2, axis=-1)
    scale = dist_m / dist_g

    offset = gpos_m1 - gpos_g1 * scale

    def trasnform(gpos):
        return gpos * scale + offset

    return trasnform
