import pickle

import numpy as np

from sandbox.embed2learn.misc.metrics import rrse


def get_coordinate_transformation(gazebo_env, mujoco_env):

    # Force both envs to be position control
    gazebo_env._robot._control_mode = "position"
    mujoco_env._control_method = "position_control"
    
    mujoco_env._randomize_start_jpos = True    
    gpos_g = []
    gpos_m = []
    for _ in range(100):
        obs_m = mujoco_env.reset()["observation"]
        jpos = obs_m[:7]
        joint_positions = {
            "right_j{}".format(i): jpos[i]
            for i in range(7)
        }
        gazebo_env._robot._limb.move_to_joint_positions(joint_positions)
        gpos_m.append(obs_m[7:].copy())
        gpos_g.append(gazebo_env._robot.gripper_position.copy())
    gpos_g = np.array(gpos_g)
    gpos_m = np.array(gpos_m)
    data = dict(
        points_gazebo=gpos_g,
        points_mujoco=gpos_m
    )
    pickle.dump(data, open("data.pkl", "wb"))
    print("Current rrse: {}".format(rrse(gpos_g, gpos_m)))
    M = np.hstack([gpos_g, np.ones((gpos_g.shape[0], 1))])
    G = np.hstack([gpos_m, np.ones((gpos_m.shape[0], 1))])
    A, res, rank, s = np.linalg.lstsq(M, G)

    def transform(gpos):
        if len(gpos.shape) == 1:
            gpos = np.array([gpos])
        pad_g = np.hstack([gpos_g, np.ones((gpos.shape[0], 1))])
        return np.dot(pad_g, A)[:, :-1]
    print(rrse(transform(gpos_g), gpos_m))
    gpos_g = []
    gpos_m = []
    for _ in range(10):
        obs_m = mujoco_env.reset()["observation"]
        jpos = obs_m[:7]
        joint_positions = {
            "right_j{}".format(i): jpos[i]
            for i in range(7)
        }
        gazebo_env._robot._limb.move_to_joint_positions(joint_positions)
        gpos_m.append(obs_m[7:].copy())
        gpos_g.append(gazebo_env._robot.gripper_position.copy())
    gpos_g = np.array(gpos_g)
    gpos_m = np.array(gpos_m)
    # T = np.linalg.lstsq(gpos_g, gpos_m, rcond=None)
    print("------Eval--------")
    print(rrse(gpos_g, gpos_m))
    print(rrse(transform(gpos_g), gpos_m))
    
    return transform
