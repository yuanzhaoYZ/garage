import numpy as np

from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.envs.mujoco.sawyer.sawyer_env import Configuration
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
from garage.core.serializable import Serializable


class PusherEnv(SawyerEnv):

    def __init__(self,
                 goal_position=None,
                 start_position=None,
                 **kwargs):
        def generate_start_goal():
            nonlocal start_position
            if start_position is None:
                center = (0, 0, 0)
                start_position = np.concatenate([center[:2], [0.15]])

            start = Configuration(
                gripper_pos=start_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=(0.65, 0.04, 0.04))

            nonlocal goal_position
            if goal_position is None:
                goal_position = (0.9, 0.04, 0.04)

            goal = Configuration(
                gripper_pos=goal_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=goal_position)

            return start, goal

        SawyerEnv.__init__(self,
                           start_goal_config=generate_start_goal,
                           file_path="pusher.xml",
                           **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        if self._control_method == 'task_space_control':
            obs = np.concatenate([gripper_pos, self.object_pos])
        elif self._control_method == 'position_control':
            obs = np.concatenate([self.joint_positions[2:], gripper_pos, self.object_position])
        else:
            raise NotImplementedError

        achieved_goal = self.object_position
        desired_goal = self._goal_configuration.object_pos

        # achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        # self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        # desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
        # self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'has_object': False,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos,
            'object_pos': achieved_goal,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        gripper_reward = np.linalg.norm(self.object_position - self.gripper_position, axis=-1)
        block_reward = np.linalg.norm(self.object_position - self._goal_configuration.object_pos, axis=-1)
        reward = 0.3 * gripper_reward + 0.7 * block_reward # + ori_penalty
        if self._reward_type == 'sparse':
            return (reward < self._distance_threshold).astype(np.float32)

        return - reward


class SimplePusherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(PusherEnv(*args, **kwargs))