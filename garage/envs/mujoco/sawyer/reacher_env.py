import numpy as np

from garage.core.serializable import Serializable
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.envs.mujoco.sawyer.sawyer_env import Configuration
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper


class ReacherEnv(SawyerEnv):
    def __init__(self,
                 goal_position,
                 start_position=None,
                 randomize_start_position=False,
                 **kwargs):
        def generate_start_goal():
            nonlocal start_position
            if start_position is None or randomize_start_position:
                center = self.sim.data.get_geom_xpos('target2')
                start_position = np.concatenate([center[:2], [0.15]])

            start = Configuration(
                gripper_pos=start_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=goal_position)
            goal = Configuration(
                gripper_pos=goal_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=goal_position)

            return start, goal

        SawyerEnv.__init__(self,
                           start_goal_config=generate_start_goal,
                           **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        if self._control_method == 'task_space_control':
            obs = np.concatenate([gripper_pos])
        elif self._control_method == 'position_control':
            obs = np.concatenate([self.joint_positions, gripper_pos])
        else:
            raise NotImplementedError

        self._achieved_goal = gripper_pos
        self._desired_goal = self._goal_configuration.gripper_pos

        # achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        # self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        # desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
        # self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs,
            'achieved_goal': self._achieved_goal,
            'desired_goal': self._desired_goal,
            'has_object': False,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos,
            'object_pos': self._desired_goal,
        }

    def compute_reward(env, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if env._reward_type == 'sparse':
            return (d < env._distance_threshold).astype(np.float32)

        return -d


class SimpleReacherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(ReacherEnv(*args, **kwargs))
