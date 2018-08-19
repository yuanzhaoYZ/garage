from gym.envs.robotics import rotations
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
                object_pos=(0.7, 0.04, 0.04))

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
                           free_object=False,
                           **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        if self._control_method == 'task_space_control':
            obs = np.concatenate([gripper_pos, self.object_pos])
        elif self._control_method == 'position_control':
            obs = np.concatenate([self.joint_positions[2:], gripper_pos, self.object_position])
        else:
            raise NotImplementedError

        self._achieved_goal = self.object_position
        self._desired_goal = self._goal_configuration.object_pos

        return {
            'observation': obs,
            'achieved_goal': self._achieved_goal,
            'desired_goal': self._desired_goal,
            'has_object': False,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos,
            'object_pos': self._achieved_goal,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        gripper_penalty = np.linalg.norm(self.object_position - self.gripper_position, axis=-1)
        block_penalty = np.linalg.norm(self.object_position - self._goal_configuration.object_pos, axis=-1)
        
        # Force the direction of the gripper
        upright_gripper = np.array([np.pi, 0, np.pi])
        gripper_rot = rotations.mat2euler(self.sim.data.get_site_xmat('grip'))
        gripper_norot = np.linalg.norm(np.sin(upright_gripper) - np.sin(gripper_rot))  

        # Block rotation 




        d = block_penalty
        if self._reward_type == 'sparse':
            return (d < self._distance_threshold).astype(np.float32)

        assert gripper_penalty > 0
        assert block_penalty > 0
        assert - (0.2 * gripper_penalty + block_penalty) < 0, "{}, {}".format(gripper_penalty, block_penalty)

        # return - (3 * block_penalty + gripper_penalty + 0.3 * gripper_norot)
        return - gripper_penalty - 1.2 * block_penalty


class SimplePusherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(PusherEnv(*args, **kwargs))