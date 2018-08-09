import sys

import moveit_commander
import cloudpickle
import rospy

from garage.contrib.ros.robots import Sawyer

# Please configure joint lists based on your experiment.
JOINT_LISTS = [
    'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5',
    'right_j6'
]

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.140923828125,
    'right_j1': -1.2789248046875,
    'right_j2': -3.043166015625,
    'right_j3': -2.139623046875,
    'right_j4': -0.047607421875,
    'right_j5': -0.7052822265625,
    'right_j6': -1.4102060546875,
}


def parse_traj(trajectory):
    length = len(trajectory['right_j0'])

    cmds = []

    for i in range(length):
        joint_cmd = {}
        for joint in JOINT_LISTS:
            joint_cmd[joint] = trajectory[joint][i]
        cmds.append(joint_cmd)

    return cmds


def run(trajectory):
    # Initialize moveit_commander
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('traj_placyback', anonymous=True)

    # Initialize moveit to get safety check
    moveit_robot = moveit_commander.RobotCommander()
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_group_name = 'right_arm'
    moveit_group = moveit_commander.MoveGroupCommander(moveit_group_name)

    sawyer = Sawyer(
        initial_joint_pos=INITIAL_ROBOT_JOINT_POS,
        control_mode='position',
        moveit_group=moveit_group_name)

    trajectory = cloudpickle.load(open(trajectory, 'rb'))

    cmds = parse_traj(trajectory)

    sawyer.reset()

    for cmd in cmds:
        sawyer.set_joint_positions(cmd)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trajectory', action='store', required=True, type=str)

    args = parser.parse_args()

    run(args.trajectory)
