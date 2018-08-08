import numpy as np

# Please configure joint lists based on your experiment.
JOINT_LISTS = []


def run(robot, trajectory):
    trajectory = np.load(trajectory)

    # instantiate robot env


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', action='store', required=True, type=str)
    parser.add_argument(
        '--trajectory', action='store', required=True, type=str)

    args = parser.parse_args()

    run(args.robot, args.trajectory)
