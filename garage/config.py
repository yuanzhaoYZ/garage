import os
import os.path as osp
from string import Template
import json
import sys


CONFIG_TEMPLATE = Template("""
import os.path as osp
import os

USE_GPU = bool(os.getenv("USE_GPU",False))

USE_TF = True

AWS_REGION_NAME = os.getenv("AWS_REGION_NAME","us-west-1")

S3_BUCKET_NAME = os.environ["GARAGE_S3_BUCKET"]


if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://%s/garage/experiments" % S3_BUCKET_NAME

AWS_CODE_SYNC_S3_PATH = "s3://%s/garage/code" % S3_BUCKET_NAME

ALL_REGION_AWS_SECURITY_GROUP_IDS = eval(os.getenv("ALL_REGION_AWS_SECURITY_GROUP_IDS","dict()"))

ALL_REGION_AWS_KEY_NAMES = eval(os.getenv("ALL_REGION_AWS_KEY_NAMES","dict()"))

ALL_REGION_AWS_IMAGE_IDS = {
    "ap-northeast-1": "ami-002f0167",
    "ap-northeast-2": "ami-590bd937",
    "ap-south-1": "ami-77314318",
    "ap-southeast-1": "ami-1610a975",
    "ap-southeast-2": "ami-9dd4ddfe",
    "eu-central-1": "ami-63af720c",
    "eu-west-1": "ami-41484f27",
    "sa-east-1": "ami-b7234edb",
    "us-east-1": "ami-83f26195",
    "us-east-2": "ami-66614603",
    "us-west-1": "ami-576f4b37",
    "us-west-2": "ami-b8b62bd8"
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "garage"

AWS_SECURITY_GROUPS = ["garage-sg"]

FAST_CODE_SYNC_IGNORES = [
    ".git",
    "data/local",
    "data/s3",
    "data/video",
    "src",
    ".idea",
    ".pods",
    "tests",
    "examples",
    "docs",
    ".idea",
    ".DS_Store",
    ".ipynb_checkpoints",
    "blackbox",
    "blackbox.zip",
    "*.pyc",
    "*.ipynb",
    "scratch-notebooks",
    "conopt_root",
    "private/key_pairs",
]

FAST_CODE_SYNC = True

""")


def write_config():
    print("Writing config file...")
    content = CONFIG_TEMPLATE.substitute()
    config_personal_file = os.path.join(PROJECT_PATH,
                                        "garage/config_personal.py")
    if os.path.exists(config_personal_file):
        if not query_yes_no("garage/config_personal.py exists. Override?"):
            sys.exit()
    with open(config_personal_file, "wb") as f:
        f.write(content.encode("utf-8"))

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = PROJECT_PATH + "/data"

USE_TF = False

DOCKER_IMAGE = "DOCKER_IMAGE"

DOCKERFILE_PATH = "/path/to/Dockerfile"

KUBE_PREFIX = "garage_"

DOCKER_LOG_DIR = "/tmp/expt"

POD_DIR = PROJECT_PATH + "/.pods"

AWS_S3_PATH = None

AWS_IMAGE_ID = None

AWS_INSTANCE_TYPE = "m4.xlarge"

AWS_KEY_NAME = "AWS_KEY_NAME"

AWS_SPOT = True

AWS_SPOT_PRICE = '1.0'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "garage"

AWS_SECURITY_GROUPS = ["garage-sg"]

AWS_NETWORK_INTERFACES = []

AWS_EXTRA_CONFIGS = dict()

AWS_REGION_NAME = "us-east-1"

CODE_SYNC_IGNORES = ["*.git/*", "*data/*", "*.pod/*"]

DOCKER_CODE_DIR = "/root/code/garage"

AWS_CODE_SYNC_S3_PATH = "s3://to/be/overriden/in/personal"

# whether to use fast code sync
FAST_CODE_SYNC = True

FAST_CODE_SYNC_IGNORES = [".git", "data", ".pods"]

KUBE_DEFAULT_RESOURCES = {
    "requests": {
        "cpu": 0.8,
    }
}

KUBE_DEFAULT_NODE_SELECTOR = {
    "aws/type": "m4.xlarge",
}

MUJOCO_KEY_PATH = osp.expanduser("~/.mujoco")
# MUJOCO_KEY_PATH = osp.join(osp.dirname(__file__), "../vendor/mujoco")

ENV = {}

EBS_OPTIMIZED = True

if osp.exists(osp.join(osp.dirname(__file__), "config_personal.py")):
    from garage.config_personal import *  # noqa: F401, F403
else:
    print("Creating your personal config from template...")
    from shutil import copy
    copy(
        osp.join(PROJECT_PATH, "garage/config_personal_template.py"),
        osp.join(PROJECT_PATH, "garage/config_personal.py"))
    write_config()
    from garage.config_personal import *  # noqa: F401, F403
    print("Personal config created, but you should probably edit it before "
          "further experiments are run")
    if 'CIRCLECI' not in os.environ:
        print("Exiting.")
        import sys
        sys.exit(0)

LABEL = ""
