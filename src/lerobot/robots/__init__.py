from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Import robot modules to trigger registration
from . import (  # noqa: F401
    bi_so100_follower,
    franka_fer,
    franka_fer_xhand,
    hope_jr,
    koch_follower,
    so100_follower,
    so101_follower,
    xhand,
)
