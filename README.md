# LeFranka

LeRobot Integration for Franka FER Robot, and optionally XHand dexterous hand.

## Architecture

There are three main parts to this LeRobot Franka robot extension:

1. Franka server - needs to be built and deployed on your real-time machine that controls the robot.
```bash
cd franka_server
bash build.sh
```
3. Franka teleoperator - needs to be built and added to your environment
```bash
cd franka_xhand_teleoperator
[uv] pip install -e .
```
4. LeRobot classes - copy to merge with files under LeRobot's `src` directory. This will include new Robot and Teleoperator class implementations needed to work with the rest of the framework.

## Note regarding gripper

I didn't develop an interface for the Franka Hand gripper since I didn't use it, but contributions are welcome!

## Usage

Call any LeRobot utility as you would with the new Robots! Examples can be found under `scripts`.

>[!CAUTION] 
>If you're comboing robots together like I did, it might be a good idea to call some utility methods directly as opposed to using the existing python implementations (for training and rollout, etc.). Combo robot setup will cause circular import calls with Draccus; I'm currently bypassing this issue by constructing train and deployment scripts directly. Please let me know if you would have a better method to deal with this.


