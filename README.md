# LeFranka

LeRobot Integration for Franka FER Robot. Check out `XHand` branch for Franka FER + XHand combo robot deployment.

## Architecture

There are three main parts to this LeRobot Franka robot extension, `franka_server`, `franka_xhand_teleoperator`, and added class implementation under `src/lerobot`. Check out system flowchart below:

## Use

Project was tested on [`LeRobot`](https://github.com/huggingface/lerobot) commit [`ce3b9f627e55223d6d1c449d348c6b351b35d082`](https://github.com/huggingface/lerobot/commit/ce3b9f627e55223d6d1c449d348c6b351b35d082). To use this extension, copy and paste all content inside the repo over to your `LeRobot` directory and do the following:

1. Franka server - needs to be built and deployed on your real-time machine that controls the robot.
```bash
cd franka_server
bash build.sh
```

Find the built `franka_server` and copy over to your robot RTPC (or run in a second terminal if it's the same PC) 

>[!NOTE]
> You will need to run the following commands on RTPC to start up the server before using any LeRobot utilities to move the arm.

```bash
./franka_server [YOUR_FRANKA_ROBOT_IP]
```

2. Franka teleoperator - needs to be built and added to your environment
   
```bash
cd franka_xhand_teleoperator
[uv] pip install -e .
```
3. LeRobot classes - copy to merge with files under LeRobot's `src` directory. This will include new Robot and Teleoperator class implementations needed to work with the rest of the framework.

>[!NOTE]
> Due to time constraints and the nature of the project, I didn't develop an interface for the Franka Hand gripper since I didn't use it, but contributions are welcome!

## Usage

Call any LeRobot utility as you would with the new Robots! Examples can be found under `scripts`.

>[!CAUTION] 
>If you're comboing robots together like I did with Franka + XHand, it might be a good idea to call utility methods directly in your own python script as opposed to using the existing python implementations with arguments (for training and rollout, etc.). Combo robot setup will cause circular import calls with `Draccus`; I'm currently bypassing this issue by constructing train and deployment scripts directly. Please let me know if you would have a better method to deal with this.

## Demo tasks
### Pick up orange cube and place in blue bin:

### Pick up toast, place in toaster, and press toast lever:

### Open box lid, pick up pie and place in brown bin:


## Datasets
Open-source datasets for the demo tasks could be found on HuggingFace [here](https://huggingface.co/wengmister).

## License
Apache-2.0
