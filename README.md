# LeFranka

LeRobot Integration for Franka FER Robot. Check out `XHand` branch for Franka FER + XHand combo robot deployment.

[![Watch the video](https://img.youtube.com/vi/TzlUEWCjQ1M/0.jpg)](https://www.youtube.com/watch?v=TzlUEWCjQ1M)

## Architecture

There are three main parts to this LeRobot Franka robot extension, `franka_server`, `franka_xhand_teleoperator`, and added class implementation under `src/lerobot`. Check out system flowchart below:


<img width="1891" height="1649" alt="flow-chart" src="https://github.com/user-attachments/assets/cfd8389a-2ecf-4e1c-8f6f-ca1aa0905fbf" />

## Build

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

<details>
<summary>XHand Dependencies</summary>

For `XHand`, we will use a repository adapted based on Yuzhe Qin's amazing work on [`dex-retargeting`](https://github.com/dexsuite/dex-retargeting) to map human hand motion to the robot hand.

To enable XHand Motion Retargeting:

```bash
# First, update all git submodule
git submodule update --init --recursive 

# Build dependencies
cd vr-dex-retargeting
[uv] pip install -e .
```

   
</details>

3. LeRobot classes - copy to merge with files under LeRobot's `src` directory. This includes new Robot and Teleoperator class implementations needed to work with the rest of the framework.

>[!NOTE]
> Due to time constraints and the nature of the project, I didn't develop an interface for the Franka Hand gripper since I didn't use it, but contributions are welcome!

## Usage

Call any LeRobot utility as you would with the new Robots! Examples can be found under `scripts`.

>[!CAUTION] 
>If you're comboing robots together like I did with Franka + XHand, it might be a good idea to call utility methods directly in your own python script as opposed to using the existing python implementations with arguments (for training and rollout, etc.). Combo robot setup will cause circular import calls with `Draccus`; I'm currently bypassing this issue by constructing train and deployment scripts directly. Please let me know if you would have a better method to deal with this.

## Demo tasks
### Pick up orange cube and place in blue bin:

https://github.com/user-attachments/assets/5e6e1930-6bca-4d1a-b175-423de4388dc1

### Pick up toast, place in toaster, and press toast lever:

https://github.com/user-attachments/assets/03dbfd55-91e3-40f0-9e5c-fca9b33fad30

### Open box lid, pick up pie and place in brown bin:

https://github.com/user-attachments/assets/e5b54e07-031d-42e2-994b-030e646e2768

## Datasets
Open-source datasets for the demo tasks could be found on HuggingFace [here](https://huggingface.co/wengmister).

## License
Apache-2.0
