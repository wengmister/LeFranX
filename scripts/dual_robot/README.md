Replay trajectory:

    python scripts/dual_robot/dual_robot_replay.py /path/to/dataset --episode 2
    --speed 0.5 --robot-ip [YOUR FRANKA ROBOT IP]

Teleoperate combo robot:

    python scripts/dual_robot/dual_vr_teleoperator.py

Record dataset with combo robot:

    python scripts/dual_robot/dual_vr_record.py

Train with DP/ACT:

    python scripts/dual_robot/train_act_policy.py
    python scripts/dual_robot/train_dp_policy.py

Deploy with DP/ACT:

    python scripts/dual_robot/dual_robot_deploy_act.py
    python scripts/dual_robot/dual_robot_deploy_dp.py

Please refer to the scripts for more details.
