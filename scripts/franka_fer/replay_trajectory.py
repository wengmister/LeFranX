#!/usr/bin/env python3
"""
Extract and replay joint trajectories from LeRobot parquet dataset.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_trajectory_from_dataset(dataset_path: str, episode_index: int = 0):
    """Load joint trajectory from LeRobot dataset"""
    try:
        # Load dataset
        dataset = LeRobotDataset(dataset_path)
        print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} total frames")
        
        # Get specific episode
        episode_data = []
        for frame_idx in range(dataset.num_frames):
            frame = dataset[frame_idx]
            if frame['episode_index'] == episode_index:
                episode_data.append(frame)
        
        if not episode_data:
            print(f"No data found for episode {episode_index}")
            return None, None
            
        print(f"Episode {episode_index}: {len(episode_data)} frames")
        
        # Extract joint positions and timestamps
        joint_trajectory = []
        timestamps = []
        
        for frame in episode_data:
            # Extract joint positions from action
            joint_positions = frame['action'].numpy()  # Should be 7 joint positions
            joint_trajectory.append(joint_positions)
            timestamps.append(frame['timestamp'].item())
            
        joint_trajectory = np.array(joint_trajectory)
        timestamps = np.array(timestamps)
        
        # Convert to relative time (start from 0)
        timestamps = timestamps - timestamps[0]
        
        print(f"Trajectory shape: {joint_trajectory.shape}")
        print(f"Duration: {timestamps[-1]:.2f} seconds")
        print(f"Average frequency: {len(timestamps) / timestamps[-1]:.1f} Hz")
        
        return joint_trajectory, timestamps
        
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None, None

def load_trajectory_from_parquet(parquet_file: str):
    """Load joint trajectory directly from parquet file"""
    try:
        # Load parquet file
        df = pd.read_parquet(parquet_file)
        print(f"Parquet file loaded: {len(df)} frames")
        print(f"Columns: {list(df.columns)}")
        
        # Check if action column exists
        if 'action' not in df.columns:
            print("No 'action' column found in parquet file")
            return None, None
        
        # Extract joint trajectory from action column
        # Actions are stored as arrays, so we need to stack them
        action_arrays = df['action'].values
        
        # Convert list of arrays to 2D numpy array
        joint_trajectory = np.stack(action_arrays)
        
        # Extract timestamps
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) / 30.0
        
        # Convert to relative time
        timestamps = timestamps - timestamps[0]
        
        print(f"Action data shape: {joint_trajectory.shape}")
        print(f"Duration: {timestamps[-1]:.2f} seconds")
        print(f"Sample action: {joint_trajectory[0]}")
        
        return joint_trajectory, timestamps
        
    except Exception as e:
        print(f"Error loading parquet: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def replay_trajectory(joint_trajectory, timestamps, robot_config=None, dry_run=True):
    """Replay joint trajectory on robot"""
    
    if robot_config is None:
        robot_config = FrankaFERConfig(
            server_ip="192.168.18.1",
            server_port=5000,
            cameras={}
        )
    
    if not dry_run:
        # Connect to robot
        robot = FrankaFER(robot_config)
        robot.connect(calibrate=False)
        print("Robot connected")
        
        # Move to starting position
        print("Moving to trajectory start position...")
        start_action = {f"joint_{i}.pos": float(joint_trajectory[0, i]) for i in range(7)}
        robot.send_action(start_action)
        time.sleep(2.0)  # Wait for movement to complete
    
    print("\nReplaying trajectory...")
    print("Press Ctrl+C to stop")
    
    try:
        start_time = time.time()
        
        for i, (joint_positions, target_time) in enumerate(zip(joint_trajectory, timestamps)):
            # Wait for correct timing
            current_time = time.time() - start_time
            sleep_time = float(target_time) - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Create action dictionary
            action = {f"joint_{i}.pos": float(joint_positions[i]) for i in range(7)}
            
            if dry_run:
                print(f"Frame {i:4d} @ {target_time:6.2f}s: {[f'{j:.3f}' for j in joint_positions]}")
            else:
                # Send to robot
                robot.send_action(action)
                print(f"Frame {i:4d} @ {target_time:6.2f}s: Sent to robot")
                    
    except KeyboardInterrupt:
        print("\nTrajectory replay interrupted")
    
    if not dry_run and 'robot' in locals():
        robot.disconnect()
        print("Robot disconnected")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python replay_trajectory.py <dataset_path> [episode_index] [--robot]")
        print("  python replay_trajectory.py <parquet_file> [--robot]")
        print("")
        print("Examples:")
        print("  python replay_trajectory.py datasets/meta")
        print("  python replay_trajectory.py datasets/meta 0 --robot")
        print("  python replay_trajectory.py datasets/data/chunk-000/episode_000000.parquet")
        return
    
    path = sys.argv[1]
    use_robot = "--robot" in sys.argv
    
    # Determine if it's a dataset or parquet file
    if path.endswith('.parquet'):
        # Direct parquet file
        joint_trajectory, timestamps = load_trajectory_from_parquet(path)
    else:
        # Dataset directory
        episode_index = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "--robot" else 0
        joint_trajectory, timestamps = load_trajectory_from_dataset(path, episode_index)
    
    if joint_trajectory is None:
        print("Failed to load trajectory")
        return
    
    # Show trajectory info
    print(f"\nTrajectory Summary:")
    print(f"  Frames: {len(joint_trajectory)}")
    print(f"  Duration: {timestamps[-1]:.2f} seconds")
    print(f"  Joints: {joint_trajectory.shape[1]}")
    print(f"  Start position: {[f'{j:.3f}' for j in joint_trajectory[0]]}")
    print(f"  End position: {[f'{j:.3f}' for j in joint_trajectory[-1]]}")
    
    if use_robot:
        print("\n⚠️  WARNING: This will move the real robot!")
        print("Make sure the robot is in a safe state and workspace is clear.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted")
            return
        
        replay_trajectory(joint_trajectory, timestamps, dry_run=False)
    else:
        print("\nDry run mode (no robot movement)")
        replay_trajectory(joint_trajectory, timestamps, dry_run=True)

if __name__ == "__main__":
    main()