"""
Training Data Generator for Neural Network IK
Generates ground truth IK solutions using PyBullet
"""
import numpy as np
import pickle
from simulation import RobotSimulation
from kinematics import forward_kinematics, JOINT_LIMITS, HOME_CONFIG


def generate_training_data(num_samples=100000, output_file='ik_training_data.pkl'):
    """
    Generate training dataset for neural network IK
    
    Strategy:
    1. Sample random valid joint configurations
    2. Compute FK to get end-effector pose
    3. Use PyBullet IK to get ground truth solution
    4. Verify and store successful samples
    
    Args:
        num_samples: Number of samples to generate
        output_file: Output pickle file path
    
    Returns:
        Dictionary with training data
    """
    print(f"Generating {num_samples} training samples...")
    print("="*60)
    
    sim = RobotSimulation(gui=False)
    
    dataset = {
        'positions': [],      # (N, 3) - target XYZ
        'orientations': [],   # (N, 9) - target rotation matrix flattened
        'joint_angles': []    # (N, 6) - ground truth joint angles
    }
    
    successful = 0
    attempts = 0
    
    while successful < num_samples and attempts < num_samples * 3:
        attempts += 1
        
        # Sample random joint configuration
        q_sample = np.random.uniform(
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )
        
        # Forward kinematics
        T = forward_kinematics(q_sample)
        target_pos = T[:3, 3]
        target_rot = T[:3, :3]
        
        # PyBullet IK (ground truth)
        success = sim.move_to_pose(target_pos, target_rot, interpolate=False)
        
        if success:
            q_ik = sim.get_joint_angles()
            
            # Verify solution quality
            T_verify = forward_kinematics(q_ik)
            pos_error = np.linalg.norm(T_verify[:3, 3] - target_pos)
            
            if pos_error < 0.001:
                dataset['positions'].append(target_pos)
                dataset['orientations'].append(target_rot.flatten())
                dataset['joint_angles'].append(q_ik)
                successful += 1
                
                if successful % 1000 == 0:
                    print(f"Progress: {successful}/{num_samples} samples "
                          f"(attempts: {attempts}, success rate: {100*successful/attempts:.1f}%)")
    
    # Convert to numpy arrays
    dataset['positions'] = np.array(dataset['positions'])
    dataset['orientations'] = np.array(dataset['orientations'])
    dataset['joint_angles'] = np.array(dataset['joint_angles'])
    
    # Save to file
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n{'='*60}")
    print(f"Dataset generated successfully!")
    print(f"Total samples: {successful}")
    print(f"Success rate: {100*successful/attempts:.1f}%")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Print statistics
    print("Dataset Statistics:")
    print(f"Position range:")
    print(f"  X: [{dataset['positions'][:, 0].min():.3f}, {dataset['positions'][:, 0].max():.3f}]")
    print(f"  Y: [{dataset['positions'][:, 1].min():.3f}, {dataset['positions'][:, 1].max():.3f}]")
    print(f"  Z: [{dataset['positions'][:, 2].min():.3f}, {dataset['positions'][:, 2].max():.3f}]")
    print(f"\nJoint angles range (degrees):")
    for i in range(6):
        angles_deg = np.degrees(dataset['joint_angles'][:, i])
        print(f"  Joint {i+1}: [{angles_deg.min():.1f}, {angles_deg.max():.1f}]")
    
    sim.close()
    return dataset


def load_training_data(input_file='ik_training_data.pkl'):
    """Load training data from file"""
    with open(input_file, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset['positions'])} samples from {input_file}")
    return dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into train/val/test sets
    
    Args:
        dataset: Dictionary with positions, orientations, joint_angles
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        train_data, val_data, test_data dictionaries
    """
    n_samples = len(dataset['positions'])
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    def extract_subset(idx):
        return {
            'positions': dataset['positions'][idx],
            'orientations': dataset['orientations'][idx],
            'joint_angles': dataset['joint_angles'][idx]
        }
    
    return extract_subset(train_idx), extract_subset(val_idx), extract_subset(test_idx)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IK training data')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='ik_training_data.pkl',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # Generate data
    dataset = generate_training_data(args.samples, args.output)
    
    # Split into train/val/test
    train_data, val_data, test_data = split_dataset(dataset)
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_data['positions'])} samples")
    print(f"  Validation: {len(val_data['positions'])} samples")
    print(f"  Test:       {len(test_data['positions'])} samples")
    
    # Save splits
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print("\nSaved train/val/test splits to separate files.")
