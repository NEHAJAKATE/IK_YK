"""
PyBullet simulation environment for 6-DOF robotic arm
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
from pathlib import Path
from kinematics import forward_kinematics, HOME_CONFIG


class RobotSimulation:
    def __init__(self, gui=True):
        """Initialize PyBullet simulation"""
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot
        urdf_path = Path(__file__).parent / "assets" / "my_robot.urdf"
        self.robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
        
        # Set to home configuration
        self.set_joint_angles(HOME_CONFIG)
        
        # Debug sphere for target visualization
        self.target_sphere = None
        
    def set_joint_angles(self, q: np.ndarray, interpolate=False, steps=100):
        """
        Set joint angles with optional smooth interpolation
        
        Args:
            q: Joint angles in radians (6-element array)
            interpolate: If True, smoothly move to target
            steps: Number of interpolation steps
        """
        if interpolate:
            current_q = self.get_joint_angles()
            for alpha in np.linspace(0, 1, steps):
                intermediate_q = current_q + alpha * (q - current_q)
                for idx, angle in zip(self.joint_indices[:6], intermediate_q):
                    p.resetJointState(self.robot_id, idx, angle)
                p.stepSimulation()
                time.sleep(0.01)
        else:
            for idx, angle in zip(self.joint_indices[:6], q):
                p.resetJointState(self.robot_id, idx, angle)
    
    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles"""
        angles = []
        for idx in self.joint_indices[:6]:
            state = p.getJointState(self.robot_id, idx)
            angles.append(state[0])
        return np.array(angles)
    
    def get_end_effector_pose(self) -> tuple:
        """
        Get end-effector position and orientation
        
        Returns:
            (position, rotation_matrix)
        """
        # Get link state for the last link (magnet)
        link_state = p.getLinkState(self.robot_id, self.joint_indices[-1] + 1)
        pos = np.array(link_state[0])
        orn_quat = link_state[1]
        
        # Convert quaternion to rotation matrix
        rot_mat = np.array(p.getMatrixFromQuaternion(orn_quat)).reshape(3, 3)
        
        return pos, rot_mat
    
    def move_to_pose(self, target_pos: np.ndarray, target_rot: np.ndarray, interpolate=True):
        """
        Move end-effector to target pose using PyBullet IK
        
        Args:
            target_pos: Target position [x, y, z] in meters
            target_rot: Target 3x3 rotation matrix
            interpolate: Smooth motion if True
        
        Returns:
            True if successful, False otherwise
        """
        # Convert rotation matrix to quaternion
        # Using Shepperd's method
        trace = np.trace(target_rot)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (target_rot[2, 1] - target_rot[1, 2]) * s
            y = (target_rot[0, 2] - target_rot[2, 0]) * s
            z = (target_rot[1, 0] - target_rot[0, 1]) * s
        else:
            if target_rot[0, 0] > target_rot[1, 1] and target_rot[0, 0] > target_rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + target_rot[0, 0] - target_rot[1, 1] - target_rot[2, 2])
                w = (target_rot[2, 1] - target_rot[1, 2]) / s
                x = 0.25 * s
                y = (target_rot[0, 1] + target_rot[1, 0]) / s
                z = (target_rot[0, 2] + target_rot[2, 0]) / s
            elif target_rot[1, 1] > target_rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + target_rot[1, 1] - target_rot[0, 0] - target_rot[2, 2])
                w = (target_rot[0, 2] - target_rot[2, 0]) / s
                x = (target_rot[0, 1] + target_rot[1, 0]) / s
                y = 0.25 * s
                z = (target_rot[1, 2] + target_rot[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + target_rot[2, 2] - target_rot[0, 0] - target_rot[1, 1])
                w = (target_rot[1, 0] - target_rot[0, 1]) / s
                x = (target_rot[0, 2] + target_rot[2, 0]) / s
                y = (target_rot[1, 2] + target_rot[2, 1]) / s
                z = 0.25 * s
        
        target_orn = [x, y, z, w]
        
        # Use PyBullet IK
        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.joint_indices[-1] + 1,  # End-effector link
            target_pos.tolist(),
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        if solution:
            solution_array = np.array(solution[:6])
            
            # Check joint limits
            from kinematics import JOINT_LIMITS
            if np.all((solution_array >= JOINT_LIMITS[:, 0]) & (solution_array <= JOINT_LIMITS[:, 1])):
                self.set_joint_angles(solution_array, interpolate=interpolate)
                return True
        
        return False
    
    def show_target_sphere(self, position: np.ndarray, radius=0.02, color=[1, 0, 0, 0.5]):
        """Visualize target position with a debug sphere"""
        if self.target_sphere is not None:
            p.removeBody(self.target_sphere)
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        self.target_sphere = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
    
    def validate_fk_ik(self, num_tests=10, verbose=True):
        """
        Validate FK-IK consistency using PyBullet IK
        
        Test: Random joint angles -> FK -> IK -> FK
        Check position and orientation errors
        """
        from kinematics import JOINT_LIMITS
        
        print(f"\n{'='*60}")
        print("FK-IK VALIDATION TEST (using PyBullet IK)")
        print(f"{'='*60}")
        
        position_errors = []
        orientation_errors = []
        
        for i in range(num_tests):
            # Generate random joint angles within limits
            q_random = np.random.uniform(
                JOINT_LIMITS[:, 0],
                JOINT_LIMITS[:, 1]
            )
            
            # FK: q -> pose
            T = forward_kinematics(q_random)
            pos_target = T[:3, 3]
            rot_target = T[:3, :3]
            
            # IK: pose -> q' (using PyBullet)
            success = self.move_to_pose(pos_target, rot_target, interpolate=False)
            
            if not success:
                print(f"Test {i+1}: IK failed (no solution)")
                continue
            
            q_solution = self.get_joint_angles()
            
            # FK again: q' -> pose'
            T_verify = forward_kinematics(q_solution)
            pos_verify = T_verify[:3, 3]
            rot_verify = T_verify[:3, :3]
            
            # Compute errors
            pos_error = np.linalg.norm(pos_target - pos_verify) * 1000  # mm
            
            # Orientation error using Frobenius norm
            rot_error_mat = rot_target.T @ rot_verify
            rot_error = np.arccos(np.clip((np.trace(rot_error_mat) - 1) / 2, -1, 1))
            rot_error_deg = np.degrees(rot_error)
            
            position_errors.append(pos_error)
            orientation_errors.append(rot_error_deg)
            
            if verbose:
                print(f"Test {i+1}: Pos error = {pos_error:.4f} mm, "
                      f"Rot error = {rot_error_deg:.4f} deg")
        
        if position_errors:
            print(f"\n{'='*60}")
            print(f"Position error:    mean = {np.mean(position_errors):.4f} mm, "
                  f"max = {np.max(position_errors):.4f} mm")
            print(f"Orientation error: mean = {np.mean(orientation_errors):.4f} deg, "
                  f"max = {np.max(orientation_errors):.4f} deg")
            print(f"{'='*60}\n")
            
            return np.max(position_errors) < 10.0 and np.max(orientation_errors) < 5.0
        
        return False
    
    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect()
