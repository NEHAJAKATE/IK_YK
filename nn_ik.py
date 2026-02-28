"""
Neural Network IK Solver - Real-time inverse kinematics

Outputs raw joint angles from NN then clamps to JOINT_LIMITS.
predict_with_seed() exposes both the raw (seed) and clamped values
for display in the telemetry panel.
"""
import numpy as np
import pickle
import os


class SimpleIKNetwork:
    """Minimal neural network for IK (no external dependencies)"""
    
    def __init__(self):
        self.weights = []
        self.biases = []
        self.trained = False
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        """Forward pass through network"""
        if not self.trained:
            return None
        
        a = x
        for i in range(len(self.weights) - 1):
            a = self.relu(a @ self.weights[i] + self.biases[i])
        
        # Output layer (no activation)
        a = a @ self.weights[-1] + self.biases[-1]
        return a
    
    def predict(self, position, orientation):
        """
        Predict joint angles from pose, clamped to joint limits.

        Args:
            position:    (3,) array [x, y, z] in meters
            orientation: (3, 3) rotation matrix

        Returns:
            (6,) array of joint angles in radians, within JOINT_LIMITS,
            or None if network is not trained.
        """
        from kinematics import JOINT_LIMITS

        # Normalize inputs if trained normalization stats exist
        x = np.concatenate([position, orientation.flatten()])
        if hasattr(self, 'x_mean') and hasattr(self, 'x_std'):
            x = (x - self.x_mean) / self.x_std

        q = self.forward(x)

        if q is None:
            return None

        # Enforce joint limits
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return q

    def predict_with_seed(self, position, orientation):
        """
        Predict joint angles and return both raw NN output (seed) and
        clamped result.

        Returns:
            (q_seed_raw, q_clamped) tuple of (6,) arrays, or (None, None)
        """
        from kinematics import JOINT_LIMITS

        if not self.trained:
            return None, None

        x = np.concatenate([position, orientation.flatten()])
        if hasattr(self, 'x_mean') and hasattr(self, 'x_std'):
            x = (x - self.x_mean) / self.x_std

        q_raw = self.forward(x)
        if q_raw is None:
            return None, None

        q_clamped = np.clip(q_raw, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return q_raw, q_clamped
    
    def train_from_data(self, data_file='ik_training_data.pkl', epochs=100):
        """Train network from dataset"""
        print("Training neural network IK solver...")

        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found. Run generate_training_data.py first.")
            return False

        # Load data
        with open(data_file, 'rb') as f:
            dataset = pickle.load(f)

        X = np.concatenate([dataset['positions'], dataset['orientations']], axis=1)
        y = dataset['joint_angles']
        
        # Normalize inputs
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X = (X - self.x_mean) / self.x_std
        
        # Initialize network (12 inputs -> 256 -> 128 -> 6 outputs)
        np.random.seed(42)
        self.weights = [
            np.random.randn(12, 256) * 0.01,
            np.random.randn(256, 128) * 0.01,
            np.random.randn(128, 6) * 0.01
        ]
        self.biases = [
            np.zeros(256),
            np.zeros(128),
            np.zeros(6)
        ]
        
        # Simple gradient descent
        lr = 0.001
        batch_size = 32
        n_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                a1 = self.relu(X_batch @ self.weights[0] + self.biases[0])
                a2 = self.relu(a1 @ self.weights[1] + self.biases[1])
                y_pred = a2 @ self.weights[2] + self.biases[2]
                
                # Loss
                loss = np.mean((y_pred - y_batch) ** 2)
                total_loss += loss
                
                # Backward pass (simplified)
                grad_out = 2 * (y_pred - y_batch) / len(X_batch)
                
                grad_w2 = a2.T @ grad_out
                grad_b2 = grad_out.sum(axis=0)
                
                grad_a2 = grad_out @ self.weights[2].T
                grad_a2[a2 <= 0] = 0
                
                grad_w1 = a1.T @ grad_a2
                grad_b1 = grad_a2.sum(axis=0)
                
                grad_a1 = grad_a2 @ self.weights[1].T
                grad_a1[a1 <= 0] = 0
                
                grad_w0 = X_batch.T @ grad_a1
                grad_b0 = grad_a1.sum(axis=0)
                
                # Update
                self.weights[2] -= lr * grad_w2
                self.biases[2] -= lr * grad_b2
                self.weights[1] -= lr * grad_w1
                self.biases[1] -= lr * grad_b1
                self.weights[0] -= lr * grad_w0
                self.biases[0] -= lr * grad_b0
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.6f}")
        
        self.trained = True
        print("Training complete!")
        return True
    
    def save(self, filename='ik_network.pkl'):
        """Save trained network"""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'trained': self.trained
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Network saved to {filename}")
    
    def load(self, filename='ik_network.pkl'):
        """Load trained network"""
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.weights = data['weights']
        self.biases = data['biases']
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.trained = data['trained']
        
        print(f"Network loaded from {filename}")
        return True


if __name__ == "__main__":
    # Train network
    network = SimpleIKNetwork()
    
    if network.train_from_data('ik_training_data.pkl', epochs=50):
        network.save('ik_network.pkl')
        print("\nNetwork ready for use!")
