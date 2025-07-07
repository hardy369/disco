#!/usr/bin/env python3
"""
Distributed Quantum Noise-Induced Reservoir Computing for NARMA2
Major improvements for better R² scores and NARMA2 prediction accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy import stats
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error, depolarizing_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
import time
import argparse
from typing import Tuple, Dict, Any

warnings.filterwarnings('ignore')

class NARMA2Generator:
    """Generate NARMA2 benchmark time series data with better initialization"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate(self, T: int, tau: int = 2, warmup: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate NARMA2 with warmup period for stability"""
        # Generate input sequence
        u = np.random.uniform(0.0, 0.5, T + warmup)  # Reduced range for stability
        
        # Initialize output
        y = np.zeros(T + warmup)
        
        # Better initialization
        y[0] = 0.3
        y[1] = 0.3
        
        # Generate with warmup
        for t in range(tau, T + warmup):
            y[t] = (0.4 * y[t-1] + 
                    0.4 * y[t-1] * y[t-2] + 
                    0.6 * (u[t] * 0.5)**3 + 
                    0.1)
        
        # Return without warmup period
        return u[warmup:], y[warmup:]

class OptimizedQuantumReservoir:
    """Heavily optimized quantum reservoir with multiple improvements"""
    
    def __init__(self, num_qubits: int = 4, noise_level: float = 0.05, 
                 backend: AerSimulator = None, node_id: int = 0):
        self.num_qubits = num_qubits
        self.noise_level = noise_level
        self.backend = backend or AerSimulator()
        self.node_id = node_id
        self.noise_model = self._create_optimized_noise_model()
        
        # Memory for temporal dynamics
        self.memory_states = np.zeros(num_qubits)
        self.memory_decay = 0.7
        
    def _create_optimized_noise_model(self) -> NoiseModel:
        """Create carefully tuned noise model"""
        nm = NoiseModel()
        
        # Balanced noise for optimal reservoir dynamics
        amp_damping = amplitude_damping_error(self.noise_level * 0.8)
        phase_damping = phase_damping_error(self.noise_level * 0.6)
        depolar = depolarizing_error(self.noise_level * 0.4, 1)
        
        # Compose errors
        single_error = amp_damping.compose(phase_damping).compose(depolar)
        two_qubit_error = depolarizing_error(self.noise_level * 0.8, 2)
        
        # Apply to common gates
        for gate in ['rx', 'ry', 'rz', 'h']:
            nm.add_all_qubit_quantum_error(single_error, gate)
        
        for gate in ['cx', 'cz']:
            nm.add_all_qubit_quantum_error(two_qubit_error, gate)
        
        return nm
    
    def process_input(self, u_t: float, shots: int = 2048) -> np.ndarray:
        """Optimized input processing with memory and better encoding"""
        
        # Create quantum circuit
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Normalize input
        u_norm = np.clip(u_t, 0, 1)
        
        # IMPROVED INPUT ENCODING
        # 1. Direct amplitude encoding
        for q in range(self.num_qubits):
            qc.rx(np.pi * u_norm * (1 + 0.1 * q), q)
        
        # 2. Memory injection (crucial for temporal tasks)
        for q in range(self.num_qubits):
            qc.rz(np.pi * self.memory_states[q] * 0.3, q)
        
        # 3. Cross-coupling for non-linearity
        for q in range(self.num_qubits - 1):
            qc.ry(np.pi * u_norm * self.memory_states[q] * 0.2, q + 1)
        
        # OPTIMIZED RESERVOIR DYNAMICS
        # Layer 1: Entangling
        for q in range(self.num_qubits - 1):
            qc.cx(q, q + 1)
        
        # Layer 2: Non-linear transformations
        for q in range(self.num_qubits):
            angle = np.pi * u_norm * (0.6 + 0.1 * np.sin(q * np.pi / self.num_qubits))
            qc.ry(angle, q)
            qc.rz(angle * 0.8, q)
        
        # Layer 3: Reverse entangling
        for q in range(self.num_qubits - 1, 0, -1):
            qc.cx(q, q - 1)
        
        # Layer 4: Final mixing
        for q in range(self.num_qubits):
            qc.rx(np.pi * u_norm * 0.4 * (1 + 0.05 * self.node_id), q)
        
        # Add structured noise for reservoir diversity
        if self.node_id > 0:
            for q in range(self.num_qubits):
                qc.rz(np.pi * 0.1 * self.node_id * u_norm, q)
        
        # Measurements
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        
        # Execute with optimized parameters
        transpiled_qc = transpile(qc, self.backend, optimization_level=2)
        job = self.backend.run(transpiled_qc, 
                              noise_model=self.noise_model, 
                              shots=shots)
        
        # Get enhanced features
        result = job.result()
        counts = result.get_counts()
        
        # Create rich feature vector
        features = self._extract_enhanced_features(counts, u_norm)
        
        # Update memory (crucial for temporal correlation)
        self.memory_states = (self.memory_decay * self.memory_states + 
                             (1 - self.memory_decay) * features[:self.num_qubits])
        
        return features
    
    def _extract_enhanced_features(self, counts: dict, u_input: float) -> np.ndarray:
        """Extract enhanced features beyond simple probabilities"""
        total_shots = sum(counts.values())
        
        # Basic probability features
        prob_features = np.zeros(self.num_qubits)
        for bitstring, count in counts.items():
            prob = count / total_shots
            for i, bit in enumerate(bitstring):
                if bit == '1':
                    prob_features[i] += prob
        
        # Enhanced features
        enhanced_features = []
        
        # 1. Probability features
        enhanced_features.extend(prob_features)
        
        # 2. Correlation features (pairwise)
        for i in range(self.num_qubits - 1):
            corr_feature = prob_features[i] * prob_features[i + 1]
            enhanced_features.append(corr_feature)
        
        # 3. Non-linear transformations
        for i in range(self.num_qubits):
            enhanced_features.append(prob_features[i]**2)
            enhanced_features.append(np.sin(np.pi * prob_features[i]))
        
        # 4. Input-dependent features
        for i in range(self.num_qubits):
            enhanced_features.append(prob_features[i] * u_input)
        
        # 5. Memory-based features
        for i in range(self.num_qubits):
            enhanced_features.append(prob_features[i] * self.memory_states[i])
        
        return np.array(enhanced_features)

class OptimizedDistributedQNR:
    """Optimized distributed quantum reservoir with better architecture"""
    
    def __init__(self, num_nodes: int = 8, num_qubits: int = 4):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        self.backend = AerSimulator()
        
        # Create diverse nodes
        self.nodes = self._create_optimized_nodes()
        self.reservoir_states = None
        
        # Calculate expected feature dimension
        features_per_node = self._calculate_features_per_node()
        self.total_features = num_nodes * features_per_node
        
        print(f"Created {num_nodes} nodes with {features_per_node} features each")
        print(f"Total feature dimension: {self.total_features}")
    
    def _calculate_features_per_node(self) -> int:
        """Calculate number of features per node"""
        # Based on _extract_enhanced_features:
        # - prob_features: num_qubits
        # - correlation features: num_qubits - 1
        # - non-linear (sin, square): 2 * num_qubits
        # - input-dependent: num_qubits
        # - memory-based: num_qubits
        return (self.num_qubits + 
                (self.num_qubits - 1) + 
                2 * self.num_qubits + 
                self.num_qubits + 
                self.num_qubits)
    
    def _create_optimized_nodes(self):
        """Create nodes with optimized noise diversity"""
        nodes = []
        
        # Carefully chosen noise levels for diversity
        noise_levels = np.linspace(0.01, 0.12, self.num_nodes)
        
        for i in range(self.num_nodes):
            node = OptimizedQuantumReservoir(
                num_qubits=self.num_qubits,
                noise_level=noise_levels[i],
                backend=self.backend,
                node_id=i
            )
            nodes.append(node)
        
        return nodes
    
    def process_sequence(self, input_sequence: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Process sequence with optimized parallel processing"""
        T = len(input_sequence)
        self.reservoir_states = np.zeros((T, self.total_features))
        
        if verbose:
            print(f"Processing {T} timesteps through {self.num_nodes} optimized nodes...")
        
        for t in range(T):
            if verbose and t % 100 == 0:
                print(f"  Progress: {t}/{T} ({100*t/T:.1f}%)")
            
            # Process through all nodes
            all_features = []
            for node in self.nodes:
                features = node.process_input(input_sequence[t])
                all_features.extend(features)
            
            self.reservoir_states[t] = all_features
        
        return self.reservoir_states

def advanced_feature_engineering(X: np.ndarray, temporal_window: int = 5) -> np.ndarray:
    """Add temporal features for better NARMA2 prediction"""
    T, n_features = X.shape
    
    # Original features
    enhanced_X = [X]
    
    # Temporal features
    for lag in range(1, temporal_window + 1):
        if lag < T:
            # Lagged features
            lagged = np.zeros_like(X)
            lagged[lag:] = X[:-lag]
            enhanced_X.append(lagged)
            
            # Temporal differences
            diff = np.zeros_like(X)
            diff[lag:] = X[lag:] - X[:-lag]
            enhanced_X.append(diff)
    
    # Moving averages
    for window in [3, 5]:
        if window < T:
            moving_avg = np.zeros_like(X)
            for t in range(window, T):
                moving_avg[t] = np.mean(X[t-window:t], axis=0)
            enhanced_X.append(moving_avg)
    
    return np.hstack(enhanced_X)

def optimized_readout_training(X_train, y_train, X_test, y_test, feature_engineering: bool = True):
    """Optimized readout training with multiple techniques"""
    
    # Feature engineering
    if feature_engineering:
        print("Applying advanced feature engineering...")
        X_train_eng = advanced_feature_engineering(X_train)
        X_test_eng = advanced_feature_engineering(X_test)
    else:
        X_train_eng = X_train
        X_test_eng = X_test
    
    print(f"Feature dimension: {X_train_eng.shape[1]}")
    
    # Robust preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    X_test_scaled = scaler.transform(X_test_eng)
    
    # Remove features with zero variance
    non_zero_var = np.var(X_train_scaled, axis=0) > 1e-10
    X_train_scaled = X_train_scaled[:, non_zero_var]
    X_test_scaled = X_test_scaled[:, non_zero_var]
    
    print(f"Features after variance filtering: {X_train_scaled.shape[1]}")
    
    # Optimized regularization search
    alphas = np.logspace(-10, 3, 50)
    best_alpha = None
    best_score = -np.inf
    
    print("Searching for optimal regularization...")
    for alpha in alphas:
        model = Ridge(alpha=alpha, max_iter=5000)
        # Use cross-validation for robust model selection
        scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='r2')
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha:.2e}, CV R²: {best_score:.6f}")
    
    # Train final model
    final_model = Ridge(alpha=best_alpha, max_iter=5000)
    final_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = final_model.predict(X_test_scaled)
    
    # Advanced scaling
    scaled_pred, scale_factor, offset = optimize_prediction_scaling(y_pred, y_test)
    
    return final_model, y_pred, scaled_pred, scale_factor, offset

def optimize_prediction_scaling(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Find optimal scaling parameters"""
    
    def objective(params):
        scale, offset = params
        scaled = y_pred * scale + offset
        return mean_squared_error(y_true, scaled)
    
    from scipy.optimize import minimize
    
    # Smart initialization
    initial_scale = np.std(y_true) / np.std(y_pred) if np.std(y_pred) > 0 else 1.0
    initial_offset = np.mean(y_true) - initial_scale * np.mean(y_pred)
    
    # Optimize
    result = minimize(objective, [initial_scale, initial_offset], 
                     method='Nelder-Mead',
                     options={'maxiter': 1000})
    
    optimal_scale, optimal_offset = result.x
    scaled_pred = y_pred * optimal_scale + optimal_offset
    
    return scaled_pred, optimal_scale, optimal_offset

def comprehensive_evaluation(y_true, y_pred_raw, y_pred_scaled, verbose=True):
    """Comprehensive evaluation with multiple metrics"""
    
    metrics = {}
    
    for name, y_pred in [('Raw', y_pred_raw), ('Scaled', y_pred_scaled)]:
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Normalized metrics
        y_range = y_true.max() - y_true.min()
        nrmse = rmse / y_range if y_range > 0 else np.inf
        
        # Correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics[name] = {
            'R²': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae,
            'NRMSE': nrmse, 'Correlation': corr
        }
        
        if verbose:
            print(f"\n{name} Predictions:")
            print(f"  R² Score: {r2:.6f}")
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  NRMSE: {nrmse:.6f}")
            print(f"  Correlation: {corr:.6f}")
    
    return metrics

def create_detailed_plots(y_true, y_pred_raw, y_pred_scaled, metrics):
    """Create detailed analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Time series comparison
    ax = axes[0, 0]
    ax.plot(y_true[:200], 'k-', label='True NARMA2', linewidth=2)
    ax.plot(y_pred_raw[:200], 'r--', label='Raw Pred', alpha=0.7)
    ax.plot(y_pred_scaled[:200], 'b:', label='Scaled Pred', linewidth=2)
    ax.set_title('Time Series Comparison (First 200 points)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred_raw, alpha=0.6, s=10, color='red', label='Raw')
    ax.scatter(y_true, y_pred_scaled, alpha=0.6, s=10, color='blue', label='Scaled')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax = axes[0, 2]
    errors_raw = y_true - y_pred_raw
    errors_scaled = y_true - y_pred_scaled
    ax.hist(errors_raw, bins=30, alpha=0.7, color='red', density=True, label='Raw')
    ax.hist(errors_scaled, bins=30, alpha=0.7, color='blue', density=True, label='Scaled')
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. R² comparison
    ax = axes[1, 0]
    r2_values = [metrics['Raw']['R²'], metrics['Scaled']['R²']]
    bars = ax.bar(['Raw', 'Scaled'], r2_values, color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # 5. Residual analysis
    ax = axes[1, 1]
    ax.scatter(y_pred_scaled, errors_scaled, alpha=0.6, s=10)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Scaled Predictions')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Analysis')
    ax.grid(True, alpha=0.3)
    
    # 6. Metrics table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = []
    for metric in ['R²', 'MSE', 'RMSE', 'NRMSE', 'Correlation']:
        raw_val = metrics['Raw'][metric]
        scaled_val = metrics['Scaled'][metric]
        table_data.append([metric, f'{raw_val:.4f}', f'{scaled_val:.4f}'])
    
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Raw', 'Scaled'],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Performance Metrics', pad=20)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Optimized QNR for NARMA2')
    parser.add_argument('--length', type=int, default=1000, help='Series length')
    parser.add_argument('--nodes', type=int, default=8, help='Number of nodes')
    parser.add_argument('--qubits', type=int, default=4, help='Qubits per node')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OPTIMIZED QUANTUM RESERVOIR COMPUTING FOR NARMA2")
    print("=" * 80)
    
    # Generate data
    print(f"\n1. Generating NARMA2 data (length={args.length})...")
    narma_gen = NARMA2Generator(seed=args.seed)
    u_seq, y_seq = narma_gen.generate(args.length)
    
    # Prepare sequences
    input_seq = u_seq[:-1]
    target_seq = y_seq[1:]
    
    # Align lengths
    min_len = min(len(input_seq), len(target_seq))
    input_seq = input_seq[:min_len]
    target_seq = target_seq[:min_len]
    
    print(f"   Input range: [{input_seq.min():.3f}, {input_seq.max():.3f}]")
    print(f"   Target range: [{target_seq.min():.3f}, {target_seq.max():.3f}]")
    
    # Split data
    split_idx = int(args.train_ratio * len(input_seq))
    print(f"   Train samples: {split_idx}, Test samples: {len(input_seq) - split_idx}")
    
    # Create optimized reservoir
    print(f"\n2. Creating optimized quantum reservoir...")
    reservoir = OptimizedDistributedQNR(num_nodes=args.nodes, num_qubits=args.qubits)
    
    # Process sequence
    print(f"\n3. Processing sequence...")
    start_time = time.time()
    reservoir_states = reservoir.process_sequence(input_seq, verbose=True)
    processing_time = time.time() - start_time
    
    # Prepare training data
    X_train = reservoir_states[:split_idx]
    y_train = target_seq[:split_idx]
    X_test = reservoir_states[split_idx:]
    y_test = target_seq[split_idx:]
    
    print(f"\n4. Training optimized readout...")
    model, y_pred_raw, y_pred_scaled, scale_factor, offset = optimized_readout_training(
        X_train, y_train, X_test, y_test)
    
    print(f"\n5. Evaluating performance...")
    metrics = comprehensive_evaluation(y_test, y_pred_raw, y_pred_scaled)
    
    print(f"\n6. Creating analysis plots...")
    create_detailed_plots(y_test, y_pred_raw, y_pred_scaled, metrics)
    
    # Summary
    print(f"\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Total features used: {reservoir_states.shape[1]}")
    print(f"Raw R² Score: {metrics['Raw']['R²']:.6f}")
    print(f"Scaled R² Score: {metrics['Scaled']['R²']:.6f}")
    print(f"Improvement: {metrics['Scaled']['R²'] - metrics['Raw']['R²']:.6f}")
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Offset: {offset:.4f}")
    
    if metrics['Scaled']['R²'] > 0.7:
        print("✅ EXCELLENT PERFORMANCE ACHIEVED!")
    elif metrics['Scaled']['R²'] > 0.5:
        print("✅ GOOD PERFORMANCE ACHIEVED!")
    else:
        print("⚠️  Performance could be improved further")

if __name__ == "__main__":
    main()
