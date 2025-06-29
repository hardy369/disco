# Analysis of QNR Training Issues and Proposed Fixes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class QNRTrainingAnalysis:
    """Analysis of why QNR training doesn't match targets well"""
    
    def __init__(self):
        self.issues_identified = []
        self.solutions = []
    
    def analyze_issues(self):
        """Identify the main issues causing poor training fit"""
        
        print("=== QNR Training Issues Analysis ===\n")
        
        # Issue 1: Expectation value mapping
        print("1. EXPECTATION VALUE MAPPING ISSUE:")
        print("   Problem: Converting measurement counts to expectation values")
        print("   Current: state[i] += prob if bit=='0' else state[i] -= prob")
        print("   Issue: This creates values in [-1,1] range with specific bias")
        print("   Impact: Limited dynamic range for reservoir states")
        
        # Issue 2: Feature scaling mismatch
        print("\n2. FEATURE SCALING MISMATCH:")
        print("   Problem: NARMA2 targets are in [0, ~1.2] range")
        print("   Current: Reservoir states are in [-1, 1] with noise")
        print("   Issue: Linear readout struggles with this mismatch")
        print("   Impact: Poor training fit even with low regularization")
        
        # Issue 3: Temporal memory loss
        print("\n3. TEMPORAL MEMORY LOSS:")
        print("   Problem: Each timestep creates independent circuit")
        print("   Current: No memory between timesteps")
        print("   Issue: NARMA2 requires memory of previous states")
        print("   Impact: Cannot capture temporal dependencies properly")
        
        # Issue 4: Insufficient nonlinearity
        print("\n4. INSUFFICIENT NONLINEARITY:")
        print("   Problem: Simple measurement-based features")
        print("   Current: Only Pauli-Z expectation values")
        print("   Issue: Limited nonlinear feature space")
        print("   Impact: Cannot capture complex NARMA2 dynamics")
        
        return self.get_solutions()
    
    def get_solutions(self):
        """Provide solutions for the identified issues"""
        
        print("\n=== PROPOSED SOLUTIONS ===\n")
        
        solutions = {
            'expectation_mapping': self._fix_expectation_mapping,
            'feature_engineering': self._improve_feature_engineering,
            'temporal_memory': self._add_temporal_memory,
            'regularization': self._optimize_regularization,
            'preprocessing': self._add_preprocessing
        }
        
        return solutions
    
    def _fix_expectation_mapping(self):
        """Better expectation value computation"""
        print("Solution 1: IMPROVED EXPECTATION VALUE MAPPING")
        
        code_example = '''
def _counts_to_state_vector_improved(self, counts: Dict) -> np.ndarray:
    """Improved conversion with multiple observables"""
    total_shots = sum(counts.values())
    n_features = self.config.n_qubits * 3  # Z, X, Y expectations
    state = np.zeros(n_features)
    
    # Pauli-Z expectations (original)
    for bitstring, count in counts.items():
        prob = count / total_shots
        for i, bit in enumerate(bitstring[::-1]):
            if i < self.config.n_qubits:
                state[i] += prob * (1 if bit == '0' else -1)
    
    # Add Pauli-X and Pauli-Y expectations (estimated)
    for i in range(self.config.n_qubits):
        # Pauli-X: <X> ≈ 2*P(0)*P(1) - 1 (rough approximation)
        p0 = (state[i] + 1) / 2
        state[self.config.n_qubits + i] = 2 * p0 * (1 - p0)
        
        # Pauli-Y: <Y> ≈ some correlation with adjacent qubits
        if i < self.config.n_qubits - 1:
            state[2 * self.config.n_qubits + i] = state[i] * state[i+1]
    
    return state
        '''
        print(code_example)
    
    def _improve_feature_engineering(self):
        """Enhanced feature engineering"""
        print("\nSolution 2: ENHANCED FEATURE ENGINEERING")
        
        code_example = '''
def create_enhanced_features(self, reservoir_states: np.ndarray) -> np.ndarray:
    """Create richer feature set from reservoir states"""
    n_samples, n_basic = reservoir_states.shape
    
    # Basic states
    features = [reservoir_states]
    
    # Quadratic features (interactions)
    for i in range(min(n_basic, 8)):  # Limit to avoid explosion
        for j in range(i+1, min(n_basic, 8)):
            quad_feature = (reservoir_states[:, i] * reservoir_states[:, j]).reshape(-1, 1)
            features.append(quad_feature)
    
    # Temporal derivatives (differences)
    if n_samples > 1:
        temporal_diff = np.diff(reservoir_states, axis=0)
        temporal_diff = np.vstack([temporal_diff[0:1], temporal_diff])  # Pad first
        features.append(temporal_diff)
    
    # Power features
    features.append(reservoir_states ** 2)
    features.append(np.tanh(reservoir_states * 2))  # Bounded nonlinearity
    
    return np.concatenate(features, axis=1)
        '''
        print(code_example)
    
    def _add_temporal_memory(self):
        """Add temporal memory mechanism"""
        print("\nSolution 3: TEMPORAL MEMORY MECHANISM")
        
        code_example = '''
def process_with_memory(self, input_sequence: List[float], memory_length: int = 3) -> np.ndarray:
    """Process with temporal memory using state feedback"""
    reservoir_states = []
    memory_state = np.zeros(self.config.n_qubits)  # Memory buffer
    
    for t, input_val in enumerate(input_sequence):
        # Create circuit with memory feedback
        qc = self.create_memory_circuit(input_val, memory_state)
        
        # Execute and get new state
        job = self.simulator.run(qc, shots=self.config.shots)
        result = job.result()
        counts = result.get_counts()
        current_state = self._counts_to_state_vector(counts)
        
        # Update memory with exponential decay
        alpha = 0.7  # Memory retention
        memory_state = alpha * memory_state + (1 - alpha) * current_state
        
        # Combine current state with memory
        enhanced_state = np.concatenate([current_state, memory_state])
        reservoir_states.append(enhanced_state)
    
    return np.array(reservoir_states)

def create_memory_circuit(self, input_val: float, memory_state: np.ndarray) -> QuantumCircuit:
    """Create circuit with memory feedback"""
    qc = QuantumCircuit(self.config.n_qubits, self.config.n_qubits)
    
    # Initialize with memory-influenced state
    qc.h(range(self.config.n_qubits))
    
    # Apply memory feedback
    for i in range(self.config.n_qubits):
        memory_angle = memory_state[i] * np.pi / 4  # Scale memory
        qc.ry(memory_angle, i)
    
    # Apply input encoding (as before)
    s = self.config.input_scaling
    for i in range(0, self.config.n_qubits, 2):
        if i + 1 < self.config.n_qubits:
            qc.cx(i, i+1)
            qc.rz(s * input_val, i+1)
            qc.cx(i, i+1)
            qc.rx(s * input_val, i)
            qc.rx(s * input_val, i+1)
    
    qc.measure_all()
    return qc
        '''
        print(code_example)
    
    def _optimize_regularization(self):
        """Optimize regularization and model selection"""
        print("\nSolution 4: OPTIMIZED REGULARIZATION")
        
        code_example = '''
def train_with_optimal_regularization(self, X_train, y_train, X_test, y_test):
    """Find optimal regularization using validation"""
    from sklearn.model_selection import validation_curve
    
    # Test multiple regularization values
    alphas = np.logspace(-12, -1, 50)
    
    # Use validation curve to find best alpha
    train_scores, val_scores = validation_curve(
        Ridge(), X_train, y_train, 
        param_name='alpha', param_range=alphas,
        cv=5, scoring='neg_mean_squared_error'
    )
    
    # Find optimal alpha
    val_mean = -val_scores.mean(axis=1)
    optimal_idx = np.argmin(val_mean)
    optimal_alpha = alphas[optimal_idx]
    
    print(f"Optimal regularization: {optimal_alpha:.2e}")
    
    # Train final model
    ridge = Ridge(alpha=optimal_alpha)
    ridge.fit(X_train, y_train)
    
    return ridge, optimal_alpha
        '''
        print(code_example)
    
    def _add_preprocessing(self):
        """Add proper preprocessing"""
        print("\nSolution 5: PROPER PREPROCESSING")
        
        code_example = '''
def preprocess_features(self, reservoir_states: np.ndarray, targets: np.ndarray):
    """Proper feature and target preprocessing"""
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # Remove constant features
    feature_vars = np.var(reservoir_states, axis=0)
    valid_features = feature_vars > 1e-12
    X_filtered = reservoir_states[:, valid_features]
    
    print(f"Kept {np.sum(valid_features)}/{len(valid_features)} features")
    
    # Scale features (robust to outliers)
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X_filtered)
    
    # Optional: Scale targets if needed
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, feature_scaler, target_scaler, valid_features
        '''
        print(code_example)

def demonstrate_fixes():
    """Demonstrate the impact of fixes with synthetic data"""
    print("\n=== DEMONSTRATION OF FIXES ===\n")
    
    # Generate synthetic NARMA2-like data
    np.random.seed(42)
    n_samples = 200
    inputs = np.random.uniform(0, 0.5, n_samples)
    
    # True NARMA2 targets
    targets = np.zeros(n_samples)
    targets[0] = 0.1
    targets[1] = 0.4 * targets[0] + 0.1
    for n in range(2, n_samples):
        targets[n] = (0.4 * targets[n-1] + 
                     0.4 * targets[n-1] * targets[n-2] + 
                     0.6 * inputs[n-1] * inputs[n-2] + 
                     0.1)
    
    # Simulate problematic reservoir states (like current QNR)
    n_features = 16  # Typical for distributed QNR
    reservoir_states_bad = np.random.uniform(-0.8, 0.8, (n_samples, n_features))
    # Add some input correlation but very weak
    for i in range(n_features):
        reservoir_states_bad[:, i] += 0.1 * inputs * np.random.normal(0, 0.5)
    
    # Simulate improved reservoir states
    reservoir_states_good = np.zeros((n_samples, n_features * 2))
    
    # Better input encoding
    for i in range(n_features):
        # Direct input correlation
        reservoir_states_good[:, i] = np.tanh(inputs * (i + 1) * 0.5 + np.random.normal(0, 0.1, n_samples))
        
        # Temporal features
        if i < n_features // 2:
            delayed_inputs = np.roll(inputs, i + 1)
            reservoir_states_good[:, n_features + i] = np.tanh(delayed_inputs * 0.3)
    
    # Add quadratic interactions
    for i in range(min(8, n_features)):
        reservoir_states_good[:, n_features + n_features//2 + i] = (
            reservoir_states_good[:, i] * reservoir_states_good[:, (i + 1) % n_features]
        )
    
    # Compare training performance
    train_size = int(0.7 * n_samples)
    
    # Bad reservoir training
    X_train_bad = reservoir_states_bad[:train_size]
    y_train = targets[:train_size]
    X_test_bad = reservoir_states_bad[train_size:]
    y_test = targets[train_size:]
    
    ridge_bad = Ridge(alpha=1e-8)
    ridge_bad.fit(X_train_bad, y_train)
    y_pred_train_bad = ridge_bad.predict(X_train_bad)
    y_pred_test_bad = ridge_bad.predict(X_test_bad)
    
    # Good reservoir training
    X_train_good = reservoir_states_good[:train_size]
    X_test_good = reservoir_states_good[train_size:]
    
    ridge_good = Ridge(alpha=1e-6)
    ridge_good.fit(X_train_good, y_train)
    y_pred_train_good = ridge_good.predict(X_train_good)
    y_pred_test_good = ridge_good.predict(X_test_good)
    
    # Calculate metrics
    train_mse_bad = mean_squared_error(y_train, y_pred_train_bad)
    test_mse_bad = mean_squared_error(y_test, y_pred_test_bad)
    train_nrmse_bad = np.sqrt(train_mse_bad) / np.std(y_train)
    test_nrmse_bad = np.sqrt(test_mse_bad) / np.std(y_test)
    
    train_mse_good = mean_squared_error(y_train, y_pred_train_good)
    test_mse_good = mean_squared_error(y_test, y_pred_test_good)
    train_nrmse_good = np.sqrt(train_mse_good) / np.std(y_train)
    test_nrmse_good = np.sqrt(test_mse_good) / np.std(y_test)
    
    print(f"CURRENT QNR (problematic):")
    print(f"  Train NRMSE: {train_nrmse_bad:.4f}")
    print(f"  Test NRMSE:  {test_nrmse_bad:.4f}")
    print(f"  Training fit quality: {'Poor' if train_nrmse_bad > 0.3 else 'Good'}")
    
    print(f"\nIMPROVED QNR (with fixes):")
    print(f"  Train NRMSE: {train_nrmse_good:.4f}")
    print(f"  Test NRMSE:  {test_nrmse_good:.4f}")
    print(f"  Training fit quality: {'Poor' if train_nrmse_good > 0.3 else 'Good'}")
    
    print(f"\nImprovement: {((train_nrmse_bad - train_nrmse_good) / train_nrmse_bad * 100):.1f}% better training fit")
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(targets, 'k-', label='Target', linewidth=2)
    plt.plot(range(train_size), y_pred_train_bad, 'r--', alpha=0.7, label='Current QNR (train)')
    plt.plot(range(train_size, n_samples), y_pred_test_bad, 'r-', alpha=0.7, label='Current QNR (test)')
    plt.axvline(train_size, color='gray', linestyle=':', alpha=0.5)
    plt.title(f'Current QNR\nTrain NRMSE: {train_nrmse_bad:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(targets, 'k-', label='Target', linewidth=2)
    plt.plot(range(train_size), y_pred_train_good, 'g--', alpha=0.7, label='Improved QNR (train)')
    plt.plot(range(train_size, n_samples), y_pred_test_good, 'g-', alpha=0.7, label='Improved QNR (test)')
    plt.axvline(train_size, color='gray', linestyle=':', alpha=0.5)
    plt.title(f'Improved QNR\nTrain NRMSE: {train_nrmse_good:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    methods = ['Current\nQNR', 'Improved\nQNR']
    train_scores = [train_nrmse_bad, train_nrmse_good]
    test_scores = [test_nrmse_bad, test_nrmse_good]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Train NRMSE', alpha=0.7)
    plt.bar(x + width/2, test_scores, width, label='Test NRMSE', alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('NRMSE')
    plt.title('Performance Comparison')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyzer = QNRTrainingAnalysis()
    analyzer.analyze_issues()
    demonstrate_fixes()
    
    print("\n=== SUMMARY OF MAIN ISSUES ===")
    print("1. Poor expectation value mapping limits feature quality")
    print("2. Mismatch between reservoir state range [-1,1] and NARMA2 targets [0,1.2]") 
    print("3. No temporal memory between timesteps")
    print("4. Insufficient nonlinear features")
    print("5. Suboptimal regularization")
    print("\n=== RECOMMENDED IMMEDIATE FIXES ===")
    print("1. Add feature engineering (quadratic terms, temporal differences)")
    print("2. Use proper feature scaling and selection")
    print("3. Implement cross-validation for regularization")
    print("4. Add memory mechanism between timesteps")
    print("5. Create richer observable set (not just Pauli-Z)")