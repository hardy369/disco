#!/usr/bin/env python3
"""
PARALLEL SPATIAL MULTIPLEXING for Quantum Reservoir Computing

This script implements the parallel spatial multiplexing architecture,
configured for a large-scale NARMA2 benchmark test.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import time
import warnings
import multiprocessing
from itertools import repeat

warnings.filterwarnings('ignore')

class NARMA2Generator:
    """
    Generates the second-order nonlinear autoregressive moving average
    (NARMA2) sequence. This is a common benchmark for testing a system's
    ability to learn nonlinear dynamics with memory.
    """
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate(self, T: int, tau: int = 2) -> tuple:
        u = np.random.uniform(0.0, 0.5, T)
        y = np.zeros(T)
        y[0] = y[1] = 0.3
        for t in range(tau, T):
            y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * y[t-2] + 1.5 * u[t-1] * u[t-2] + 0.1)
        return u, y

class QuantumReservoir:
    """
    Represents a single Quantum Noise-induced Reservoir (QNR), where inherent
    quantum noise is leveraged for computation.
    """
    def __init__(self, num_qubits: int = 4, backend=None):
        self.num_qubits = num_qubits
        if self.num_qubits % 2 != 0:
            raise ValueError("Number of qubits must be even for the paper's circuit.")
        self.backend = backend if backend else AerSimulator()
        self.noise_model = None
        if hasattr(self.backend.configuration(), 'simulator') and not self.backend.configuration().simulator:
             self.noise_model = NoiseModel.from_backend(self.backend)

    def process_batch(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        batch_size = len(inputs)
        features = np.zeros((batch_size, self.num_qubits))
        circuits = []
        input_scaling_s = np.pi
        for u_t in inputs:
            qc = QuantumCircuit(self.num_qubits)
            for i in range(0, self.num_qubits, 2):
                q1, q2 = i, i + 1
                # This circuit structure is based on the model from the research.
                angle = input_scaling_s * u_t
                qc.rx(angle, q1); qc.rx(angle, q2)
                qc.cx(q1, q2); qc.rz(angle, q2); qc.cx(q1, q2)
            qc.measure_all()
            circuits.append(qc)
        transpiled_circuits = transpile(circuits, self.backend, optimization_level=1)
        result = self.backend.run(transpiled_circuits, noise_model=self.noise_model, shots=shots).result()
        for i in range(batch_size):
            features[i] = self._extract_features(result.get_counts(i))
        return features
    
    def _extract_features(self, counts: dict) -> np.ndarray:
        """
        Extracts features as the expectation value of the Pauli Z operator for each qubit.
        This corresponds to the reservoir state x_i,t = Tr(ρ_t * O_i), a method
        described in the research.
        """
        total_shots = sum(counts.values())
        expectation_values = np.zeros(self.num_qubits)
        for q in range(self.num_qubits):
            prob_0 = sum(count for bitstring, count in counts.items() if bitstring[self.num_qubits - 1 - q] == '0')
            expectation_values[q] = (prob_0 / total_shots) * 2 - 1 if total_shots > 0 else 0
        return expectation_values

def run_node_processing(node, batch_inputs):
    """A helper function to be called by each worker process in parallel."""
    return node.process_batch(batch_inputs)

class DistributedQNR:
    """
    Manages multiple QNRs using spatial multiplexing and executes them in parallel.
    The concept of boosting computational power by combining multiple disjoint
    reservoirs is called spatial multiplexing.
    """
    def __init__(self, num_nodes: int, num_qubits: int, base_backend):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        self.nodes = [QuantumReservoir(num_qubits, base_backend) for _ in range(num_nodes)]
        self.total_features = num_nodes * num_qubits
        print(f"Created {num_nodes} reservoir node(s) with {self.total_features} total features.")
    
    def process_sequence(self, input_sequence: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """Processes the sequence in batches across all nodes IN PARALLEL."""
        T = len(input_sequence)
        all_features = np.zeros((T, self.total_features))
        print(f"Processing {T} timesteps using distributed computing...")
        with multiprocessing.Pool() as pool:
            for start_idx in range(0, T, batch_size):
                end_idx = min(start_idx + batch_size, T)
                batch_inputs = input_sequence[start_idx:end_idx]
                if start_idx % (batch_size * 100) == 0:
                    print(f"  Progress: {start_idx}/{T} ({100*start_idx/T:.1f}%) - Distributing to {pool._processes} workers...")
                args = zip(self.nodes, repeat(batch_inputs))
                node_features = pool.starmap(run_node_processing, args)
                batch_features = np.hstack(node_features)
                all_features[start_idx:end_idx] = batch_features
        return all_features

def _temporal_features(X: np.ndarray, max_lags: int = 3) -> np.ndarray:
    """Creates temporal features (lags) to provide memory to the readout model."""
    T, n_features = X.shape
    features = [X]
    for lag in range(1, max_lags + 1):
        lagged = np.roll(X, lag, axis=0); lagged[:lag] = 0
        features.append(lagged)
    return np.hstack(features)

def _readout_training(X_train, y_train, X_test, y_test):
    """Trains the final linear readout model."""
    X_train_eng = _temporal_features(X_train)
    X_test_eng = _temporal_features(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    X_test_scaled = scaler.transform(X_test_eng)
    ridge_model = Ridge(alpha=5.0)
    ridge_model.fit(X_train_scaled, y_train)
    return {'Ridge': ridge_model.predict(X_test_scaled)}

def evaluate_and_plot(y_true, predictions):
    """Calculates and prints performance metrics and creates plots."""
    print("\n" + "-"*25 + " RESULTS " + "-"*25)
    name, y_pred = list(predictions.items())[0]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    nmse = numerator / denominator if denominator != 0 else float('inf')
    print(f"{name} | R²: {r2:7.4f} | RMSE: {rmse:7.4f} | NMSE: {nmse:7.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(y_true[:200], 'k-', label='True NARMA2', linewidth=2)
    axes[0].plot(y_pred[:200], 'r--', label='QRC Prediction', linewidth=1.5)
    axes[0].set_title('Time Series Comparison'); axes[0].legend(); axes[0].grid(True, alpha=0.4)
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=15, c='blue')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    axes[1].set_xlabel('True Values'); axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('True vs. Predicted Values'); axes[1].grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()

def main():
    """Main execution function."""
    print("=" * 60)
    print(" PARALLEL SPATIAL MULTIPLEXING QRC FOR NARMA2 (LARGE SCALE)")
    print("=" * 60)
    
    # --- Parameters Updated as Requested ---
    LENGTH = 50000  # Total time steps for the NARMA2 sequence.
    NODES = 25      # Number of reservoirs for spatial multiplexing.
    QUBITS = 4      # Number of qubits in each reservoir (must be even).
    TRAIN_RATIO = 0.75
    
    print(f"\nParameters: Length={LENGTH}, Nodes={NODES}, Qubits={QUBITS}")
    
    print("\n1. Initializing backend...")
    backend = FakeVigoV2() 
    
    print("\n2. Generating NARMA2 data...")
    u_seq, y_seq = NARMA2Generator(seed=42).generate(LENGTH)
    input_seq, target_seq = u_seq[:-1], y_seq[1:]
    split_idx = int(TRAIN_RATIO * len(input_seq))
    
    print(f"\n3. Running distributed quantum reservoir...")
    reservoir = DistributedQNR(num_nodes=NODES, num_qubits=QUBITS, base_backend=backend)
    
    start_time = time.time()
    reservoir_states = reservoir.process_sequence(input_seq, batch_size=25)
    print(f"   Processing completed in {time.time() - start_time:.2f} seconds")
    
    X_train, y_train = reservoir_states[:split_idx], target_seq[:split_idx]
    X_test, y_test = reservoir_states[split_idx:], target_seq[split_idx:]
    
    print(f"\n4. Training and evaluating...")
    predictions = _readout_training(X_train, y_train, X_test, y_test)
    evaluate_and_plot(y_test, predictions)

if __name__ == "__main__":
    main()