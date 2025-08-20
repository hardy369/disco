# Quantum Noise-Induced Reservoir Computing (QNR)


## Overview
This project implements **Quantum Noise-Induced Reservoirs (QNRs)** — a framework where quantum noise, often considered detrimental, is instead used as a **computational resource** for temporal information processing.

**Key idea:**
- A noiseless quantum circuit may fail to retain input information.
- Introducing noise channels (e.g., amplitude damping, depolarization, phase-flip, unitary imperfections) enables meaningful input-driven dynamics.
- This turns noisy intermediate-scale quantum (NISQ) devices into powerful **reservoir computers**.

---

##  Features
- Simulation of QNRs with various noise models:
  - Amplitude damping
  - Phase damping
  - Depolarization
  - Bit-flip, Phase-flip
  - Unitary noise (CNOT bias, unintended entanglement)
- Benchmarks:
  - **NARMA2** sequence prediction
- Experiments on **IBM Quantum devices** (via Qiskit).

---

Make sure all the dependencies are installed (both python and qiskit are required.)
