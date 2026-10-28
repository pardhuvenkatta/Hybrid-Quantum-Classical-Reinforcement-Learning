import numpy as np
from typing import Dict, Tuple, Optional

class QuantumNoiseModel:
    """
    Simulates physical NISQ quantum hardware noise channels:
    1. Depolarizing Noise (gate error)
    2. Amplitude Damping (energy loss / T1 decay)
    3. Dephasing (phase damping / T2 decay)
    4. Measurement Readout Bit-Flip Error
    """
    def __init__(
        self,
        name: str = "Noise-Free",
        depolarizing_p: float = 0.0,
        amplitude_damping_gamma: float = 0.0,
        dephasing_lambda: float = 0.0,
        measurement_error_p: float = 0.0
    ):
        self.name = name
        self.depolarizing_p = depolarizing_p
        self.amplitude_damping_gamma = amplitude_damping_gamma
        self.dephasing_lambda = dephasing_lambda
        self.measurement_error_p = measurement_error_p

    def is_noisy(self) -> bool:
        return (
            self.depolarizing_p > 0
            or self.amplitude_damping_gamma > 0
            or self.dephasing_lambda > 0
            or self.measurement_error_p > 0
        )

    def apply_statevector_noise(self, psi: np.ndarray) -> np.ndarray:
        """Apply single-qubit / multi-qubit physical noise Kraus channels to state vector amplitudes."""
        if not self.is_noisy():
            return psi

        N = len(psi)
        num_qubits = int(np.log2(N))
        psi_noisy = psi.copy().astype(complex)

        # 1. Amplitude Damping (T1 Decay): Decays state 1 to 0 with probability gamma
        if self.amplitude_damping_gamma > 0:
            gamma = self.amplitude_damping_gamma
            decay_mask = np.random.rand(N) < gamma
            for idx in range(N):
                if decay_mask[idx] and idx > 0:
                    # Shift state bit to 0 (decay to lower energy state)
                    decayed_idx = idx & (idx - 1)
                    psi_noisy[decayed_idx] += psi_noisy[idx] * np.sqrt(gamma)
                    psi_noisy[idx] *= np.sqrt(1.0 - gamma)

        # 2. Dephasing Noise (T2 Decay): Randomizes phase angles with probability lambda
        if self.dephasing_lambda > 0:
            lmbda = self.dephasing_lambda
            random_phases = np.random.normal(loc=0.0, scale=np.sqrt(lmbda), size=N)
            psi_noisy *= np.exp(1j * random_phases)

        # 3. Depolarizing Noise: Randomizes state vector direction with probability p
        if self.depolarizing_p > 0:
            p = self.depolarizing_p
            uniform_psi = np.ones(N, dtype=complex) / np.sqrt(N)
            psi_noisy = (1.0 - p) * psi_noisy + p * uniform_psi

        # Re-normalize state vector
        norm = np.linalg.norm(psi_noisy)
        if norm > 0:
            psi_noisy = psi_noisy / norm
        else:
            psi_noisy = np.ones(N, dtype=complex) / np.sqrt(N)

        return psi_noisy

    def apply_measurement_noise(self, counts: np.ndarray, shots: int) -> np.ndarray:
        """Apply probabilistic readout bit-flip error to measurement counts."""
        if self.measurement_error_p <= 0 or shots <= 0:
            return counts

        N = len(counts)
        num_qubits = int(np.log2(N))
        p_err = self.measurement_error_p
        noisy_counts = np.zeros_like(counts)

        for idx, count in enumerate(counts):
            if count == 0:
                continue
            for _ in range(count):
                # Flip bits with probability p_err
                bits = list(format(idx, f'0{num_qubits}b'))
                for b_i in range(len(bits)):
                    if np.random.rand() < p_err:
                        bits[b_i] = '1' if bits[b_i] == '0' else '0'
                flipped_idx = int("".join(bits), 2) % N
                noisy_counts[flipped_idx] += 1

        return noisy_counts


# Factory Constructors for Standard Noise Conditions
def noise_free_model() -> QuantumNoiseModel:
    return QuantumNoiseModel(name="Noise-Free")

def depolarizing_noise_model(p: float = 0.005) -> QuantumNoiseModel:
    return QuantumNoiseModel(name=f"Depolarizing (p={p})", depolarizing_p=p)

def amplitude_damping_model(gamma: float = 0.01) -> QuantumNoiseModel:
    return QuantumNoiseModel(name=f"Amplitude Damping (gamma={gamma})", amplitude_damping_gamma=gamma)

def dephasing_model(lambda_val: float = 0.01) -> QuantumNoiseModel:
    return QuantumNoiseModel(name=f"Dephasing (lambda={lambda_val})", dephasing_lambda=lambda_val)

def measurement_error_model(p_meas: float = 0.05) -> QuantumNoiseModel:
    return QuantumNoiseModel(name=f"Measurement Error (p={p_meas})", measurement_error_p=p_meas)

def combined_nisq_noise_model() -> QuantumNoiseModel:
    return QuantumNoiseModel(
        name="Combined NISQ Noise",
        depolarizing_p=0.005,
        amplitude_damping_gamma=0.01,
        dephasing_lambda=0.01,
        measurement_error_p=0.05
    )
