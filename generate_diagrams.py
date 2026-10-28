import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.environment.gridworld import GridWorld
from src.quantum.qiskit_hardware_model import QiskitHardwareSimulator
from src.quantum.quantum_world_model import QuantumWorldModel

def generate_diagrams():
    fig_dir = "artifacts/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Architecture Diagram PNG
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.9, "HYBRID QUANTUM-CLASSICAL WORLD MODEL ARCHITECTURE", fontsize=14, fontweight='bold', ha='center')
    
    # Blocks
    plt.gca().add_patch(plt.Rectangle((0.05, 0.55), 0.38, 0.25, fill=True, color='skyblue', ec='black', lw=2))
    plt.text(0.24, 0.67, "CLASSICAL COMPONENT\n(GridWorld & Tabular/MCTS Q-Prior)", fontsize=10, fontweight='bold', ha='center')

    plt.gca().add_patch(plt.Rectangle((0.57, 0.55), 0.38, 0.25, fill=True, color='plum', ec='black', lw=2))
    plt.text(0.76, 0.67, "QUANTUM COMPONENT\n(VQC + Grover Oracle + Diffusion)", fontsize=10, fontweight='bold', ha='center')

    plt.gca().add_patch(plt.Rectangle((0.30, 0.15), 0.40, 0.25, fill=True, color='lightgreen', ec='black', lw=2))
    plt.text(0.50, 0.27, "POLICY COMBINATION INTERFACE\nQ(s,a) = 0.8 * P_quant(tau) + 0.2 * Q_tab", fontsize=10, fontweight='bold', ha='center')

    # Arrows
    plt.annotate('', xy=(0.57, 0.67), xytext=(0.43, 0.67), arrowprops=dict(arrowstyle="->", lw=2))
    plt.annotate('', xy=(0.50, 0.40), xytext=(0.76, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    plt.annotate('', xy=(0.24, 0.55), xytext=(0.35, 0.40), arrowprops=dict(arrowstyle="<-", lw=2))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hybrid_architecture_diagram.png"), dpi=300)
    plt.close()

    # 2. Circuit Diagram PNG using Qiskit
    env = GridWorld(6, 6)
    sim = QiskitHardwareSimulator(env=env, horizon=3)
    qwm = QuantumWorldModel(env=env, horizon=3)
    qc = sim.build_qiskit_circuit(qwm, (0, 0))

    fig = qc.draw(output='mpl')
    fig.savefig(os.path.join(fig_dir, "circuit_diagram.png"), dpi=300)
    plt.close()

    print("Generated architecture and circuit diagram PNG files in artifacts/figures/")

if __name__ == "__main__":
    generate_diagrams()
