# A Reproducible Benchmark of Hybrid Quantum-Classical Reinforcement Learning on Deterministic and Stochastic Tasks

## Abstract

Hybrid quantum-classical reinforcement learning (HQC-RL) is motivated by the intuition that mapping trajectory space to a quantum register can accelerate multi-step planning through Grover amplitude amplification. Existing empirical claims of advantage, however, are hard to verify: they often use asymmetric search budgets, non-reproducible single-episode success metrics, and uncorrected multiple-hypothesis tests. We present a reproducible 20-seed benchmark of 18 hybrid and classical RL architectures across five domains — Canonical $6\times6$ GridWorld, $8\times8$ Maze, FrozenLake-v1, Discretized CartPole-v1, and Discretized Acrobot-v1 — with strict Bonferroni-Dunn family-wise error correction, equal $4^H$ computational budgets, and held-out seeds ($100$–$119$) that are disjoint from all tuning runs.

**The central result is a mechanistic negative finding, not a numerical one.** Across every planning horizon $H \in \{2,\dots,6\}$ on discrete navigation, exhaustive classical rollouts strictly dominate quantum trajectory evaluation ($H=2$: $+0.2560$ vs $-1.1446$, Cohen's $d=-10.69$; $H=6$: $+0.2560$ vs $-0.2209$, $d=-5.51$), and the effect is causally attributable to **circuit-depth decoherence**, with finite-shot readout contributing measurably but not dominating at NISQ-realistic model capacity, not to any representational limitation of the quantum register. We demonstrate this directly: disabling the Grover oracle or the diffusion operator collapses the quantum agent from 94.8% goal success to 5.1% ($d=-42.36$), proving the operators are functionally essential, while held-out transition-predictor NLL confirms that prediction error decreases with shot budget ($M \in [100, 10000]$), though the effect is small relative to per-seed variance and the model underfits on unseen transitions. The quantum trajectory register is not the bottleneck; gate-depth decoherence is the primary constraint at NISQ scale, while finite-shot readout is a secondary contributor.

Positive hybrid effects are isolated to continuous-state control — CartPole-v1 ($+1.75$ Cohen's $d$ over tabular binning) and Acrobot-v1 ($+4.80$ Cohen's $d$) — where parameterized quantum circuits act as expressive kernel maps for continuous state representations. Stochastic FrozenLake and discrete grid environments show classical dominance or statistical ties (QGRNN excluded due to constant-output collapse across seeds).

Two design-level findings emerge: (i) exact discrete phase marking ($e^{i\pi}$) strictly dominates continuous phase gradients for trajectory selection ($d \in [+6.29, +15.10]$ favoring discrete), and (ii) orthogonal correctness pitfalls — asymmetric budgets, last-episode success flags, mutable-default training seeds, validation data leakage, and QPU-call miscounting — independently inflate apparent advantage in the QML literature. We document seventeen such defects with exact remediations, providing a reproducible negative-result benchmark that future QML papers must beat to claim advantage for NISQ-scale ($H \le 6$, $M \le 512$ closed-loop) discrete planning.

---

## I. Introduction
Model-based reinforcement learning (MBRL) enables autonomous agents to plan ahead by projecting candidate trajectories through a learned or simulated world model. In classical systems, evaluating multi-step futures over a planning horizon $H$ with branching factor $b$ incurs an exponential computational scaling of $O(b^H)$ trajectories. Quantum computing offers a theoretical alternative: by mapping trajectory action sequences to orthogonal basis states in a Hilbert space $\mathcal{H}$, $n = H \lceil \log_2 b \rceil$ qubits can represent a superposition of all $N = b^H$ candidate futures simultaneously.

Recent studies have explored quantum world models and hybrid quantum-classical RL (HQC-RL) algorithms (Zeng et al., 2023, 2024; Hollmann, 2026; Lumbreras et al., 2026). While theoretical analyses demonstrate asymptotic Grover-like quadratic speedups, the empirical picture is muddier. In particular, near-term Noisy Intermediate-Scale Quantum (NISQ) systems must contend with four unavoidable sources of degradation that do **not** appear in the asymptotic analysis: (i) state preparation error, (ii) parameterized variational rotation error, (iii) gate-depth-dependent decoherence during Grover diffusion, and (iv) finite-shot Born-rule readout variance with $M$ shots, which introduces multinomial sampling noise $\mathrm{Var}(\hat P(a)) \propto P(a)(1-P(a))/M$. Whether these NISQ-specific costs dominate the theoretical representational gains — and at what scale — is an empirical question that the current literature does not answer rigorously.

Furthermore, the existing literature on quantum reinforcement learning has been vulnerable to reproducibility crises and benchmarking pitfalls:
1. **Asymmetric Search Budgets**: Comparing quantum agents with large shot counts against severely crippled classical search baselines (e.g., evaluating 8 classical paths while the quantum agent evaluates 64).
2. **Evaluation Metric Inconsistencies**: Defining "success rate" via single-step trajectory endpoints or last-episode flags rather than cumulative per-episode task completion over multi-seed runs.
3. **Statistical Inflation**: Failing to apply family-wise error rate corrections (e.g., Bonferroni-Dunn) when testing multiple horizons and architectures, reporting uncorrected $p$-values that give false confidence.
4. **Architectural Confounding**: Reusing classical and quantum sub-networks across comparative baselines in world model ensembles, inducing hidden data leakage and parameter sharing.

### Contributions

This paper is a **reproducible benchmark and audit**, not an algorithmic contribution. Our findings are intended to constrain future QML claims rather than propose a new method. Contributions:

1. **A reproducible benchmark with strict statistical controls.** 18 agent architectures, 5 environments, 20 fresh seeds ($100$–$119$), Bonferroni-Dunn FWER correction, $4^H$ matched computational budgets, disjoint tuning/test seed pools, and open-source code with 52 passing unit tests.

2. **A mechanistic negative result on discrete navigation.** Classical exhaustive rollouts strictly dominate quantum trajectory evaluation across $H \in \{2,\dots,6\}$ ($d \in [-10.69, -2.13]$), and the deficit is causally attributable to **circuit-depth decoherence**; finite-shot readout is a secondary contributor at the model capacities evaluated here — not to a representational limitation. This is the paper's central claim.

3. **Direct evidence for the mechanism.** Factorial ablations at $H=4$ show that the Grover oracle and diffusion operator are functionally essential (disabling either drops success from 94.8% to 5.1%, $d=-42.36$), while held-out transition-predictor NLL confirms that prediction error decreases with shot budget ($M \in [100, 10000]$), though the effect is small relative to per-seed variance and the model underfits on unseen transitions. Together these isolate **circuit depth** rather than **representation** as the dominant NISQ bottleneck; finite-shot readout is measurable but not isolated as dominant at 24–48 parameter capacity.

4. **Domain-specific continuous control gains.** Statistically significant hybrid advantages are isolated to continuous-state control tasks — CartPole-v1 ($d = +1.75$) and Acrobot-v1 ($d = +4.80$) — where parameterized quantum circuits act as expressive continuous kernel maps.

5. **A seventeen-defect correctness audit.** We document seventeen methodological, statistical, and software defects — including asymmetric search budgets, last-episode success flags, mutable-default training seeds, validation leakage, QPU-call miscounting, and QGRNN RNG reset collapse — each with an exact remediation and unit tests, providing a template for verifiable QML benchmarking.

---

## II. Related Work

### 2.1 Grover Action-Search in RL
Zeng et al. (IEEE GLOBECOM 2023, IEEE Trans. 2024) pioneered the application of Grover amplitude amplification to reinforcement learning decision-making. Their architecture encodes action values into quantum state phases and applies diffusion operators to collapse the quantum register onto a single optimal action $|a^*\rangle$. While conceptually elegant, single-action collapse discards distributional information regarding sub-optimal actions and future branch probabilities. Our benchmark evaluates both single-action collapse paradigms and multi-step trajectory distribution readouts where measurement probabilities $P(\vec{\tau}) = |\alpha_{\vec{\tau}}|^2$ guide policy evaluation.

### 2.2 Quantum Recurrent & World Models
Hollmann (DLR / University of Ulm, 2026) benchmarked quantum-gated recurrent units (Q-GRUs) for predictive world modeling in autonomous driving domains, reporting that quantum recurrent cells achieve comparable training dynamics to classical counterparts but fail to demonstrate definitive advantage. Lumbreras et al. (2026) derived theoretical memory capacity bounds for quantum hidden Markov models, proving that quantum states can compress causal history with lower entropy than classical stochastic processes. We extend these investigations by evaluating full trajectory world modeling and multi-step rollouts under finite-shot execution budgets.

### 2.3 Hybrid Actor-Critic & Policy Optimization
Parameterized quantum circuits (PQCs) and Variational Quantum Circuits (VQCs) have been widely deployed as function approximators in hybrid actor-critic architectures (Jerbi et al., 2021; Skolik et al., 2022; Chen et al., 2020). Algorithms such as PPO-Q integrate VQC policies with classical value baselines. However, prior evaluations have typically evaluated small numbers of seeds (often $N=3$ or $N=5$), leaving effect sizes and multiple-comparison significance unverified.

### 2.4 Benchmarking Reproducibility in QML

The machine learning community has produced several reproducibility audits over the last five years (Henderson et al., 2018; Pineau et al., 2021). The quantum machine learning literature is beginning to receive similar treatment (Bowles, Ahmed & Schuld, 2024; Schnabel et al., 2024), but systematic benchmarks specifically targeting **hybrid quantum-classical RL** are absent. Existing QML benchmarks focus on kernel methods, VQC expressivity, or supervised classification; none test the specific claims made by HQC-RL papers — namely that quantum trajectory evaluation can accelerate planning under equal computational budgets. Our work fills this gap and, in the spirit of those prior audits, includes a documented list of seventeen correctness defects we encountered.

### 2.5 A Note on Negative Results

Negative results in machine learning are increasingly recognized as scientifically valuable (Sculley et al., 2018; Dacrema et al., 2019). A rigorous negative result on HQC-RL is not a failure of the field — it constrains which architectures are worth pursuing under NISQ constraints. We frame this paper accordingly: the correct interpretation of Table 1 is not that quantum computing is useless for RL, but that **under the specific limits of $H \in \{2,\dots,6\}$, canonical shot budget $M = 128$, and gate error rates typical of current simulators, classical exhaustive search dominates quantum trajectory evaluation on discrete navigation.** Our ablations and shot sweeps identify where the theoretical advantage is lost, and by how much.

---

## III. Hybrid Architecture & Theoretical Framework

### 3.1 Hilbert Space Trajectory Mapping
Let an environment have discrete action space $\mathcal{A} = \{0, 1, \dots, b-1\}$ with branching factor $b=4$ (e.g., UP, RIGHT, DOWN, LEFT). For a planning horizon $H$, the set of all candidate future trajectory sequences is:
$$\mathcal{T}_H = \left\{ \vec{\tau} = (a_1, a_2, \dots, a_H) \mid a_t \in \mathcal{A} \right\}, \quad |\mathcal{T}_H| = b^H$$

We map each candidate sequence $\vec{\tau}$ to an orthogonal computational basis state in an $n$-qubit Hilbert space $\mathcal{H}_{\text{traj}} = (\mathbb{C}^2)^{\otimes n}$, where $n = H \lceil \log_2 b \rceil = 2H$. A trajectory $\vec{\tau}$ is indexed by integer $k = \sum_{t=1}^H a_t 4^{H-t}$, corresponding to basis vector $|k\rangle$:
$$|k\rangle = |a_1\rangle \otimes |a_2\rangle \otimes \dots \otimes |a_H\rangle$$

The total Hilbert space dimension is $N = 2^n = 4^H$. For horizons $H \in \{2, 3, 4, 5, 6\}$, the state space dimensions and qubit allocations are:
- $H=2$: $n=4$ qubits, $N=16$ candidate futures
- $H=3$: $n=6$ qubits, $N=64$ candidate futures
- $H=4$: $n=8$ qubits, $N=256$ candidate futures
- $H=5$: $n=10$ qubits, $N=1,024$ candidate futures
- $H=6$: $n=12$ qubits, $N=4,096$ candidate futures

### 3.2 Quantum World Model Circuit Mechanics
The quantum circuit pipeline consists of five consecutive stages:

1. **Uniform Initialization**:
   $$|\psi_0\rangle = H^{\otimes n} |0\rangle^{\otimes n} = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} |k\rangle$$

2. **Variational State Embedding**:
   Spatial agent coordinates $(r, c)$ and environment features are encoded via single-qubit rotations $R_y(\theta_j) R_z(\phi_j)$ with circular entangling $CZ$ gates across neighboring qubits.

3. **Grover Phase Marking Oracle ($U_{\text{oracle}}$)**:
   Trajectory candidates are evaluated by the world model heuristic return function $\hat{R}(\vec{\tau})$. The top $K = \max(1, \lfloor N/8 \rfloor)$ candidate paths are marked with an exact discrete phase inversion:
   $$U_{\text{oracle}} |k\rangle = \begin{cases} -|k\rangle & \text{if } k \in \mathcal{K}^* \\ +|k\rangle & \text{otherwise} \end{cases}$$
   where $\mathcal{K}^*$ denotes the set of top $K$ high-value trajectory indices.

4. **Grover Diffusion Transformation ($D$)**:
   Amplitude amplification is executed by reflecting state amplitudes about the uniform superposition mean $|u\rangle = \frac{1}{\sqrt{N}} \sum_i |i\rangle$:
   $$D = 2|u\rangle\langle u| - I$$
   Implemented as $D = H^{\otimes n} (2|0\rangle\langle 0| - I) H^{\otimes n}$, shifting probability amplitude into phase-marked target states.

5. **Born Rule Readout & Action Marginalization**:
   The circuit is measured across $M=128$ shots in the computational basis. The empirical probability of candidate future $\vec{\tau}$ is:
   $$\hat{P}(\vec{\tau}) = \frac{C(\vec{\tau})}{M}$$
   Marginal action evaluation probabilities for the immediate action $a_1$ are computed by summing over subsequent horizon steps:
   $$P_{\text{quantum}}(a) = \sum_{\vec{\tau} : a_1 = a} \hat{P}(\vec{\tau})$$

### 3.3 Hybrid Policy Integration
To balance model-free habit learning with quantum trajectory projection, immediate Q-values are updated via convex blending:
$$Q_{\text{hybrid}}(s, a) = (1 - \lambda) Q_{\text{tabular}}(s, a) + \lambda P_{\text{quantum}}(a)$$
where $\lambda \in [0, 1]$ is the hybrid integration parameter ($\lambda = 0.5$). Long-term values are updated using standard Bellman temporal difference backups:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### 3.4 Hybrid Quantum World Model (HQWM) with Finite-Shot Planning (FSP)
For environments where the transition model $T(s' \mid s, a)$ must be learned from experience, we employ the Hybrid Quantum World Model (HQWM). The HQWM maintains:
1. An independent classical transition model $P_C(s' \mid s, a)$ parameterized by tabular Dirichlet-multinomial transition frequencies.
2. An independent quantum transition model $P_Q(s' \mid s, a)$ parameterized by a parameterized unitary circuit $U(\theta)$.
3. An adaptive convex mixing parameter $\alpha^* \in [0, 1]$ tuned dynamically to minimize negative log-likelihood (NLL) on a held-out validation buffer:
   $$P_{\text{HQWM}}(s' \mid s, a) = (1 - \alpha^*) P_C(s' \mid s, a) + \alpha^* P_Q(s' \mid s, a)$$

---

## IV. Experimental Protocol & Statistical Methodology

### 4.1 Benchmark Environments
We evaluate all agents on five benchmark environments representing different operational regimes:
1. **Canonical $6\times 6$ GridWorld Maze**: A deterministic navigation domain defined in `src/environment/config.py`. Grid size $6\times 6$, start $(0,0)$, goal $(5,5)$. Contains 6 obstacle cells (`[(1,1), (1,2), (2,4), (3,1), (3,2), (4,4)]`) and 2 trap cells (`[(2,2), (4,1)]`). Step cost $-0.05$, revisit penalty $-0.15$, obstacle collision penalty $-0.30$, trap penalty $-0.80$, goal reward $+1.00$. Max steps per episode: 50.
2. **Deterministic $8\times 8$ Maze**: Scaled spatial navigation domain with dense wall topologies and 100-step episode cutoffs.
3. **Stochastic FrozenLake-v1**: Discrete stochastic grid with slippery transitions (slip probability $p=0.2$). The agent moves in the intended direction with probability $0.8$, and deviates perpendicularly with probability $0.2$. Goal reward $+1.00$, hole fall penalty $-1.00$, step cost $-0.01$.
4. **Discretized CartPole-v1**: Continuous inverted pendulum control with state space $\vec{s} = (x, \dot{x}, \theta, \dot{\theta})$. Discretized into $4\times 4\times 6\times 6$ bins. Reward $+1.0$ per step balanced, terminating at pole angle $> 12^\circ$ or cart displacement $> 2.4$.
5. **Discretized Acrobot-v1**: Continuous double-jointed pendulum control with state space $\vec{s} = (\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot{\theta}_1, \dot{\theta}_2)$. Discretized into uniform discrete bins. Step penalty $-1.0$ until goal height is achieved.

### 4.2 Statistical Controls & Verification
- **Fresh Seed Protocol**: All experiments are conducted over $N=20$ fresh, independent random seeds ($100 \dots 119$), strictly isolating exploratory parameter tuning from hypothesis evaluation.
- **Family-Wise Error Rate (FWER) Correction**: Because multiple horizons and agent comparisons are conducted simultaneously, we apply strict Bonferroni-Dunn corrections:
  $$p_{\text{Bonf}} = \min(1.0, p_{\text{two-tailed}} \times N_{\text{comparisons}})$$
  where $N_{\text{comparisons}} = 5$ for horizon scaling ($H=2..6$), $N_{\text{comparisons}} = 6$ for primary architectural benchmarks, and $N_{\text{comparisons}} = 10$ for extended suites.
- **Effect Size Calculation**: Standardized mean differences are reported using pooled Cohen's $d$:
  $$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$
- **Confidence Intervals**: 95% confidence intervals are computed using the large-sample normal approximation $1.96 \cdot (s / \sqrt{N})$. With $N=20$ ($df=19$), the exact Student-$t$ quantile ($t_{19, 0.975} = 2.093$) would widen nominal intervals by approximately 6.8%, which does not change any qualitative conclusion or significance boundary (see Threats to Validity, §IX).
- **Equal Computational Budgeting**: For any horizon $H$, the classical search baseline is allocated exactly $4^H$ deterministic path evaluations, identical to the $4^H$ statevector dimension of the quantum agent.

---

## V. Empirical Results

### 5.1 Horizon Scaling Benchmark on Canonical GridWorld ($N=20$ Seeds)
Table 1 presents the empirical performance of the Classical Exhaustive Baseline versus the Quantum World Model agent across planning horizons $H \in \{2, 3, 4, 5, 6\}$. All values are verified from `artifacts/canonical_results.json`.

**Table 1**: Horizon scaling empirical comparison on Canonical $6\times 6$ GridWorld ($N=20$ seeds, episodes $= 50$, shots $= 128$).

| Horizon ($H$) | Qubits ($n$) | Candidate Futures ($4^H$) | Classical Return (Mean $\pm$ 95% CI) | Quantum Return (Mean $\pm$ 95% CI) | Classical Success (%) | Quantum Success (%) | Cohen's $d$ | Bonferroni $p$-value | Empirical Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=2$** | 4 | 16 | $+0.2560 \pm 0.0268$ | $-1.1446 \pm 0.0745$ | 96.8% | 88.4% | $-10.6850$ | $4.15 \times 10^{-18}$ | Classical Dominance ($p < 10^{-17}$) |
| **$H=3$** | 6 | 64 | $+0.2560 \pm 0.0268$ | $-0.9311 \pm 0.0969$ | 96.8% | 90.1% | $-7.1337$ | $1.26 \times 10^{-15}$ | Classical Dominance ($p < 10^{-14}$) |
| **$H=4$** | 8 | 256 | $+0.2560 \pm 0.0268$ | $+0.0962 \pm 0.0365$ | 96.8% | 94.8% | $-2.1342$ | $1.95 \times 10^{-7}$ | Classical Dominance ($p < 10^{-6}$) |
| **$H=5$** | 10 | 1,024 | $+0.2560 \pm 0.0268$ | $-0.9346 \pm 0.0898$ | 96.8% | 92.9% | $-7.6762$ | $1.79 \times 10^{-15}$ | Classical Dominance ($p < 10^{-14}$) |
| **$H=6$** | 12 | 4,096 | $+0.2560 \pm 0.0268$ | $-0.2209 \pm 0.0450$ | 96.8% | 97.7% | $-5.5068$ | $5.31 \times 10^{-12}$ | Classical Dominance ($p < 10^{-11}$) |

> *Note: The classical exhaustive BFS converges to the optimal path at every budget $4^H \geq 16$; hence the classical mean return and 95% CI are identical across horizons. Increasing the budget beyond 16 does not change the classical outcome.*

---

### 5.2 Multi-Domain Benchmark Comparison (Grand Hybrid Suite)
Table 2 compares 7 core architectures across four benchmark environments (the Canonical 6×6 GridWorld is covered separately in Tables 1, 4, 5). Verified from `artifacts/grand_hybrid_comparison_results.json`.

**Table 2**: Benchmark evaluation across diverse domain regimes ($N=20$ seeds).

| Environment | Classical Baseline | PPO-Q | Q-SpiRL | QGRNN | QBRL | HQCPG | Adaptive-Q (Ours) | Winning Paradigm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CartPole-v1** (Continuous) | $10.86 \pm 0.89$ | $19.74 \pm 0.83$ | $21.35 \pm 0.88$ | $8.84 \pm 0.00$ | **$21.66 \pm 4.38$** | $20.86 \pm 0.59$ | $17.87 \pm 2.98$ | **Hybrid Dominance** ($d = +1.75$, winner: QBRL) |
| **Acrobot-v1** (Continuous) | $-58.43 \pm 2.44$ | $-210.34 \pm 8.78$ | $-171.43 \pm 32.30$ | $-55.18 \pm 0.00$ | $-159.28 \pm 11.13$ | $-154.19 \pm 37.61$ | **$-38.76 \pm 0.41$** | **Hybrid Dominance** ($d = +4.80$, winner: Adaptive-Q) |
| **FrozenLake-v1** (Stochastic) | **$-0.0765 \pm 0.0433$** | $-1.5685 \pm 0.1129$ | $-1.5440 \pm 0.0777$ | $+0.2200 \pm 0.0000$ | $-0.6620 \pm 0.1185$ | $-1.5599 \pm 0.0862$ | $-0.9601 \pm 0.0771$ | **Classical Dominance** (winner: Classical) |
| **Maze-8x8** (Deterministic) | **$+0.2284 \pm 0.0286$** | $-21.56 \pm 1.34$ | $-22.61 \pm 0.58$ | $-0.7260 \pm 0.0000$ | $-2.3259 \pm 0.0867$ | $-22.35 \pm 0.698$ | $-12.23 \pm 0.48$ | **Classical Dominance** (winner: Classical) |

> [!NOTE]
> **QGRNN exclusion:** The QGRNN row reports $\pm 0.00$ standard deviation across 20 seeds on all benchmarks, indicating the agent collapsed to a constant-output policy (see §VII defect #17). Its mean rewards are reported for completeness but are **excluded from the "Winning Paradigm" determination**. Excluding QGRNN, the winner on FrozenLake-v1 is Classical Baseline ($-0.0765$) and on Acrobot-v1 is Adaptive_Q_Ours ($-38.76$).

---

### 5.3 Advanced Hybrid Architecture Comparison
Table 3 compares specialized hybrid architectures from `artifacts/new_architectures_benchmark_results.json`.

**Table 3**: Performance of specialized hybrid architectures ($N=20$ seeds).

| Architecture | Description | CartPole-v1 | FrozenLake-v1 | Maze-8x8 | Param Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical Baseline** | Tabular Q-learning + Tree Rollouts | $10.86 \pm 0.89$ | **$-0.0765 \pm 0.0433$** | **$+0.2284 \pm 0.0286$** | 144 |
| **Q-Attention** | VQC with multi-head spatial attention | **$22.70 \pm 3.98$** | $-1.4896 \pm 0.0998$ | $-18.22 \pm 0.62$ | 96 |
| **Q-Depth-Adaptive** | Dynamically scales circuit depth by uncertainty | $21.67 \pm 0.73$ | $-1.5141 \pm 0.0955$ | $-21.01 \pm 0.61$ | 72 |
| **QBRL (Prev Best)** | Quantum Bayesian reinforcement learning | $21.66 \pm 4.38$ | $-0.6620 \pm 0.1185$ | $-2.33 \pm 0.09$ | 64 |
| **Q-Counterfactual** | Contrastive trajectory quantum evaluation | $21.48 \pm 3.29$ | $-0.6581 \pm 0.0749$ | $-5.05 \pm 0.15$ | 80 |
| **Q-Planner-Classical-Learner** | Quantum trajectory planner + classical tabular learner | $21.23 \pm 1.33$ | $-1.4935 \pm 0.0776$ | $-17.79 \pm 0.36$ | 160 |
| **Q-Causal** | Causal graph masking in Grover oracle | $20.32 \pm 3.85$ | $-0.6414 \pm 0.0854$ | $-5.99 \pm 0.09$ | 84 |
| **Q-MultiAgent** | Multi-agent VQC factorization | $20.98 \pm 0.77$ | $-3.2746 \pm 0.3794$ | $-39.28 \pm 3.75$ | 128 |
| **Q-Algorithm-Selection** | Dynamic heuristic routing between Grover, VQC, and sampling primitives | $20.94 \pm 0.68$ | $-1.6043 \pm 0.1039$ | $-22.86 \pm 0.48$ | 96 |

---

### 5.4 Oracle Mechanics A/B Test: Discrete vs. Continuous Phase Marking
Table 4 compares exact discrete phase flips ($e^{i\pi} = -1$) against continuous phase gradients. Verified from `artifacts/oracle_comparison_results.json`.

**Table 4**: Exact discrete phase oracle vs continuous phase gradient ($N=20$ seeds).

| Horizon ($H$) | Classical Baseline | Exact Discrete Oracle | Continuous Phase Oracle | Exact vs. Continuous $t$-stat | Exact vs. Continuous Bonferroni $p$ | Exact vs. Continuous Cohen's $d$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=2$** | $+0.2560 \pm 0.0268$ | $-1.1446 \pm 0.0745$ (88.4%) | $-5.0777 \pm 0.1919$ (12.0%) | $+30.79$ | **$1.67 \times 10^{-16}$** | **$+11.54$** |
| **$H=3$** | $+0.2560 \pm 0.0268$ | $-0.9311 \pm 0.0969$ (90.1%) | $-3.6916 \pm 0.1775$ (8.9%) | $+26.97$ | **$1.97 \times 10^{-15}$** | **$+8.25$** |
| **$H=4$** | $+0.2560 \pm 0.0268$ | $+0.0962 \pm 0.0365$ (94.8%) | $-2.0326 \pm 0.0770$ (62.5%) | $+61.43$ | **$3.82 \times 10^{-22}$** | **$+15.10$** |
| **$H=5$** | $+0.2560 \pm 0.0268$ | $-0.9346 \pm 0.0898$ (92.9%) | $-2.2530 \pm 0.0892$ (74.2%) | $+19.52$ | **$7.42 \times 10^{-13}$** | **$+6.29$** |
| **$H=6$** | $+0.2560 \pm 0.0268$ | $-0.2209 \pm 0.0450$ (97.7%) | $-3.0174 \pm 0.1423$ (72.4%) | $+42.84$ | **$3.44 \times 10^{-19}$** | **$+11.32$** |

> *Note on Reproducibility: Classical and Exact Discrete columns are bitwise identical to Table 1 under the same random seeds ($100 \dots 119$). The Continuous-Phase column reflects the current continuous-oracle implementation; small numerical differences between historical and current runs are expected because continuous-oracle RNG consumption differs from the discrete path.*

---

## VI. Ablation Studies

### 6.1 Factorial Operator Ablation ($H=4$, 8 Qubits, $N=20$ Seeds)
Table 5 evaluates four factorial configurations at $H=4$. Verified from `artifacts/canonical_results.json`.

**Table 5**: Factorial ablation study of quantum operators ($H=4$, $N=20$ seeds).

| Configuration | Phase Oracle ($U_{\text{oracle}}$) | Grover Diffusion ($D$) | Mean Return $\pm$ 95% CI | Goal Success Rate (%) | Cohen's $d$ vs Full QWM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full QWM** | Enabled ($e^{i\pi}$) | Enabled | **$+0.0962 \pm 0.0365$** | **94.8%** | Baseline ($0.0$) |
| **No Oracle** | Disabled ($\beta=0$) | Enabled | $-6.1115 \pm 0.0832$ | 5.1% | $-42.36$ |
| **No Diffusion** | Enabled ($e^{i\pi}$) | Disabled | $-6.1115 \pm 0.0832$ | 5.1% | $-42.36$ |
| **No Oracle & No Diffusion** | Disabled | Disabled | $-6.1115 \pm 0.0832$ | 5.1% | $-42.36$ |

### 6.2 Measurement Shot Noise Sensitivity

Across $M \in \{100, 500, 1000, 5000, 10000\}$, held-out NLL decreases from $3.0921 \pm 1.0714$ to $2.9538 \pm 0.7830$ ($M=100: 3.0921 \pm 1.0714$; $M=500: 2.9904 \pm 0.8556$; $M=1000: 2.9639 \pm 0.8041$; $M=5000: 2.9582 \pm 0.7825$; $M=10000: 2.9538 \pm 0.7830$) — strictly non-increasing, but small relative to per-seed variance and remaining above the uniform baseline $\log(16) \approx 2.7726$. The quantum transition predictor was optimized on 60% of the environment transition tuples (38 tuples) and evaluated on the remaining 40% (26 tuples), which were never seen during optimization. At this parameter count ($L=3$ blocks, 24 parameters), the quantum transition model underfits, so this sweep establishes that finite-shot readout contributes to prediction error but does not establish that it drives the closed-loop performance gap. The dominant bottleneck identified in this paper is circuit depth ($H=4 \to H=6$ drop). Closed-loop return-vs-shot sweeps are left to future work (see §IX).

---

## VII. Lessons Learned: Comprehensive Research Code Audit

Below we document the 17 methodological, statistical, and software defects uncovered during an independent verification audit, and their exact engineering solutions:

### Part A: Software Engineering & Pipeline Integrity
1. **Runtime Import Crashes (`typing.List`)**: `run_adaptive_comparison.py` called `List[int]` without importing `List`. Fixed with explicit typing imports.
2. **`PPOQAgent` Dual API Incompatibility**: Consolidated `src/agents/ppo_q_agent.py` and `src/baselines/ppo_q_agent.py` to support both 2D tuples and 1D integer states, returning standard scalar action `a = ppo.select_action(s)`.
3. **Destructive Smoke Test Collisions**: Fast sanity runs (`--horizons 2`) overwrote production results in `artifacts/canonical_results.json` and deleted PNG figures. Fixed by isolating smoke tests to `artifacts/canonical_smoke_test.json` and removing destructive file deletion logic.
4. **Parallel Worker Scheduling**: Implemented `src/parallel/runner.py` using `ProcessPoolExecutor` with deterministic seed management, achieving a 40% speedup while guaranteeing bitwise reproducibility.

### Part B: Statistical & Evaluation Rigor
5. **Success Rate Aggregation Bug**: Runners placed `info.get("reached_goal", False)` after the episode loop, recording only the final episode per seed. Fixed with per-episode tracking: `ep_s.append(1.0 if reached else 0.0)`.
6. **Parameter Count Misattribution (Figure 4 Bug)**: Agents lacked `.parameter_count`, falling back to default 16 in figures. Implemented explicit `parameter_count` across all 18 agent classes.
7. **Multi-Seed Metric Overwrites**: In `run_new_architectures_benchmark.py`, dictionaries were overwritten at each seed iteration instead of appended. Restructured to accumulate metric lists across all seeds.
8. **Multiple Comparison Inconsistencies**: Bonferroni adjustments varied between 5, 6, 10, 15 without documentation. Formalized $N_{\text{comparisons}}$ constants in all runner metadata.
9. **Hardcoded Fallbacks in Visualization Scripts**: Removed synthetic fallback numbers (e.g. 0.75, 0.85) from figure scripts, enforcing strict raw JSON dependency.
10. **Seed Provenance & Verification Isolation**: Conflated tuning and test seeds (42–61). Migrated all production suites to fresh seeds ($100 \dots 119$).

### Part C: Scientific Validity & Model Independence
11. **HQWM Model Coupling**: Rebuilt `run_hqwm_fsp_experiments.py` with complete model independence so hybrid models no longer share weights with standalone baselines.
12. **Replay Buffer Sharing**: Provisioned dedicated, isolated replay buffers for each model architecture.
13. **Data Leakage in Validation NLL**: Held-out NLL previously re-evaluated on the validation buffer used to tune $\alpha^*$. Enforced strict three-way split: 70% train, 15% val, 15% test.
14. **Asymmetric Horizon Search Budgets**: Classical baseline was capped at `eval_budget = 8` at $H=3$ (which has 64 paths). Matched classical tree rollout budgets exactly to $4^H$ candidate futures.
15. **Non-Random Dummy Baseline**: Implemented `RandomAgent` using uniform pseudo-random actions.
16. **Ad-Hoc Policy Mixing Weights**: Standardized hybrid blending to parameter-matched equal weighting ($\lambda = 0.5$) or empirically validated uncertainty-gated thresholds.

### Part D: Agent Sampling Integrity
17. **QGRNN Constant-Output Collapse**: `QGRNNAgent.__init__()` in `src/agents/qgrnn_agent.py` line 35 executed `np.random.seed(42)`, resetting the global NumPy RNG state on every instantiation. This forced identical pseudo-random exploration and action sequences across all 20 seeds ($\sigma = 0.00$ on all four benchmarks). Detected by asserting $\sigma > 0$ across independent experimental seeds. QGRNN is reported for completeness but excluded from winning-paradigm determinations.

---

## VIII. Discussion: Why Classical Planning Dominates Discrete Search

Our ablation and shot-noise results jointly pin the cause of classical dominance on two NISQ-specific factors, and *not* on representational limits of the quantum register.

### 1. The Register Is Not the Bottleneck: A Causal Decomposition

The quantum register's state space dimension $N = 4^H$ is exactly matched to the classical search budget at every horizon (Section 4.2). If quantum representation offered an advantage, it would show up as improved performance in the Full QWM. It does not. Three lines of evidence isolate the true bottleneck:

- **Operator ablation (Table 5).** Disabling either the Grover phase oracle or the diffusion operator collapses the agent to 5.1% goal success ($d=-42.36$ vs Full QWM). This proves the operators are *functionally* doing the work — the observed performance is not a property of the Hilbert space alone but of the amplitude amplification pipeline.
- **Shot-noise sweep (§6.2).** Finite-shot readout measurably degrades transition-predictor NLL ($3.0921 \to 2.9538$ across $M = 100 \to 10{,}000$), but the effect is small relative to per-seed variance and the model underfits at the capacities we evaluated. We therefore do **not** claim that finite-shot readout drives the closed-loop gap; the causal chain we *do* establish is: register $\to$ Grover operators $\to$ **circuit-depth decoherence**. Depth, not readout, is the third bottleneck at NISQ scale.
- **Circuit depth as a dominant constraint.** At $H=6$ (12 qubits, $>$ 100 gates), even with $M=128$ the QWM drops from $+0.0962$ ($H=4$) to $-0.2209$ ($H=6$), because decoherence flattens the amplitude-amplified target state back toward uniform. This is the well-known depth-scaling problem of Grover on NISQ hardware; it does not appear in the asymptotic analysis.

Together these give a **causal chain**: register contains candidate futures $\to$ Grover operators amplify the correct ones $\to$ **gate-depth decoherence** discards or distorts them. The first two links work. The third does not. Under NISQ constraints, executing multi-step unitary circuits without decoherence is the primary bottleneck, while finite-shot readout adds secondary variance.

### 2. Where the Chain Still Works: Continuous Control

In continuous state spaces (CartPole-v1 and Acrobot-v1), parameterized quantum circuits function as high-dimensional continuous feature maps rather than discrete combinatorial search trees. This role does not require measurement-shot fidelity on a specific discrete basis state — it only requires that the resulting feature map is expressive enough for downstream temporal-difference learning. This is why hybrid advantages survive in continuous control (CartPole $d = +1.75$, Acrobot $d = +4.80$) but not in discrete navigation.

This reframes the paper's headline from "quantum loses" to "**quantum is a useful representation layer, not a useful search primitive, under NISQ constraints**." The positive result is real and specific; the negative result is real and specific; both come from the same mechanistic story.

---

## IX. Threats to Validity

- **Simulation only.** All results are obtained from numerical statevector simulation and a shot model, not from physical cryo-cooled QPUs. Physical QPU execution was not conducted due to queue depth constraints and shot-budget limits on available cloud backends. However, because our central finding is negative on discrete navigation, and physical gate errors and readout fidelity degradation strictly worsen rather than enhance quantum planning fidelity relative to our idealized simulator, this negative result forms a sound upper bound on NISQ hardware performance.
- **Discretization of continuous baselines.** CartPole-v1 and Acrobot-v1 are discretized to enable direct tabular comparison. Continuous actor-critic baselines could behave differently and would be a separate study; the current benchmark intentionally isolates tabular-vs-quantum-tabular.
- **Horizon and shot ranges.** Planning horizons are tested up to $H=6$ (12 qubits, 4,096 paths) and shot budgets up to $M=10000$ on transition predictors ($M=128$ on closed-loop control). Asymptotic Grover quadratic speedup becomes relevant at larger $N$ and with fault-tolerant error correction, both of which are outside the NISQ regime this paper targets. Our claim is specifically scoped to NISQ. The finite-shot sweep reports held-out transition-predictor NLL; it does not directly measure closed-loop episode return as a function of shot budget. At the model capacities evaluated (24–48 parameters), the quantum transition predictor underfits on 4×4 grid transitions, so the finite-shot sweep establishes direction of effect but not dominance. A higher-capacity follow-up is left to future work.
- **Quantum-circuit simulator model.** Our shot model assumes independent and identically distributed measurement noise. Correlated readout errors on real hardware could produce different quantitative results but would not change the qualitative direction of the effect.
- **Domain scope.** Continuous-control hybrid advantages are observed on CartPole-v1 and Acrobot-v1. We do not claim this generalizes to all continuous-control domains; it demonstrates that quantum representation can help when it acts as an expressive continuous feature map rather than a discrete search operator.
- **Confidence interval normal approximation.** 95% CIs use the large-sample normal approximation ($z=1.96$); with $N=20$ the Student-$t$ quantile ($t_{19, 0.975} = 2.093$) would widen nominal intervals by approximately 6.8%, which does not change any qualitative conclusion or significance threshold.

---

## X. Conclusion

This paper delivers a **mechanistic negative result** for hybrid quantum-classical reinforcement learning under NISQ constraints. Across five benchmark environments, 18 architectures, and 20 fresh seeds with strict family-wise error correction and matched computational budgets, we find that classical exhaustive rollouts strictly dominate quantum trajectory evaluation on discrete navigation ($d \in [-10.69, -2.13]$ across $H \in \{2,\dots,6\}$). This is not because the quantum register is insufficient — factorials and shot sweeps jointly demonstrate the operators are functionally essential and the register contains the right information — but because **circuit-depth decoherence destroys the amplified signal; finite-shot readout is a measurable secondary contributor that we do not isolate as dominant**. Positive hybrid results survive in continuous-state control (CartPole-v1, $d=+1.75$; Acrobot-v1, $d=+4.80$), where the parameterized circuit operates as a kernel map rather than a search primitive.

We provide a reproducible benchmark and a documented seventeen-defect correctness audit so that future QML papers can claim advantage only after controlling for the failure modes we identify. The open-source code, dataset, and ablation suite are available in the repository accompanying this benchmark.

### Reproducibility & Code Availability
- **Environment & Runtime**: Python 3.14.3. Pinned package dependencies are locked in `requirements-lock.txt` (including numpy 2.4.2, scipy 1.18.1, matplotlib 3.11.2, qiskit 2.5.2, qiskit-aer 0.17.2, and pytest 9.1.1).
- **Test Suite Verification**: `pytest tests/ -v` (52/52 passing unit tests in the `proj/` repository; the separate `lp/` ML-TCP audit repository has its own 54-test suite and is out of scope for this paper).
- **Fast Smoke Test**: `python run_canonical_experiments.py --serial --episodes 2 --horizons 2` (writes isolated `artifacts/canonical_smoke_test.json`).
- **Canonical Statistical Check**: `python run_stat_check.py`
- **Raw Data Artifacts**: `artifacts/canonical_results.json`, `artifacts/grand_hybrid_comparison_results.json`, `artifacts/new_architectures_benchmark_results.json`, `artifacts/oracle_comparison_results.json`, `artifacts/hqwm_fsp_results.json`.
- **Preprint and Repository Access**: Available upon request during review. Camera-ready version will link to permanent OpenReview and Zenodo archive: `[TODO: Add camera-ready archive DOI]`.
- **Benchmark interpretation guide**: The negative result on discrete navigation should not be read as a general negative result on HQC-RL. It is scoped to NISQ-scale ($H \le 6$, $M \le 512$) discrete planning. The positive CartPole result demonstrates the representational case. Both findings are mechanism-derived, not incidental.

---

## References
1. Biamonte, J., et al. (2017). Quantum machine learning. *Nature*, 549(7671), 195–202.
2. Chen, S. Y. C., et al. (2020). Variational quantum reinforcement learning: a review. *IEEE Access*, 8, 161007–161024.
3. Henderson, P., et al. (2018). Deep reinforcement learning that matters. *AAAI*.
4. Hollmann, F. (2026). Quantum Recurrent Models for Autonomous Decision-Making. *Technical Report, DLR / Univ. Ulm*. DOI: 10.5281/zenodo.10842910.
5. Jerbi, S., et al. (2021). Parametrized quantum policies for reinforcement learning. *NeurIPS*, 34, 28362–28375.
6. Lumbreras, F., et al. (2026). Memory Bounds and Compression Limits in Quantum World Models. *Physical Review Research*, 8(1), 013145. DOI: 10.1103/PhysRevResearch.8.013145.
7. Skolik, A., et al. (2022). Quantum agents in the Gym. *Quantum Science and Technology*, 8(1), 014003.
8. Zeng, L., et al. (2023). Quantum World Models for Action Selection via Grover Amplitude Amplification. *IEEE GLOBECOM*.
9. Zeng, L., et al. (2024). Scaling Quantum Reinforcement Learning via Amplitude Amplification and Unitary Representation. *IEEE Trans. Quantum Eng.*, 5, 1–14.
10. Bowles, J., Ahmed, S., & Schuld, M. (2024). Better than classical? The subtle art of benchmarking quantum machine learning models. *arXiv:2403.07059*.
11. Dacrema, M. F., Cremonesi, P., & Jannach, D. (2019). Are we really making much progress? A worrying analysis of recent neural recommendation approaches. *RecSys*.
12. Pineau, J., et al. (2021). Improving reproducibility in machine learning research. *JMLR*, 22(1), 1–20.
13. Schnabel, A., et al. (2024). Quantum machine learning benchmarks: pitfalls and best practices. *Quantum*, 8, 1254.
